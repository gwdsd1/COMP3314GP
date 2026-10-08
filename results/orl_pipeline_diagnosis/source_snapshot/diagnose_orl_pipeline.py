"""固定 ORL 2/8 划分与论文维数，定位预处理/初始化；所有探查仅作诊断。"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps,ImageDraw
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA,PCA,LDA
from gabor import features,file_hash
from run_available_reproduction import ORL_CASES,split,lda_coordinates
from run_orl_gray import write_csv
from prepare_available_data import alignment_matrix,TARGET_EYES
from run_feret import read_csv

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'.deps'))
import cv2


def reference_fit(data,ranks,init='identity',seed=3314):
    """独立逐样本矩阵/einsum 更新，固定三轮；不调用主 CSA 的模式积或展开。"""
    x=np.asarray(data,dtype=np.float64);rng=np.random.default_rng(seed)
    def leading(c,r):return np.linalg.eigh((c+c.T)/2)[1][:,-r:]
    bases=[]
    for mode,(m,r) in enumerate(zip(x.shape[1:],ranks)):
        if init=='identity':u=np.eye(m,r)
        elif init=='random':u=np.linalg.qr(rng.normal(size=(m,r)))[0]
        elif init=='hosvd':
            moved=np.moveaxis(x,mode+1,1).reshape(len(x),m,-1)
            u=leading(sum(sample@sample.T for sample in moved),r)
        else:raise ValueError(init)
        bases.append(u)
    errors=[]
    for iteration in range(3):
        for mode,r in enumerate(ranks):
            covariance=np.zeros((x.shape[mode+1],)*2)
            for sample in x:
                if x.ndim==3:
                    b=sample@bases[1] if mode==0 else sample.T@bases[0]
                elif mode==0:b=np.einsum('abc,bj,ck->ajk',sample,bases[1],bases[2],optimize=True).reshape(x.shape[1],-1)
                elif mode==1:b=np.einsum('abc,ai,ck->bik',sample,bases[0],bases[2],optimize=True).reshape(x.shape[2],-1)
                else:b=np.einsum('abc,ai,bj->cij',sample,bases[0],bases[1],optimize=True).reshape(x.shape[3],-1)
                covariance+=b@b.T
            bases[mode]=leading(covariance,r)
        core=reference_transform(x,bases)
        errors.append(max(0.,float(np.sum(x*x)-np.sum(core*core))))
    assert np.max(np.diff(errors))<=max(errors)*1e-10+1e-6
    for u in bases:np.testing.assert_allclose(u.T@u,np.eye(u.shape[1]),atol=1e-11)
    return bases,errors


def reference_transform(x,bases):
    if x.ndim==3:return np.einsum('nab,ai,bj->nij',x,*bases,optimize=True)
    return np.einsum('nabc,ai,bj,ck->nijk',x,*bases,optimize=True)


def evaluate(g,p,labels,train,probe,lda_dim,metric,scaling='unit_columns'):
    diagnostics={}
    if lda_dim:
        lda=LDA().fit(g,labels[train]);diagnostics={'sw_rank':lda.sw_rank,'sw_condition':float(np.linalg.cond(lda.sw))}
        # Independent symmetric generalized eigensolver, full-rank real-data Sw.
        values,vectors=eigh(lda.sb,lda.sw)
        np.testing.assert_allclose(lda.eigenvalues[:lda_dim],values[-lda_dim:][::-1],rtol=1e-7,atol=1e-7)
        expected=vectors[:,-lda_dim:][:,::-1]
        q=np.linalg.qr(expected)[0];actual=np.linalg.qr(lda.basis[:,:lda_dim])[0]
        np.testing.assert_allclose(q@q.T,actual@actual.T,rtol=1e-6,atol=1e-7)
        residual=lda.sb@lda.basis[:,:lda_dim]-lda.sw@lda.basis[:,:lda_dim]@np.diag(lda.eigenvalues[:lda_dim])
        diagnostics['lda_relative_residual']=float(np.linalg.norm(residual)/(np.linalg.norm(lda.sb@lda.basis[:,:lda_dim])+1e-30))
        g,p=lda_coordinates(g,lda,lda_dim,scaling),lda_coordinates(p,lda,lda_dim,scaling)
    distances=cdist(p,g,metric);assert np.isfinite(distances).all()
    neighbor=np.argmin(distances,axis=1);predicted=labels[train[neighbor]]
    return int(np.sum(predicted==labels[probe])),predicted,train[neighbor],diagnostics


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir()
    source=['diagnose_orl_pipeline.py','csa.py','gabor.py','prepare_available_data.py','run_available_reproduction.py']
    for name in source:shutil.copy2(BASE/name,snapshot/name)
    old=BASE/'data/processed/available';labels=np.load(old/'orl_labels.npy');eyes=read_csv(BASE/'data/annotations/orl_eye_coordinates.csv')
    raw=np.stack([np.asarray(Image.open(BASE/'data/raw/orl'/r['file']).convert('L')) for r in eyes])
    resized=np.stack([np.asarray(Image.fromarray(x).resize((46,56),Image.Resampling.BILINEAR)) for x in raw])
    resized_eq=np.stack([np.asarray(ImageOps.equalize(Image.fromarray(x))) for x in resized])
    warped=[];support=[]
    for x,r in zip(raw,eyes):
        points=np.array([[float(r[f'{side}_x']),float(r[f'{side}_y'])] for side in ['left','right']])
        matrix=alignment_matrix(points)
        warped.append(cv2.warpAffine(x,matrix,(46,56),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0))
        mask=cv2.warpAffine(np.ones_like(x,dtype=np.float32),matrix,(46,56),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        support.append(float(mask.mean()))
    warped=np.stack(warped);aligned=np.load(old/'orl_aligned.npy')
    np.testing.assert_array_equal(aligned,np.stack([np.asarray(ImageOps.equalize(Image.fromarray(x))) for x in warped]))
    geometries={'aligned_equalized':aligned,'resize_equalized':resized_eq,'aligned_no_equalize':warped,'resize_no_equalize':resized}
    canvas=Image.new('RGB',(4*156,8*134),'white');draw=ImageDraw.Draw(canvas)
    for column,(name,data) in enumerate(geometries.items()):
        draw.text((column*156,3),name,fill='black')
        for row,index in enumerate([0,8,10,30,40,130,210]):
            canvas.paste(Image.fromarray(data[index].astype(np.uint8)).resize((92,112)),(column*156,22+row*145))
            draw.text((column*156,135+row*145),eyes[index]['file'],fill='black')
    canvas.save(output/'preprocessing_comparison.png')
    protocol={'scope':'paired diagnostic, not independently selected new main result; previous probes already observed',
              'seeds':[20,21,22],'split':'2 training/gallery +8 probe per person, unchanged',
              'known_difference':'paper ORL paragraph states resize/equalization; instructor additionally requires eye alignment',
              'features':'same five-scale/eight-orientation Gabor; fixed base4 linear magnitude; no L2 or sqrt',
              'initialization':'main identity truncated; independent reference includes train-only mode-covariance HOSVD orthogonal bases',
              'source_sha256':{name:file_hash(BASE/name) for name in source},'eye_sha256':file_hash(BASE/'data/annotations/orl_eye_coordinates.csv'),
              'support_fraction_mean':float(np.mean(support)),'support_fraction_min':float(np.min(support))}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];predictions=[];comparisons=[]
    for geometry,gray in geometries.items():
        if geometry=='aligned_equalized':gabor=np.load(old/'orl_gabor_full.npy',mmap_mode='r')[:,::2,::2]
        elif geometry=='resize_equalized':gabor=np.load(old/'orl_resize_gabor.npy',mmap_mode='r')
        else:gabor=features(gray,step=2)
        assert gabor.shape==(400,28,23,40)
        for seed in protocol['seeds']:
            train,probe=split(labels,2,seed)
            for case in ORL_CASES:
                method,kind,algorithm,dims,lda_dim,paper=case
                data=np.asarray(gray if kind=='gray' else gabor,dtype=np.float64)
                configs=[('standard',algorithm=='pca','identity','unit_columns','euclidean' if kind=='gray' and algorithm=='csa' else 'cosine')]
                # One factor changes, fixed before this diagnostic's scoring.
                if algorithm=='csa':configs.append(('hosvd',False,'hosvd','unit_columns',configs[0][-1]))
                for name,center,init,scaling,metric in configs:
                    training=data[train];queries=data[probe]
                    if algorithm=='csa' and init=='hosvd':
                        bases,errors=reference_fit(training,dims,init)
                        g,p=reference_transform(training,bases).reshape(80,-1),reference_transform(queries,bases).reshape(320,-1)
                    else:
                        model=(PCA(dims,center=center,solver='gram') if algorithm=='pca' else CSA(dims,center=center)).fit(training)
                        g,p=model.transform(training).reshape(80,-1),model.transform(queries).reshape(320,-1)
                        errors=[r['sse'] for r in model.history] if algorithm=='csa' else []
                        if algorithm=='csa' and seed==20 and geometry in ['aligned_equalized','resize_equalized']:
                            reference,ref_errors=reference_fit(training,dims)
                            for a,b in zip(model.bases,reference):np.testing.assert_allclose(a@a.T,b@b.T,rtol=1e-7,atol=1e-8)
                            np.testing.assert_allclose(errors,ref_errors,rtol=1e-10,atol=1e-4)
                            q=reference_transform(queries,reference).reshape(320,-1)
                            np.testing.assert_allclose(cdist(p,g),cdist(q,reference_transform(training,reference).reshape(80,-1)),rtol=1e-8,atol=1e-6)
                            comparisons.append({'geometry':geometry,'method':method,'seed':seed,'independent_projector_and_distance_match':True})
                    correct,predicted,nearest,diagnostics=evaluate(g,p,labels,train,probe,lda_dim,metric,scaling)
                    row={'geometry':geometry,'seed':seed,'method':method,'initialization':init,'center':center,'metric':metric,'scaling':scaling,
                         'pre_dims':str(dims),'lda_dim':lda_dim or '', 'correct':correct,'probe_count':320,'accuracy_percent':100*correct/320,
                         'paper_percent':paper,'training_sse':errors[-1] if errors else '',**diagnostics}
                    rows.append(row)
                    predictions.extend({'geometry':geometry,'seed':seed,'method':method,'initialization':init,'probe_index':int(i),'true':int(labels[i]),
                                        'predicted':int(v),'nearest_gallery_index':int(j),'correct':int(v==labels[i])} for i,v,j in zip(probe,predicted,nearest))
                    print(f'{geometry} seed={seed} {method} init={init}: {row["accuracy_percent"]:.2f}%',flush=True)
        del gabor
    # CSV accepts heterogeneous diagnostics via a common field list.
    fieldnames=list(dict.fromkeys(k for r in rows for k in r))
    write_csv(output/'fixed_results.csv',[{k:r.get(k,'') for k in fieldnames} for r in rows]);write_csv(output/'predictions.csv',predictions)
    summary=[]
    for geo,method,init in dict.fromkeys((r['geometry'],r['method'],r['initialization']) for r in rows):
        rs=[r for r in rows if (r['geometry'],r['method'],r['initialization'])==(geo,method,init)]
        values=[r['accuracy_percent'] for r in rs]
        summary.append({'geometry':geo,'method':method,'initialization':init,'mean_percent':float(np.mean(values)),'std_pp':float(np.std(values,ddof=1)),
                        'paper_percent':rs[0]['paper_percent']})
    result={'summary':summary,'independent_checks':comparisons,'limits':'diagnostic probe evaluations cannot be reused as blind test-selected reproduction'}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE/'results/orl_pipeline_diagnosis')
    with threadpool_limits(limits=4):run(parser.parse_args().output)
