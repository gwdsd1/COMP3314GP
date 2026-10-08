"""固定 ORL 分支后的有限 Gabor 单因素诊断；所有 probe 成绩仅作开发分析。"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.signal import convolve2d
from threadpoolctl import threadpool_limits

from csa import CSA, PCA, LDA
from gabor import CONFIG, file_hash
from run_available_reproduction import split, nn, transform_batches, lda_coordinates
from run_orl_gray import write_csv
from run_feret import read_csv

BASE=Path(__file__).resolve().parent
PREPARED=BASE/'data/processed/orl_branches_v2'
BRANCHES=['eyes_only','paper_description']
SEEDS=[20,21,22]
# 此有限名单在本轮 Gabor probe 评分前写入 protocol.json；每项只变一个因素。
FEATURES=['baseline','zero_boundary','wrap_boundary','fixed39_support','wavelength2',
          'sigma_pi','jet_l2','sqrt_response','offset11','mean_pool2','tensor_l2','train_channel_rms']
MODEL_VARIANTS=['centered','euclidean','iterations10']


def specification(name):
    s={'base_wavelength':4.,'sigma':float(2*np.pi),'radius':None,'boundary':'reflect',
       'finite_zero_mean':True,'sampling':'stride00','normalization':'none'}
    if name=='zero_boundary':s['boundary']='constant'
    elif name=='wrap_boundary':s['boundary']='wrap'
    elif name=='fixed39_support':s['radius']=19
    elif name=='wavelength2':s['base_wavelength']=2.
    elif name=='sigma_pi':s['sigma']=float(np.pi)
    elif name=='offset11':s['sampling']='stride11'
    elif name=='mean_pool2':s['sampling']='mean2'
    elif name in ['jet_l2','sqrt_response','tensor_l2','train_channel_rms']:s['normalization']=name
    elif name!='baseline':raise ValueError(name)
    return s


def make_kernel(wavelength,theta,s):
    k=2*np.pi/wavelength;sigma=s['sigma']
    radius=s['radius'] if s['radius'] is not None else int(np.ceil(3*sigma/k))
    y,x=np.mgrid[-radius:radius+1,-radius:radius+1]
    values=(k*k/sigma**2)*np.exp(-k*k*(x*x+y*y)/(2*sigma*sigma))*(
        np.exp(1j*k*(x*np.cos(theta)+y*np.sin(theta)))-np.exp(-sigma*sigma/2))
    values-=values.mean()
    return values.astype(np.complex64)


def filtered(images,s,batch=32):
    n,h,w=images.shape;result=np.empty((n,h//2,w//2,40),np.float32)
    for scale in range(5):
        wavelength=s['base_wavelength']*np.sqrt(2)**scale
        kernels=[make_kernel(wavelength,o*np.pi/8,s) for o in range(8)]
        radius=kernels[0].shape[0]//2;shape=(next_fast_len(h+4*radius),next_fast_len(w+4*radius))
        spectra=[fftn(k,s=shape) for k in kernels]
        for start in range(0,n,batch):
            padded=np.pad(images[start:start+batch].astype(np.float32),((0,0),(radius,radius),(radius,radius)),mode=s['boundary'])
            transformed=fftn(padded,s=shape,axes=(-2,-1))
            for o,spectrum in enumerate(spectra):
                full=np.abs(ifftn(transformed*spectrum,axes=(-2,-1))[:,2*radius:2*radius+h,2*radius:2*radius+w])
                if s['sampling']=='stride00':sample=full[:,::2,::2]
                elif s['sampling']=='stride11':sample=full[:,1::2,1::2]
                else:sample=full.reshape(len(full),h//2,2,w//2,2).mean(axis=(2,4))
                result[start:start+len(padded),:,:,scale*8+o]=sample
    return result


def convolution_checks():
    """核索引/边界由独立直接二维卷积验证，不通过识别分数判断正确性。"""
    rng=np.random.default_rng(3414);x=rng.normal(size=(9,7)).astype(np.float32);rows=[]
    for name in ['baseline','zero_boundary','wrap_boundary','fixed39_support','wavelength2','sigma_pi']:
        s=specification(name);k=make_kernel(s['base_wavelength'],np.pi/8,s);r=k.shape[0]//2
        padded=np.pad(x,((r,r),(r,r)),mode=s['boundary']);shape=(next_fast_len(9+4*r),next_fast_len(7+4*r))
        actual=ifftn(fftn(padded,s=shape)*fftn(k,s=shape))[2*r:2*r+9,2*r:2*r+7]
        expected=convolve2d(padded.astype(np.float64),k.astype(np.complex128),mode='valid')
        error=float(np.max(np.abs(actual-expected)));assert error<2e-6
        rows.append({'variant':name,'max_abs_error':error})
    return rows


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    snap=root/'source_snapshot';snap.mkdir();sources=['diagnose_orl_gabor.py','csa.py','gabor.py','run_available_reproduction.py']
    for source in sources:shutil.copy2(BASE/source,snap/source)
    previous=BASE/'results/orl_branches_v3'
    assert json.loads((previous/'verification.json').read_text())['passed']
    protocol={'branches':BRANCHES,'seeds':SEEDS,'features':{n:specification(n) for n in FEATURES},
              'model_variants':MODEL_VARIANTS,'pure_csa_ranks':[15,1,26],'pca_dimension':76,
              'csa_lda_ranks':[4,2,3],'lda_dimension':23,'default_center':False,'default_metric':'cosine',
              'default_lda_scaling':'unit_columns','Tmax':3,'training_only_channel_scale':True,
              'stage1_protocol_sha256':file_hash(previous/'protocol.json'),
              'all_scores':'development diagnostics; probes already observed; no winner promoted as blind reproduction',
              'baseline_gabor':CONFIG,'sources':{n:file_hash(BASE/n) for n in sources},
              'reference':'https://cenl.ucsd.edu/pubs/ieee-1997.pdf',
              'reference_limit':'Wiskott wavelength 4-16 specified for 128x128 images; CSA paper does not resolve ORL rescaling/boundary/support/normalization',
              'convolution_checks':convolution_checks()}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];predictions=[];histories=[];manifests=[];scales=[]
    labels=np.repeat(np.arange(1,41),10);names=[r['file'] for r in read_csv(BASE/'data/annotations/orl_eye_coordinates_v2.csv')]
    feature_folder=root/'features';feature_folder.mkdir()
    for branch in BRANCHES:
        gray=np.load(PREPARED/f'{branch}_seed20.npy');digest=hashlib.sha256(gray.tobytes()).hexdigest()
        baseline=np.load(PREPARED/f'gabor_{digest[:16]}.npy',mmap_mode='r')
        assert all(np.array_equal(gray,np.load(PREPARED/f'{branch}_seed{s}.npy')) for s in SEEDS)
        for variant in FEATURES:
            tick=time.perf_counter();s=specification(variant)
            if variant in ['jet_l2','sqrt_response','tensor_l2','train_channel_rms','baseline']:data=np.asarray(baseline)
            else:data=filtered(gray,s)
            if variant=='jet_l2':data=data/np.maximum(np.linalg.norm(data,axis=-1,keepdims=True),1e-12)
            elif variant=='sqrt_response':data=np.sqrt(data)
            elif variant=='tensor_l2':data=data/np.maximum(np.linalg.norm(data.reshape(len(data),-1),axis=1)[:,None,None,None],1e-12)
            path=feature_folder/f'{branch}_{variant}.npy';np.save(path,data)
            assert data.shape==(400,28,23,40) and np.isfinite(data).all()
            manifests.append({'branch':branch,'variant':variant,'gray_sha256':file_hash(PREPARED/f'{branch}_seed20.npy'),
                              'path':str(path.relative_to(root)),'sha256':file_hash(path),'shape':list(data.shape),
                              'feature_seconds':time.perf_counter()-tick})
            for seed in SEEDS:
                train,probe=split(labels,2,seed);active=data
                if variant=='train_channel_rms':
                    scale=np.sqrt(np.mean(data[train].astype(np.float64)**2,axis=(0,1,2)))
                    active=(data/np.maximum(scale,1e-12)).astype(np.float32)
                    scales.append({'branch':branch,'seed':seed,'scale':scale.tolist(),'train_indices':train.tolist()})
                configs=[('CSA-3','default',False,'cosine',3,'unit_columns'),
                         ('PCA (Gabor)','default',True,'cosine',3,'unit_columns'),
                         ('CSA-3+LDA','default',False,'cosine',3,'unit_columns')]
                if variant=='baseline':
                    configs += [('CSA-3','centered',True,'cosine',3,'unit_columns'),
                                ('CSA-3','euclidean',False,'euclidean',3,'unit_columns'),
                                ('CSA-3','iterations10',False,'cosine',10,'unit_columns')]
                    configs += [('CSA-3+LDA',f'{scaling}_{metric}',False,metric,3,scaling)
                                for scaling,metric in [('unit_columns','euclidean'),('sw_identity','cosine'),('sw_identity','euclidean')]]
                for method,model_variant,center,metric,iterations,scaling in configs:
                    run_start=time.perf_counter()
                    model=PCA(76,center=center,solver='gram') if method=='PCA (Gabor)' else CSA(
                        (4,2,3) if method.endswith('+LDA') else (15,1,26),center=center,max_iter=iterations)
                    model.fit(active[train]);g=transform_batches(model,active,train);p=transform_batches(model,active,probe)
                    if method.endswith('+LDA'):
                        lda=LDA().fit(g,labels[train]);g=lda_coordinates(g,lda,23,scaling);p=lda_coordinates(p,lda,23,scaling)
                    pred,neighbor,distance=nn(g,p,labels[train],metric)
                    key={'branch':branch,'feature_variant':variant,'seed':seed,'method':method,'model_variant':model_variant}
                    rows.append({**key,'center':center,'metric':metric,'Tmax':iterations,'lda_scaling':scaling,
                                 'accuracy_percent':100*float(np.mean(pred==labels[probe])),
                                 'correct':int(np.sum(pred==labels[probe])),'total':len(probe),'seconds':time.perf_counter()-run_start})
                    for i,predicted,j,d in zip(probe,pred,neighbor,distance):
                        predictions.append({**key,'probe_index':int(i),'probe_file':names[i],'true_label':int(labels[i]),
                                            'predicted_label':int(predicted),'neighbor_index':int(train[j]),'neighbor_file':names[train[j]],
                                            'distance':float(d),'correct':int(predicted==labels[i])})
                    if isinstance(model,CSA):histories.append({**key,'initial_sse':model.initial_sse,'history':model.history})
                print(f'{branch} {variant} seed={seed}: CSA={rows[-len(configs)]["accuracy_percent"]:.2f}%',flush=True)
            del data
    write_csv(root/'fixed_results.csv',rows);write_csv(root/'predictions.csv',predictions)
    for name,content in [('feature_manifests',manifests),('training_scales',scales),('csa_history',histories)]:
        (root/f'{name}.json').write_text(json.dumps(content,indent=2),encoding='utf-8')
    summary=[]
    keys=list(dict.fromkeys((r['branch'],r['feature_variant'],r['method'],r['model_variant']) for r in rows))
    for branch,feature,method,model_variant in keys:
        values=[r['accuracy_percent'] for r in rows if (r['branch'],r['feature_variant'],r['method'],r['model_variant'])==(branch,feature,method,model_variant)]
        assert len(values)==3
        summary.append({'branch':branch,'feature_variant':feature,'method':method,'model_variant':model_variant,
                        'mean_percent':float(np.mean(values)),'std_pp':float(np.std(values,ddof=1))})
    (root/'summary.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-started,'summary':summary},indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(14,7),sharex=True)
    for axis,branch in zip(axes,BRANCHES):
        records=[r for r in summary if r['branch']==branch and r['method']=='CSA-3']
        axis.barh(np.arange(len(records)),[r['mean_percent'] for r in records],xerr=[r['std_pp'] for r in records],capsize=2,color='#278d68')
        axis.set(yticks=np.arange(len(records)),yticklabels=[r['feature_variant'] if r['model_variant']=='default' else r['model_variant'] for r in records],
                 title=branch,xlabel='Accuracy (%)',xlim=(0,100));axis.invert_yaxis();axis.axvline(94.4,color='#2878b5',linestyle='--',label='paper 94.4')
        axis.legend(loc='lower left')
    fig.suptitle('ORL Gabor CSA: one factor at a time; development diagnostics, mean +/- sample SD')
    fig.tight_layout();fig.savefig(root/'gabor_comparison.png',dpi=160);plt.close(fig)
    print(f'Finished: {len(rows)} configurations, {len(predictions)} predictions',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=BASE/'results/orl_gabor_diagnosis')
    with threadpool_limits(limits=4):run(parser.parse_args().output)
