"""论文/TA协议审计后的 ORL 训练内选参：保留2/8、共享预处理、NN和Tmax=3。"""
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone

import numpy as np
from PIL import Image,ImageOps
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA,PCA,LDA
from gabor import file_hash
from prepare_available_data import cv2,alignment_matrix
from run_available_reproduction import ORL_CASES,split,transform_batches
from run_orl_branches import points
from run_orl_gray import write_csv
from run_feret import read_csv
from run_shared_protocol import feature_spec
from diagnose_orl_gabor import filtered

BASE=Path(__file__).resolve().parent
ROOT=BASE/'results/orl_protocol_audit'
LABELS=np.repeat(np.arange(1,41),10)
NAMES=[f's{p}/{i}.pgm' for p in range(1,41) for i in range(1,11)]
SEEDS=list(range(20,30))
# 六套 NN/LDA 数值约定对所有方法共用，不为某一方法选专属距离。
POLICIES={f'{scaling}_{name}':{'scaling':scaling,'metric':metric,'lda_center':center}
          for scaling in ['unit_columns','sw_identity']
          for name,metric,center in [('cosine_centered','cosine',True),('cosine_linear','cosine',False),('euclidean','euclidean',True)]}
MODEL_CASES=[('PCA+LDA (Gray)','gray','pca',30,True),
             ('CSA-2+LDA','gray','csa',(11,2),True),('CSA-2+LDA','gray','csa',(8,4),True),
             ('PCA (Gabor)','gabor','pca',74,False),
             ('CSA-3','gabor','csa',(15,1,26),False),('CSA-3','gabor','csa',(10,2,16),False),
             ('PCA+LDA (Gabor)','gabor','pca',33,True),
             ('CSA-3+LDA','gabor','csa',(4,2,3),True),('CSA-3+LDA','gabor','csa',(4,2,4),True)]

def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'protocol.json').exists()
    folder=ROOT/'features';folder.mkdir(exist_ok=True)
    raw=np.stack([np.asarray(Image.open(BASE/'data/raw/orl'/name).convert('L')) for name in NAMES])
    old=read_csv(BASE/'data/annotations/orl_eye_coordinates_v2.csv')
    model=BASE/'data/preprocessing_models/face_detection_yunet_2023mar.onnx'
    assert file_hash(model)=='8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
    # 重新运行标签无关的检测，确认眼点CSV由固定模型/尺度产生。
    detector=cv2.FaceDetectorYN.create(str(model),'',(184,224),score_threshold=.7,nms_threshold=.3,top_k=5000)
    actual=[];eye_rows=[]
    for name,image in zip(NAMES,raw):
        bgr=cv2.cvtColor(cv2.resize(image,(184,224),interpolation=cv2.INTER_CUBIC),cv2.COLOR_GRAY2BGR)
        _,faces=detector.detect(bgr);assert faces is not None
        face=faces[np.argmax(faces[:,-1])]
        eye=face[4:8].reshape(2,2)/2;eye=eye[np.argsort(eye[:,0])];actual.append(eye)
        eye_rows.append({'file':name,'left_x':float(eye[0,0]),'left_y':float(eye[0,1]),
                         'right_x':float(eye[1,0]),'right_y':float(eye[1,1]),
                         'confidence':float(face[-1]),'method':'YuNet2023mar fixed2x'})
    if not (ROOT/'yunet_eye_coordinates.csv').exists():write_csv(ROOT/'yunet_eye_coordinates.csv',eye_rows)
    new=read_csv(ROOT/'yunet_eye_coordinates.csv')
    assert [r['file'] for r in old]==[r['file'] for r in new]==NAMES
    np.testing.assert_allclose(np.stack(actual),points(new),atol=0,rtol=0)
    y,x=np.mgrid[:112,:92];grid=np.column_stack([x.ravel(),y.ravel(),np.ones(112*92)])
    catalogue=[];quality=[];geometries={};masks={}
    for annotation,eyes in [('haar_v2',points(old)),('yunet',points(new))]:
        for distance,height in [(22,22),(20,24),(18,25),(16,26)]:
            target=np.array([[23-distance/2,height],[23+distance/2,height]],dtype=float)
            for padding in ['zero','replicate']:
                name=f'{annotation}_d{distance}_{padding}';images=[];valid=[]
                for i,(image,source) in enumerate(zip(raw,eyes)):
                    matrix=alignment_matrix(source,target)
                    mapped=np.column_stack([source,np.ones(2)])@matrix.T
                    assert np.max(np.abs(mapped-target))<1e-10
                    boundary=cv2.BORDER_CONSTANT if padding=='zero' else cv2.BORDER_REPLICATE
                    warped=cv2.warpAffine(image,matrix,(46,56),flags=cv2.INTER_LINEAR,borderMode=boundary,borderValue=0)
                    # 与原文更接近的尺寸/强度范围归一化及均衡；不额外作z-score。
                    images.append(np.asarray(ImageOps.equalize(Image.fromarray(warped)),dtype=np.float32))
                    coverage=cv2.warpAffine(np.ones_like(image,dtype=np.float32),matrix,(46,56),flags=cv2.INTER_LINEAR)
                    p=grid@matrix.T;retained=np.mean((p[:,0]>=0)&(p[:,0]<=45)&(p[:,1]>=0)&(p[:,1]<=55))
                    quality.append({'geometry':name,'index':i,'file':NAMES[i],'source_retained_fraction':float(retained),
                                    'output_coverage_fraction':float(coverage.mean()),'max_eye_error':float(np.max(np.abs(mapped-target)))})
                    valid.append(coverage)
                array=np.stack(images);gray=folder/f'{name}_gray.npy';np.save(gray,array)
                geometries[name]=array;masks[name]=np.stack(valid)
                for kernel in ['adaptive_zero','fixed39_zero']:
                    path=folder/f'{name}_{kernel}.npy';np.save(path,filtered(array,feature_spec(kernel)))
                    catalogue.append({'id':f'{name}_{kernel}','geometry':name,'eyes':annotation,'target':target.tolist(),
                                      'warp_padding':padding,'kernel':kernel,'gray_path':gray.relative_to(BASE).as_posix(),
                                      'gabor_path':path.relative_to(BASE).as_posix(),'gray_sha256':file_hash(gray),'gabor_sha256':file_hash(path)})
                print(f'Prepared {name}',flush=True)
    # 原冻结流程始终进入深层CV候选，不能因初筛而排除已有方案。
    gray=BASE/'results/shared_protocol_final/step2/valid_zscore_lanczos.npy'
    gabor=BASE/'results/shared_protocol_final/step3/adaptive_zero.npy'
    catalogue.append({'id':'previous_frozen','geometry':'previous_frozen','eyes':'haar_v2','target':[[12,22],[34,22]],
                      'warp_padding':'zero','kernel':'adaptive_zero','gray_path':gray.relative_to(BASE).as_posix(),
                      'gabor_path':gabor.relative_to(BASE).as_posix(),'gray_sha256':file_hash(gray),'gabor_sha256':file_hash(gabor)})
    write_csv(ROOT/'geometry_quality.csv',quality);dump(ROOT/'catalogue.json',catalogue)
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    sources=['audit_orl_protocol.py','csa.py','diagnose_orl_gabor.py','run_shared_protocol.py','prepare_available_data.py','run_available_reproduction.py']
    for source in sources:shutil.copy2(BASE/source,snap/source)
    dump(ROOT/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'seeds':SEEDS,'outer_split':'2 gallery / 8 probe per class, unchanged',
         'eye_sources':['retained Haar v2','YuNet2023mar fixed 2x; external pretrained detector; no CSA/recognition network'],
         'eye_files_sha256':{'old':file_hash(BASE/'data/annotations/orl_eye_coordinates_v2.csv'),'yunet':file_hash(ROOT/'yunet_eye_coordinates.csv')},
         'model_sha256':file_hash(model),'policies':POLICIES,'model_cases':MODEL_CASES,
         'preprocessing_screen':'training-gallery leave-self-out raw gray and adaptive Gabor cosine NN mean; top three geometries; no probe',
         'deep_candidates':'two Gabor supports on top three geometries, plus previous_frozen anchor (seven per seed)',
         'inner_validation':'16 folds; each holds one gallery image for five classes; fit 75 images/40 classes => N-Nc=35; all 80 gallery images validated once',
         'pure_PCA_inner':'rank74 because centered PCA on75 images has rank<=74; final optimized rank<=74; fixed-paper comparison uses76 on80 images',
         'selection':'maximize six-method mean inner accuracy; per method optimize listed ranks/LDA dimension/PCA prefix; ties prefer earlier catalog/policy/case/dimension',
         'shared_per_outer_seed':'all six methods share chosen geometry, Gabor, metric, LDA scale/origin; each seed uses only its own gallery for selection',
         'CSA':{'center':False,'init':'identity','Tmax':3,'epsilon':.001},'PCA_center':True,'LDA':'Fisher unregularized; p<=N-Nc; no ridge',
         'limitations':['historical probes already observed; not a blind study','two gallery images/class limit inner task to one held-out image; small validation sample',
                        'source retention counts raw pixel locations, not semantic face information','YuNet preprocessing is a modern group convention, not a2008 CSA component'],
         'sources_sha256':{n:file_hash(BASE/n) for n in sources},'paper_sha256':file_hash(BASE/'Reconstruction_and_Recognition_of_Tensor-Based_Objects_With_Concurrent_Subspaces_Analysis.pdf')})
    # 小范围图像QA，不根据准确率调整眼点。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sample=[0,6,8,21,78,186,197,258];names=['haar_v2_d22_zero','yunet_d22_zero','yunet_d18_replicate','yunet_d16_replicate']
    fig,axes=plt.subplots(len(sample),4,figsize=(9,14))
    for row,i in enumerate(sample):
        for col,name in enumerate(names):
            axes[row,col].imshow(geometries[name][i],cmap='gray',vmin=0,vmax=255);axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(name,fontsize=8)
            if col==0:axes[row,col].text(-.1,.5,NAMES[i],transform=axes[row,col].transAxes,rotation=90,va='center')
    fig.tight_layout();fig.savefig(ROOT/'crop_comparison.png',dpi=150);plt.close(fig)

def folds(train,seed):
    rng=np.random.default_rng(seed+3314);classes=rng.permutation(40);pairs=train.reshape(40,2)
    for side in [0,1]:
        for start in range(0,40,5):
            validation=pairs[classes[start:start+5],side];learning=train[~np.isin(train,validation)]
            assert len(learning)==75 and len(validation)==5 and len(np.unique(LABELS[learning]))==40
            yield learning,validation

def prefix_nn(gallery,queries,labels,metric):
    """所有前缀维数的NN，数学上与每个维数分别调用cdist相同。"""
    products=np.cumsum(queries[:,None,:]*gallery[None,:,:],axis=-1)
    q=np.cumsum(queries**2,axis=-1)[:,None,:];g=np.cumsum(gallery**2,axis=-1)[None,:,:]
    if metric=='euclidean':distances=q+g-2*products
    else:distances=np.clip(1-products/np.sqrt(np.maximum(q*g,1e-30)),0,2)
    assert np.isfinite(distances).all()
    positions=np.argmin(distances,axis=1)
    # 1D cosine只有正/负号，大量理论并列；采用cdist的确定性并列规则。
    if metric=='cosine':positions[:,0]=np.argmin(cdist(queries[:,:1],gallery[:,:1],metric),axis=1)
    return labels[positions],positions

def project(data,learning,validation,algorithm,dimensions,lda):
    model=PCA(dimensions,center=True,solver='gram') if algorithm=='pca' else CSA(dimensions,center=False,max_iter=3)
    model.fit(data[learning]);g=transform_batches(model,data,learning);q=transform_batches(model,data,validation)
    fisher=LDA().fit(g,LABELS[learning]) if lda else None
    return g,q,fisher,model

def lda_project(g,q,lda,policy):
    basis=lda.basis.copy()
    if policy['scaling']=='unit_columns':basis/=np.linalg.norm(basis,axis=0)
    mean=lda.mean if policy['lda_center'] else 0.
    return (g-mean)@basis,(q-mean)@basis

def screen(catalogue,train):
    results=[]
    for entry in catalogue:
        if entry['kernel']!='adaptive_zero' or entry['geometry']=='previous_frozen':continue
        scores=[]
        for kind in ['gray','gabor']:
            x=np.asarray(np.load(BASE/entry[f'{kind}_path'],mmap_mode='r')[train],dtype=np.float64).reshape(80,-1)
            distances=cdist(x,x,'cosine');np.fill_diagonal(distances,np.inf)
            pred=LABELS[train][np.argmin(distances,axis=1)];scores.append(int(np.sum(pred==LABELS[train])))
        results.append({'geometry':entry['geometry'],'gray_correct':scores[0],'gabor_correct':scores[1],'sum_correct':sum(scores),'total_each':80})
    ranked=sorted(results,key=lambda r:-r['sum_correct'])
    best={r['geometry'] for r in ranked[:3]}
    candidates=[r for r in catalogue if r['geometry'] in best or r['id']=='previous_frozen']
    assert len(candidates)==7
    return results,candidates

def select(seed):
    out=ROOT/f'seed{seed}';out.mkdir(exist_ok=True)
    assert not (out/'selected.json').exists(),'Refusing to replace a completed selection'
    train,probe=split(LABELS,2,seed);catalogue=json.loads((ROOT/'catalogue.json').read_text(encoding='utf-8'))
    screening,candidates=screen(catalogue,train);write_csv(out/'screen.csv',screening)
    dump(out/'selection_inputs.json',{'seed':seed,'outer_gallery_indices':train.tolist(),'outer_probe_indices':probe.tolist(),
         'candidate_ids':[r['id'] for r in candidates],'folds':[{'learning':a.tolist(),'validation':b.tolist()} for a,b in folds(train,seed)]})
    scores=[]
    for entry in candidates:
        arrays={kind:np.load(BASE/entry[f'{kind}_path'],mmap_mode='r') for kind in ['gray','gabor']}
        counts=defaultdict(lambda:0)
        for learning,validation in folds(train,seed):
            for case_index,(method,kind,algorithm,dimensions,use_lda) in enumerate(MODEL_CASES):
                g,q,lda,_=project(arrays[kind],learning,validation,algorithm,dimensions,use_lda)
                for policy_name,policy in POLICIES.items():
                    a,b=lda_project(g,q,lda,policy) if lda else (g,q)
                    pred,_=prefix_nn(a,b,LABELS[learning],policy['metric'])
                    ds=range(1,a.shape[1]+1) if use_lda else ([16,32,48,64,74] if algorithm=='pca' else [a.shape[1]])
                    for d in ds:counts[(policy_name,case_index,d)]+=int(np.sum(pred[:,d-1]==LABELS[validation]))
        scores.extend({'candidate':entry['id'],'policy':policy,'case_index':case,'method':MODEL_CASES[case][0],
                       'dimension':d,'correct':correct,'validation_count':80,'accuracy_percent':100*correct/80}
                      for (policy,case,d),correct in counts.items())
        print(f'Seed {seed}: inner validation complete {entry["id"]}',flush=True)
    objectives=[];best_methods={}
    for entry in candidates:
        for policy in POLICIES:
            chosen=[]
            for method in [c[0] for c in ORL_CASES]:
                pool=[r for r in scores if r['candidate']==entry['id'] and r['policy']==policy and r['method']==method]
                winner=max(pool,key=lambda r:(r['correct'],-r['case_index'],-r['dimension']));chosen.append(winner)
            key=(entry['id'],policy);best_methods[key]=chosen
            objectives.append({'candidate':entry['id'],'policy':policy,'sum_correct':sum(r['correct'] for r in chosen),
                               'mean_percent':100*sum(r['correct'] for r in chosen)/(6*80)})
    winner=max(objectives,key=lambda r:r['sum_correct']);key=(winner['candidate'],winner['policy'])
    dump(out/'selected.json',{'seed':seed,'selection':winner,'shared_input':next(e for e in candidates if e['id']==winner['candidate']),
                            'shared_policy':POLICIES[winner['policy']],'methods':best_methods[key],
                            'selected_before_outer_prediction':True})
    write_csv(out/'validation_scores.csv',scores);write_csv(out/'shared_objectives.csv',objectives)
    print(f'Seed {seed} selected {winner}',flush=True)

def evaluate():
    assert all((ROOT/f'seed{s}/selected.json').exists() for s in SEEDS)
    # 所有种子选择均在外层probe评分前冻结；每个选择只使用其自身80张gallery。
    freeze={str(s):file_hash(ROOT/f'seed{s}/selected.json') for s in SEEDS};dump(ROOT/'selection_freeze.json',freeze)
    out=ROOT/'evaluation';out.mkdir(exist_ok=False);(out/'embeddings').mkdir()
    rows=[];predictions=[];validation_predictions=[];histories=[]
    for seed in SEEDS:
        selected=json.loads((ROOT/f'seed{seed}/selected.json').read_text(encoding='utf-8'));entry=selected['shared_input'];policy=selected['shared_policy']
        arrays={k:np.load(BASE/entry[f'{k}_path'],mmap_mode='r') for k in ['gray','gabor']};train,probe=split(LABELS,2,seed)
        optimized=[]
        for row in selected['methods']:
            method,kind,algorithm,dimensions,lda=MODEL_CASES[row['case_index']]
            optimized.append((method,kind,algorithm,dimensions,row['dimension'] if lda else None,
                              next(c[-1] for c in ORL_CASES if c[0]==method),row['dimension']))
        fixed=[(*c,None) for c in ORL_CASES]
        for profile,cases in [('paper_dimensions',fixed),('training_selected',optimized)]:
            for number,(method,kind,algorithm,dimensions,lda_dim,paper,prefix) in enumerate(cases):
                g,q,lda,model=project(arrays[kind],train,probe,algorithm,dimensions,lda_dim is not None)
                if lda:
                    g,q=lda_project(g,q,lda,policy);g=g[:,:lda_dim];q=q[:,:lda_dim]
                elif prefix is not None:g=g[:,:prefix];q=q[:,:prefix]
                distance=cdist(q,g,policy['metric']);assert np.isfinite(distance).all();neighbor=np.argmin(distance,axis=1);pred=LABELS[train[neighbor]]
                identifier=f'{seed}_{profile}_{number}';np.savez_compressed(out/'embeddings'/f'{identifier}.npz',gallery=g,probe=q,train_indices=train,probe_indices=probe)
                key={'seed':seed,'profile':profile,'method':method}
                rows.append({**key,'candidate':entry['id'],'policy':selected['selection']['policy'],'pre_dims':str(dimensions),
                             'lda_dim':lda_dim,'feature_dim':g.shape[1],'correct':int(np.sum(pred==LABELS[probe])),
                             'probe_count':320,'accuracy_percent':100*float(np.mean(pred==LABELS[probe])),'paper_percent':paper,
                             'input_sha256':entry[f'{kind}_sha256'],'embedding':f'embeddings/{identifier}.npz'})
                predictions.extend({**key,'probe_index':int(i),'probe':NAMES[i],'true':int(LABELS[i]),'predicted':int(p),
                                    'neighbor_index':int(train[j]),'nearest_gallery':NAMES[train[j]],'distance':float(distance[n,j]),'correct':int(p==LABELS[i])}
                                   for n,(i,p,j) in enumerate(zip(probe,pred,neighbor)))
                if algorithm=='csa':histories.append({**key,'initial_sse':model.initial_sse,'history':model.history})
            print(f'{seed} {profile}: '+', '.join(f'{r["method"]}={r["accuracy_percent"]:.2f}' for r in rows[-6:]),flush=True)
        # 重新核对被选模型的80条训练内预测，而不是仅接受选参分数。
        for case in optimized:
            method,kind,algorithm,dimensions,lda_dim,_,prefix=case
            total=0
            for learning,validation in folds(train,seed):
                g,q,lda,_=project(arrays[kind],learning,validation,algorithm,dimensions,lda_dim is not None)
                if lda:g,q=lda_project(g,q,lda,policy);g=g[:,:lda_dim];q=q[:,:lda_dim]
                elif prefix is not None:g=g[:,:prefix];q=q[:,:prefix]
                distance=cdist(q,g,policy['metric']);neighbor=np.argmin(distance,axis=1);pred=LABELS[learning[neighbor]]
                total+=int(np.sum(pred==LABELS[validation]))
                validation_predictions.extend({'seed':seed,'method':method,'validation_index':int(i),'learning_indices':json.dumps(learning.tolist()),
                    'true':int(LABELS[i]),'predicted':int(p),'neighbor_index':int(learning[j]),'correct':int(p==LABELS[i])}
                    for i,p,j in zip(validation,pred,neighbor))
            assert total==next(r['correct'] for r in selected['methods'] if r['method']==method)
    write_csv(out/'fixed_results.csv',rows);write_csv(out/'predictions.csv',predictions);write_csv(out/'validation_predictions.csv',validation_predictions)
    dump(out/'csa_history.json',histories)
    summary=[]
    for profile in ['paper_dimensions','training_selected']:
        for method in [r[0] for r in ORL_CASES]:
            group=[r for r in rows if r['profile']==profile and r['method']==method];values=[r['accuracy_percent'] for r in group]
            summary.append({'profile':profile,'method':method,'mean_percent':float(np.mean(values)),'std_pp':float(np.std(values,ddof=1)),
                            'min_percent':min(values),'max_percent':max(values),'paper_percent':group[0]['paper_percent'],
                            'difference_pp':float(np.mean(values)-group[0]['paper_percent'])})
    dump(out/'summary.json',summary)

def verify():
    out=ROOT/'evaluation';scores=read_csv(out/'fixed_results.csv');predictions=read_csv(out/'predictions.csv');validation=read_csv(out/'validation_predictions.csv')
    assert len(scores)==120 and len(predictions)==38400 and len(validation)==4800
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    for name,sha in protocol['sources_sha256'].items():assert file_hash(ROOT/'source_snapshot'/name)==sha
    catalogue=json.loads((ROOT/'catalogue.json').read_text(encoding='utf-8'))
    for e in catalogue:
        for k in ['gray','gabor']:assert file_hash(BASE/e[f'{k}_path'])==e[f'{k}_sha256']
    for seed in SEEDS:
        folder=ROOT/f'seed{seed}';selected=json.loads((folder/'selected.json').read_text(encoding='utf-8'))
        assert file_hash(folder/'selected.json')==json.loads((ROOT/'selection_freeze.json').read_text())[str(seed)]
        inputs=json.loads((folder/'selection_inputs.json').read_text());train,probe=split(LABELS,2,seed)
        np.testing.assert_array_equal(train,inputs['outer_gallery_indices']);np.testing.assert_array_equal(probe,inputs['outer_probe_indices'])
        seen=[]
        for fold in inputs['folds']:
            a,b=fold['learning'],fold['validation'];assert len(a)==75 and len(b)==5 and set(a).isdisjoint(b) and set(a+b)==set(train)
            assert len(np.unique(LABELS[a]))==40 and 75-40>=33;seen.extend(b)
        assert sorted(seen)==sorted(train.tolist()) and set(seen).isdisjoint(probe)
        vs=read_csv(folder/'validation_scores.csv');objectives=read_csv(folder/'shared_objectives.csv')
        best=max(objectives,key=lambda r:int(r['sum_correct']))
        assert best['candidate']==selected['selection']['candidate'] and best['policy']==selected['selection']['policy']
        for objective in objectives:
            total=0
            for method in [r[0] for r in ORL_CASES]:
                subset=[r for r in vs if r['candidate']==objective['candidate'] and r['policy']==objective['policy'] and r['method']==method]
                total+=max(int(r['correct']) for r in subset)
            assert total==int(objective['sum_correct'])
        group=[r for r in scores if int(r['seed'])==seed]
        assert len({r['candidate'] for r in group})==len({r['policy'] for r in group})==1
        for s in selected['methods']:
            r=[r for r in validation if int(r['seed'])==seed and r['method']==s['method']]
            assert len(r)==80 and sorted(int(x['validation_index']) for x in r)==sorted(train.tolist())
            assert sum(int(x['correct']) for x in r)==s['correct']
            for x in r:
                learning=json.loads(x['learning_indices']);i=int(x['validation_index']);j=int(x['neighbor_index'])
                assert i not in learning and j in learning and set(learning).issubset(train)
                assert int(x['predicted'])==LABELS[j] and int(x['true'])==LABELS[i] and int(x['correct'])==int(LABELS[j]==LABELS[i])
    pg=defaultdict(list)
    for r in predictions:pg[(r['seed'],r['profile'],r['method'])].append(r)
    for s in scores:
        emb=np.load(out/s['embedding']);g,q=emb['gallery'],emb['probe'];train,probe=emb['train_indices'],emb['probe_indices']
        selected=json.loads((ROOT/f'seed{s["seed"]}/selected.json').read_text(encoding='utf-8'));metric=selected['shared_policy']['metric']
        if metric=='cosine':distance=1-(q/np.linalg.norm(q,axis=1)[:,None])@(g/np.linalg.norm(g,axis=1)[:,None]).T
        else:distance=np.sqrt(np.maximum(0,np.sum(q*q,axis=1)[:,None]+np.sum(g*g,axis=1)[None,:]-2*q@g.T))
        nearest=np.argmin(distance,axis=1);records=pg[(s['seed'],s['profile'],s['method'])];assert len(records)==320
        correct=0
        for n,(i,j,r) in enumerate(zip(probe,nearest,records)):
            assert int(r['probe_index'])==i and int(r['neighbor_index'])==train[j] and int(r['predicted'])==LABELS[train[j]]
            assert r['probe']==NAMES[i] and r['nearest_gallery']==NAMES[train[j]]
            assert abs(float(r['distance'])-distance[n,j])<1e-5*max(1,distance[n,j])
            flag=int(LABELS[i]==LABELS[train[j]]);assert int(r['correct'])==flag;correct+=flag
        assert correct==int(s['correct']) and abs(100*correct/320-float(s['accuracy_percent']))<1e-10
    for h in json.loads((out/'csa_history.json').read_text()):
        assert len(h['history'])==3
        errors=[h['initial_sse']]+[r['sse'] for r in h['history']]
        assert all(b<=a+1e-8*max(1,a) for a,b in zip(errors,errors[1:]))
    summary=json.loads((out/'summary.json').read_text())
    for s in summary:
        vals=[float(r['accuracy_percent']) for r in scores if r['method']==s['method'] and r['profile']==s['profile']]
        assert len(vals)==10 and abs(np.mean(vals)-s['mean_percent'])<1e-10 and abs(np.std(vals,ddof=1)-s['std_pp'])<1e-10
    # 实际输入两样本重算，确保新核缓存和配置一致。
    used={r['candidate'] for r in scores}
    for e in catalogue:
        if e['id'] not in used:continue
        g=np.load(BASE/e['gabor_path'],mmap_mode='r');x=np.load(BASE/e['gray_path'],mmap_mode='r')
        np.testing.assert_allclose(filtered(x[[0,59]],feature_spec(e['kernel'])),g[[0,59]],rtol=2e-6,atol=2e-6)
    result={'passed':True,'model_runs':120,'outer_predictions':38400,'selected_inner_predictions':4800,
            'checks':['raw/feature/source hashes','16 inner folds restricted to own80gallery','N-Nc=35 in inner LDA',
                      'all shared objectives independently recomputed','selected inner scores match cdist rerun',
                      'shared inputs and NN/LDA policy per seed','independent matrix NN checks all outer predictions',
                      'three CSA iterations and monotone residual','ten-seed statistics','actual feature cache regeneration'],
            'audit_code_sha256':file_hash(BASE/'audit_orl_protocol.py')}
    dump(ROOT/'verification.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','select','evaluate','verify'],required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args();ROOT=args.output.resolve()
    assert ROOT.is_relative_to(BASE),'Output must stay in the project directory'
    with threadpool_limits(limits=4):
        if args.stage=='prepare':prepare()
        elif args.stage=='select':
            for seed in args.seeds:select(seed)
        elif args.stage=='evaluate':evaluate()
        else:verify()
