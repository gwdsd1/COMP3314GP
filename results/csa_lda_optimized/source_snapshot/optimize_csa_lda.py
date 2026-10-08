"""ORL 训练内优化：普通LDA与协方差收缩扩展分开报告，保持TA协议。

只改独立实验入口；复用已有对齐、CSA和Gabor基础函数。所有选择仅用各自
80张gallery，十个种子全部冻结后才允许外层评分。历史probe已观察，非盲测。
"""
import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.signal import convolve2d
from scipy.fft import fftn,ifftn,next_fast_len
from threadpoolctl import threadpool_limits

from csa import LDA
from gabor import file_hash
from audit_orl_protocol import BASE,LABELS,NAMES,SEEDS,POLICIES,folds,split,prefix_nn
from diagnose_csa_lda import fit,classify,selected
from diagnose_orl_gabor import filtered,make_kernel
from run_shared_protocol import feature_spec
from run_orl_gray import write_csv

ROOT=BASE/'results/csa_lda_optimized'
SOURCE=BASE/'results/orl_protocol_audit'
ALPHAS=[0.,.01,.05,.15,.35,.65,1.]
NORMS=['none','channel_rms','sqrt']
# 前四个是固定论文前级；追加案例全部满足内层75-40=35。
CASES=[('PCA+LDA (Gray)','gray','pca',30,True,18),
       ('CSA-2+LDA','gray','csa',(11,2),False,16),
       ('PCA+LDA (Gabor)','gabor','pca',33,True,25),
       ('CSA-3+LDA','gabor','csa',(4,2,3),False,23),
       ('CSA-2+LDA','gray','csa',(8,4),False,16),
       ('CSA-2+LDA','gray','csa',(11,3),False,16),
       ('CSA-2+LDA','gray','csa',(11,2),True,16),
       ('CSA-3+LDA','gabor','csa',(4,2,4),False,23),
       ('CSA-3+LDA','gabor','csa',(3,3,3),False,23),
       ('CSA-3+LDA','gabor','csa',(4,2,3),True,23)]
METHODS=[c[0] for c in CASES[:4]]
SOURCES=['optimize_csa_lda.py','csa.py','gabor.py','audit_orl_protocol.py',
         'diagnose_csa_lda.py','diagnose_orl_gabor.py','run_shared_protocol.py','run_available_reproduction.py']

def dump(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def snapshot(folder):
    folder.mkdir()
    for name in SOURCES:shutil.copy2(BASE/name,folder/name)
    return {name:file_hash(folder/name) for name in SOURCES}

def fisher(g,y,alpha,check=False):
    """Sw(a)=(1-a)Sw+a*tr(Sw)/p I；a=0严格调用原LDA。

    alpha由训练CV决定，是手选网格的收缩Fisher，不称Ledoit-Wolf自动估计器。
    """
    base=LDA().fit(g,y)
    if alpha==0:
        if check:
            ev,vec=eigh(base.sb,base.sw);ev=ev[::-1];vec=vec[:,::-1][:,:base.max_components]
            np.testing.assert_allclose(ev[:base.max_components],base.eigenvalues,rtol=1e-7,atol=1e-5)
            cosine=np.abs(np.sum((vec/np.linalg.norm(vec,axis=0))*(base.basis/np.linalg.norm(base.basis,axis=0)),axis=0))
            assert np.min(cosine)>1-1e-6
        return base
    p=g.shape[1];base.reg_sw=(1-alpha)*base.sw+alpha*np.trace(base.sw)/p*np.eye(p)
    values,basis=eigh(base.sb,base.reg_sw);base.basis=basis[:,::-1][:,:base.max_components]
    base.eigenvalues=values[::-1][:base.max_components]
    if check:
        np.testing.assert_allclose(base.basis.T@base.reg_sw@base.basis,np.eye(base.max_components),atol=1e-8,rtol=1e-8)
        residual=np.linalg.norm(base.sb@base.basis-(base.reg_sw@base.basis)*base.eigenvalues)/max(1,np.linalg.norm(base.sb@base.basis))
        assert residual<1e-8
        # 独立对称平方根白化，不依赖上面的广义特征分解。
        v,u=eigh(base.reg_sw);whitener=(u/np.sqrt(v))@u.T
        ev,z=eigh(whitener@base.sb@whitener);w=whitener@z[:,::-1][:,:base.max_components]
        np.testing.assert_allclose(ev[::-1][:base.max_components],base.eigenvalues,rtol=1e-6,atol=1e-5)
        # 重根可旋转，以全子空间及实际特征方程验证，而非强制逐向量同符号。
        q1=np.linalg.qr(w)[0];q2=np.linalg.qr(base.basis)[0]
        np.testing.assert_allclose(q1@q1.T,q2@q2.T,rtol=1e-7,atol=1e-7)
    return base

def coordinates(g,q,lda,policy):
    basis=lda.basis.copy()
    if policy['scaling']=='unit_columns':basis/=np.linalg.norm(basis,axis=0)
    mean=lda.mean if policy['lda_center'] else 0.
    return (g-mean)@basis,(q-mean)@basis

def load(entry):return {k:np.load(BASE/entry[f'{k}_path'],mmap_mode='r') for k in ['gray','gabor']}

def normalize(data,norm,learning):
    if norm=='none':return data,None
    result=data.copy()
    if norm=='sqrt':result['gabor']=np.sqrt(data['gabor']);return result,None
    # 只用当前学习折估计40通道RMS，验证图/probe不参与估计。
    scale=np.sqrt(np.mean(np.asarray(data['gabor'][learning],dtype=np.float64)**2,axis=(0,1,2)))
    result['gabor']=(data['gabor']/np.maximum(scale,1e-12)).astype(np.float32)
    return result,scale

def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'features').mkdir()
    hashes=snapshot(ROOT/'source_snapshot')
    dump(ROOT/'protocol.json',{'seeds':SEEDS,'Tmax':3,'gallery_per_class':2,'probe_per_class':8,
        'gray_shape':[56,46],'gabor_shape':[28,23,40],'Gabor_scales':5,'Gabor_orientations':8,
        'inner':'16folds75learning/5validation, all40classes retained; each gallery image validated once',
        'alpha_grid':ALPHAS,'cases':CASES,'normalizations':NORMS,'policies':POLICIES,
        'LDA_pre_limit':'inner N-Nc=35; outer N-Nc=40, retained for shrinkage too',
        'selection':'maximize equal mean of two CSA+LDA gallery CV scores; shared input/normalization/distance for all four methods',
        'dimensions':'prefix2..available; ties prefer paper LDA dimension, then smaller dimension; 1D cosine excluded',
        'tie_order':'original case before added case; alpha0 before shrinkage; original input before screen winner',
        'profiles':['regularization_only','optimized_ordinary_LDA','optimized_shrinkage_LDA','shrinkage_ablation_alpha0'],
        'fixed_profile':'original per-seed input+policy and original pre-dimensions; select alpha/LDA dimension only',
        'joint_screen':'all catalogue entries and three norms, raw train leave-self-out Gray/Gabor mean; best per norm plus each original-input norm anchor, at most6',
        'not_blind':'historical probes have been observed; no current probe scores or labels enter selection objective',
        'random_indices':'NumPy default_rng seed20..29, unchanged; never select or discard splits by accuracy',
        'extension':'positive alpha is a covariance-shrinkage Fisher extension, not original CSA+ordinaryLDA',
        'references':{'shrinkage':'https://www.ledoit.net/ole1a.pdf','Gabor':'https://cenl.ucsd.edu/pubs/ieee-1997.pdf',
                      'normalization_distance':'https://www.cs.colostate.edu/evalfacerec/algorithms/version5/faceIdUsersGuide.pdf'},
        'source_snapshot_sha256':hashes,'previous_catalogue_sha256':file_hash(SOURCE/'catalogue.json')})
    catalogue=read(SOURCE/'catalogue.json')
    geometries=['yunet_d22_zero','yunet_d20_replicate','yunet_d18_replicate','yunet_d16_replicate']
    specs={}
    for name in ['adaptive_reflect','fixed39_reflect','mean2_zero','sigma_pi_zero','wavelength2_zero']:
        s=feature_spec(name if 'reflect' in name else 'adaptive_zero')
        if name=='mean2_zero':s['sampling']='mean2'
        elif name=='sigma_pi_zero':s['sigma']=float(np.pi)
        elif name=='wavelength2_zero':s['base_wavelength']=2.
        specs[name]=s
    convolution=[]
    for geometry in geometries:
        base=next(e for e in catalogue if e['geometry']==geometry)
        gray=np.load(BASE/base['gray_path'],mmap_mode='r')
        for name,s in specs.items():
            path=ROOT/'features'/f'{geometry}_{name}.npy';np.save(path,filtered(gray,s))
            assert np.load(path,mmap_mode='r').shape==(400,28,23,40)
            entry={**base,'id':f'{geometry}_{name}','kernel':name,'feature_spec':s,
                   'gabor_path':path.relative_to(BASE).as_posix(),'gabor_sha256':file_hash(path)}
            catalogue.append(entry)
        print(f'Prepared5 Gabor alternatives: {geometry}',flush=True)
    # 实际新核/补边按全部40通道独立直接卷积核验；均值池化另核采样。
    rng=np.random.default_rng(3314);x=rng.normal(size=(10,8)).astype(np.float32)
    for name,s in specs.items():
        worst=0.
        for scale in range(5):
            for o in range(8):
                k=make_kernel(s['base_wavelength']*np.sqrt(2)**scale,o*np.pi/8,s);r=k.shape[0]//2
                padded=np.pad(x,((r,r),(r,r)),mode=s['boundary']);shape=(next_fast_len(10+4*r),next_fast_len(8+4*r))
                actual=np.abs(ifftn(fftn(padded,s=shape)*fftn(k,s=shape))[2*r:2*r+10,2*r:2*r+8])
                expected=np.abs(convolve2d(padded.astype(float),k.astype(complex),mode='valid'))
                worst=max(worst,float(np.max(np.abs(actual-expected))))
                if s['sampling']=='mean2':
                    a=actual.reshape(5,2,4,2).mean((1,3));b=sum(expected[j::2,i::2] for j in [0,1] for i in [0,1])/4
                else:a=actual[::2,::2];b=expected[::2,::2]
                np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)
        assert worst<2e-6;convolution.append({'variant':name,'max_abs_error':worst,'channels_checked':40})
    dump(ROOT/'catalogue.json',catalogue);dump(ROOT/'gabor_specs.json',specs);dump(ROOT/'convolution_verification.json',convolution)
    print(f'Prepared {len(catalogue)} input entries; all new kernels independently verified',flush=True)

def screen(seed,catalogue):
    train,_=split(LABELS,2,seed);rows=[]
    gray_cache={}
    for entry in catalogue:
        if entry['gray_path'] not in gray_cache:
            x=np.asarray(np.load(BASE/entry['gray_path'],mmap_mode='r')[train],dtype=float).reshape(80,-1)
            dist=cdist(x,x,'cosine');np.fill_diagonal(dist,np.inf)
            gray_cache[entry['gray_path']]=int(np.sum(LABELS[train[np.argmin(dist,axis=1)]]==LABELS[train]))
        original=load(entry)
        for norm in NORMS:
            data,_=normalize(original,norm,train);x=np.asarray(data['gabor'][train],dtype=float).reshape(80,-1)
            dist=cdist(x,x,'cosine');np.fill_diagonal(dist,np.inf)
            count=int(np.sum(LABELS[train[np.argmin(dist,axis=1)]]==LABELS[train]));gc=gray_cache[entry['gray_path']]
            rows.append({'entry_id':entry['id'],'normalization':norm,'gray_correct':gc,'gabor_correct':count,'sum_correct':gc+count})
    anchor=selected(seed)['shared_input']['id'];candidates=[{'entry_id':anchor,'normalization':norm} for norm in NORMS]
    for norm in NORMS:
        winner=max([r for r in rows if r['normalization']==norm],key=lambda r:r['sum_correct'])
        candidate={k:winner[k] for k in ['entry_id','normalization']}
        if candidate not in candidates:candidates.append(candidate)
    return rows,candidates

def compute_cv(seed,candidate,cases,policies,folder):
    entry=next(e for e in read(ROOT/'catalogue.json') if e['id']==candidate['entry_id'])
    original=load(entry);train,probe=split(LABELS,2,seed)
    # counts[policy,case,alpha,dimension-1]，无probe图像参与拟合或转换。
    counts=np.zeros((len(policies),len(cases),len(ALPHAS),35),dtype=np.int16)
    for learning,validation in folds(train,seed):
        assert set(learning)|set(validation)==set(train) and set(learning).isdisjoint(probe)
        data,_=normalize(original,candidate['normalization'],learning)
        for j,case in enumerate(cases):
            _,kind,algorithm,dims,center,_=CASES[case]
            p=int(np.prod(dims));assert p<=len(learning)-len(np.unique(LABELS[learning]))
            _,g,q=fit(data[kind],learning,validation,algorithm,dims,center,'identity')
            for ai,alpha in enumerate(ALPHAS):
                lda=fisher(g,LABELS[learning],alpha)
                for pi,name in enumerate(policies):
                    a,b=coordinates(g,q,lda,POLICIES[name]);pred,_=prefix_nn(a,b,LABELS[learning],POLICIES[name]['metric'])
                    for d in range(2,a.shape[1]+1):counts[pi,j,ai,d-1]+=int(np.sum(pred[:,d-1]==LABELS[validation]))
    path=folder/f'cv_{candidate["entry_id"]}_{candidate["normalization"]}.npz'
    np.savez_compressed(path,counts=counts,cases=np.array(cases),policies=np.array(policies),alphas=np.array(ALPHAS))
    return counts,path

def best_methods(counts,cases,policies,allowed_alpha):
    results=[]
    for pi,name in enumerate(policies):
        methods=[]
        for method in METHODS:
            pool=[]
            for j,case in enumerate(cases):
                if CASES[case][0]!=method:continue
                p=min(int(np.prod(CASES[case][3])),39);paper_d=CASES[case][5]
                ds=([paper_d] if paper_d<=p else [])+[d for d in range(2,p+1) if d!=paper_d]
                for ai in allowed_alpha:
                    for d in ds:pool.append({'method':method,'case_index':case,'alpha':ALPHAS[ai],'dimension':d,'correct':int(counts[pi,j,ai,d-1])})
            methods.append(max(pool,key=lambda r:r['correct']))
        objective=sum(m['correct'] for m in methods if m['method'].startswith('CSA'))
        results.append({'policy':name,'methods':methods,'CSA_sum_correct':objective})
    return max(results,key=lambda r:r['CSA_sum_correct'])

def select_stage(mode,seeds):
    catalogue=read(ROOT/'catalogue.json');start=time.perf_counter()
    for seed in seeds:
        out=ROOT/f'seed{seed}';out.mkdir(exist_ok=True);target=out/f'{mode}_selected.json'
        assert not target.exists(),'Refusing to replace a frozen selection'
        train,probe=split(LABELS,2,seed)
        dump(out/'indices.json',{'seed':seed,'gallery':train.tolist(),'probe':probe.tolist(),
              'folds':[{'learning':a.tolist(),'validation':b.tolist()} for a,b in folds(train,seed)]})
        folder=out/mode;folder.mkdir(exist_ok=False)
        if mode=='fixed':
            old=selected(seed);candidates=[{'entry_id':old['shared_input']['id'],'normalization':'none'}]
            cases=list(range(4));policies=[old['selection']['policy']]
        else:
            scores,candidates=screen(seed,catalogue);write_csv(folder/'screen.csv',scores)
            cases=list(range(len(CASES)));old_policy=selected(seed)['selection']['policy']
            policies=[old_policy]+[p for p in POLICIES if p!=old_policy]
        dump(folder/'plan.json',{'candidates':candidates,'cases':cases,'policies':policies,'alpha_grid':ALPHAS})
        all_options=[]
        for candidate in candidates:
            counts,path=compute_cv(seed,candidate,cases,policies,folder)
            for profile,alphas in [('optimized_shrinkage_LDA',list(range(len(ALPHAS)))),('optimized_ordinary_LDA',[0])]:
                winner=best_methods(counts,cases,policies,alphas)
                all_options.append({**candidate,**winner,'profile':profile,'score_file':path.relative_to(ROOT).as_posix(),'score_sha256':file_hash(path)})
            print(f'{mode} seed{seed}: gallery CV complete {candidate}',flush=True)
        profiles=['optimized_shrinkage_LDA'] if mode=='fixed' else ['optimized_ordinary_LDA','optimized_shrinkage_LDA']
        winners=[]
        for profile in profiles:
            winner=max([r for r in all_options if r['profile']==profile],key=lambda r:r['CSA_sum_correct'])
            if mode=='fixed':winner={**winner,'profile':'regularization_only'}
            winners.append(winner)
        dump(folder/'candidate_best.json',all_options)
        dump(target,{'seed':seed,'mode':mode,'profiles':winners,'training_only':True,'selected_before_own_outer_scoring':True})
        print(f'{mode} seed{seed} frozen: '+str([(w['profile'],w['entry_id'],w['normalization'],w['CSA_sum_correct']) for w in winners]),flush=True)
    print(f'{mode} selection seconds={time.perf_counter()-start:.1f}',flush=True)

def evaluate():
    assert all((ROOT/f'seed{s}/{mode}_selected.json').exists() for s in SEEDS for mode in ['fixed','joint'])
    freeze={f'{s}_{mode}':file_hash(ROOT/f'seed{s}/{mode}_selected.json') for s in SEEDS for mode in ['fixed','joint']}
    out=ROOT/'evaluation';out.mkdir(exist_ok=False);(out/'embeddings').mkdir()
    dump(out/'selection_freeze.json',freeze);hashes=snapshot(out/'source_snapshot')
    catalogue={e['id']:e for e in read(ROOT/'catalogue.json')};rows=[];predictions=[];inner=[];checks=[]
    for seed in SEEDS:
        train,probe=split(LABELS,2,seed)
        for mode in ['fixed','joint']:
            sel=read(ROOT/f'seed{seed}/{mode}_selected.json')
            for winner in sel['profiles']:
                entry=catalogue[winner['entry_id']];original=load(entry);policy=POLICIES[winner['policy']]
                for kind in original:assert file_hash(BASE/entry[f'{kind}_path'])==entry[f'{kind}_sha256']
                assert file_hash(ROOT/winner['score_file'])==winner['score_sha256']
                for setting in winner['methods']:
                    case=setting['case_index'];method,kind,algorithm,dims,center,_=CASES[case];d=setting['dimension'];alpha=setting['alpha']
                    cv=0
                    for learn,valid in folds(train,seed):
                        data,_=normalize(original,winner['normalization'],learn)
                        _,g,q=fit(data[kind],learn,valid,algorithm,dims,center,'identity');lda=fisher(g,LABELS[learn],alpha)
                        a,b=coordinates(g,q,lda,policy);pred,near,flags=classify(a[:,:d],b[:,:d],learn,valid,policy['metric']);cv+=int(flags.sum())
                        inner.extend({'seed':seed,'profile':winner['profile'],'method':method,'validation_index':int(i),
                                      'neighbor_index':int(learn[j]),'true':int(LABELS[i]),'predicted':int(p),'correct':int(c)} for i,j,p,c in zip(valid,near,pred,flags))
                    assert cv==setting['correct']
                    data,scale=normalize(original,winner['normalization'],train)
                    model,g,q=fit(data[kind],train,probe,algorithm,dims,center,'identity')
                    if algorithm=='csa':assert len(model.history)==3
                    for profile,actual_alpha in [(winner['profile'],alpha)]+([('shrinkage_ablation_alpha0',0.)] if winner['profile']=='optimized_shrinkage_LDA' else []):
                        lda=fisher(g,LABELS[train],actual_alpha,check=True);a,b=coordinates(g,q,lda,policy)
                        a=a[:,:d];b=b[:,:d];pred,near,flags=classify(a,b,train,probe,policy['metric'])
                        key={'seed':seed,'profile':profile,'method':method};identifier=f'{seed}_{profile}_{case}'
                        np.savez_compressed(out/'embeddings'/f'{identifier}.npz',gallery=a,probe=b,train=train,query=probe,
                                            channel_scale=np.array([]) if scale is None else scale)
                        rows.append({**key,'entry_id':entry['id'],'normalization':winner['normalization'],'policy':winner['policy'],
                            'case_index':case,'pre_dims':str(dims),'center':center,'alpha':actual_alpha,'lda_dimension':d,
                            'inner_correct':cv if profile==winner['profile'] else '',
                            'correct':int(flags.sum()),'probe_count':320,'accuracy_percent':100*float(flags.mean()),
                            'input_sha256':entry[f'{kind}_sha256'],'embedding':f'embeddings/{identifier}.npz'})
                        predictions.extend({**key,'probe_index':int(i),'neighbor_index':int(train[j]),'true':int(LABELS[i]),'predicted':int(p),'correct':int(c)}
                                           for i,j,p,c in zip(probe,near,pred,flags))
                        checks.append({**key,'alpha':actual_alpha,'generalized_solver_independently_checked':True,'NN_independently_checked':True})
        print(f'Evaluation seed{seed}: all selected inner scores replayed and outer predictions checked',flush=True)
    assert all(file_hash(ROOT/f'seed{s}/{m}_selected.json')==freeze[f'{s}_{m}'] for s in SEEDS for m in ['fixed','joint'])
    assert len(rows)==160 and len(predictions)==51200 and len(inner)==9600
    write_csv(out/'results.csv',rows);write_csv(out/'predictions.csv',predictions);write_csv(out/'inner_predictions.csv',inner)
    dump(out/'solver_NN_checks.json',checks);dump(out/'execution.json',{'completed':True,'source_snapshot_sha256':hashes,
        'score_records':len(rows),'outer_predictions':len(predictions),'selected_inner_predictions':len(inner)})

def verify():
    out=ROOT/'evaluation';rows=read_csv(out/'results.csv');pred=read_csv(out/'predictions.csv');inner=read_csv(out/'inner_predictions.csv')
    counts=defaultdict(lambda:[0,0]);inner_counts=defaultdict(lambda:[0,0]);catalogue={e['id']:e for e in read(ROOT/'catalogue.json')}
    freeze=read(out/'selection_freeze.json')
    for s in SEEDS:
        for mode in ['fixed','joint']:assert file_hash(ROOT/f'seed{s}/{mode}_selected.json')==freeze[f'{s}_{mode}']
    for r in pred:
        seed=int(r['seed']);train,probe=split(LABELS,2,seed);i=int(r['probe_index']);j=int(r['neighbor_index'])
        assert i in probe and j in train and int(r['true'])==LABELS[i] and int(r['predicted'])==LABELS[j]
        assert int(r['correct'])==int(LABELS[i]==LABELS[j])
        key=(r['seed'],r['profile'],r['method']);counts[key][0]+=1;counts[key][1]+=int(r['correct'])
    for r in inner:
        seed=int(r['seed']);train,probe=split(LABELS,2,seed);i=int(r['validation_index']);j=int(r['neighbor_index'])
        assert i in train and j in train and i!=j and int(r['true'])==LABELS[i] and int(r['predicted'])==LABELS[j]
        assert int(r['correct'])==int(LABELS[i]==LABELS[j]);key=(r['seed'],r['profile'],r['method'])
        inner_counts[key][0]+=1;inner_counts[key][1]+=int(r['correct'])
    groups=defaultdict(list)
    for r in rows:
        key=(r['seed'],r['profile'],r['method']);assert counts[key]==[320,int(r['correct'])]
        if r['inner_correct']!='':assert inner_counts[key]==[80,int(r['inner_correct'])]
        emb=np.load(out/r['embedding']);train=emb['train'];probe=emb['query'];policy=POLICIES[r['policy']]
        p,n,f=classify(emb['gallery'],emb['probe'],train,probe,policy['metric']);assert int(f.sum())==int(r['correct'])
        saved=[q for q in pred if (q['seed'],q['profile'],q['method'])==key]
        np.testing.assert_array_equal(p,[int(q['predicted']) for q in saved]);np.testing.assert_array_equal(train[n],[int(q['neighbor_index']) for q in saved])
        kind='gray' if '(Gray)' in r['method'] or r['method']=='CSA-2+LDA' else 'gabor';entry=catalogue[r['entry_id']]
        assert r['input_sha256']==entry[f'{kind}_sha256']
        groups[(r['profile'],r['method'])].append(float(r['accuracy_percent']))
    summary=[dict(zip(['profile','method'],key))|{'n_splits':len(v),'mean_percent':float(np.mean(v)),
          'std_pp':float(np.std(v,ddof=1)),'min_percent':min(v),'max_percent':max(v)} for key,v in groups.items()]
    assert len(rows)==160 and len(pred)==51200 and len(inner)==9600
    dump(out/'summary.json',summary);dump(out/'verification.json',{'passed':True,'score_records':len(rows),'outer_predictions':len(pred),
         'selected_inner_predictions':len(inner),'all_embeddings_NN_replayed':True,'independent_Fisher_checks':160,
         'selection_files_unchanged':True,'verifier_sha256':file_hash(BASE/'optimize_csa_lda.py')})
    print(json.dumps(summary,indent=2),flush=True)

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    current=read(ROOT/'evaluation/summary.json');old=read(SOURCE/'evaluation/summary.json')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5));profiles=['regularization_only','optimized_ordinary_LDA','optimized_shrinkage_LDA','shrinkage_ablation_alpha0']
    for ax,method,paper in zip(axes,['CSA-2+LDA','CSA-3+LDA'],[84.1,90.9]):
        prev=next(r for r in old if r['profile']=='paper_dimensions' and r['method']==method)
        records=[prev]+[next(r for r in current if r['profile']==p and r['method']==method) for p in profiles]
        means=[r['mean_percent'] for r in records];std=[r['std_pp'] for r in records]
        bars=ax.bar(np.arange(5),means,yerr=std,capsize=4,color=['#8496A5','#D99F5C','#669DB4','#568B68','#BA847F'])
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in means],padding=5,fontsize=10)
        ax.axhline(paper,color='#555555',linestyle='--',label=f'Paper {paper}%')
        ax.set_xticks(np.arange(5),['Previous','Shrink only','Ordinary\njoint CV','Shrinkage\njoint CV','Same config\nalpha=0'],fontsize=9)
        ax.set_ylim(45,104);ax.set_ylabel('Accuracy (%)');ax.set_title(method);ax.legend(loc='upper right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('ORL training-only optimization | fixed seeds20-29 | mean +/- sample SD')
    fig.text(.5,.02,'Joint search shares input and distance across all methods. Positive-alpha Fisher is an extension.\n'
        'All outer splits are retained; historical probes were observed during earlier development.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.09,1,.94]);fig.savefig(ROOT/'optimization_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','select-fixed','select-joint','evaluate','verify','plot'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args();ROOT=args.output.resolve();assert ROOT.is_relative_to(BASE) and all(s in SEEDS for s in args.seeds)
    with threadpool_limits(limits=4):
        if args.stage=='prepare':prepare()
        elif args.stage.startswith('select-'):select_stage(args.stage.split('-')[1],args.seeds)
        elif args.stage=='evaluate':evaluate()
        elif args.stage=='verify':verify()
        else:plot()
