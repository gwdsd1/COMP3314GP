"""ORL CSA+LDA 定位：固定已有输入，区分压缩、Fisher 投影和度量。

diagnose 为固定消融，不用外层成绩选配置；select 仅用原80张gallery。
不修改 csa.py、已有实验或数据。regularization/QR 仅作明确的扩展诊断。
"""
import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA,PCA,LDA
from gabor import file_hash
from audit_orl_protocol import BASE,LABELS,NAMES,SEEDS,POLICIES,split,folds,transform_batches,prefix_nn,lda_project
from run_shared_protocol import feature_spec
from diagnose_orl_gabor import filtered
from run_orl_gray import write_csv

SOURCE=BASE/'results/orl_protocol_audit'
ROOT=BASE/'results/csa_lda_focus'
CASES=[('gray','PCA-paper','pca',30,18,True,'identity'),
       ('gray','PCA-equal-budget','pca',22,16,True,'identity'),
       ('gray','CSA-paper','csa',(11,2),16,False,'identity'),
       ('gray','CSA-centered','csa',(11,2),16,True,'identity'),
       ('gray','CSA-random','csa',(11,2),16,False,'random'),
       ('gabor','PCA-paper','pca',33,25,True,'identity'),
       ('gabor','PCA-equal-budget','pca',24,23,True,'identity'),
       ('gabor','CSA-paper','csa',(4,2,3),23,False,'identity'),
       ('gabor','CSA-centered','csa',(4,2,3),23,True,'identity'),
       ('gabor','CSA-random','csa',(4,2,3),23,False,'random'),
       ('gabor','CSA-highrank','csa',(15,1,26),None,False,'identity')]
NORMALIZATIONS=['baseline','tensor_l2','image_zscore','jet_l2','channel_rms']

def dump(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def selected(seed):return json.loads((SOURCE/f'seed{seed}/selected.json').read_text())

def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'features').mkdir();(ROOT/'source_snapshot').mkdir()
    for n in ['diagnose_csa_lda.py','csa.py','audit_orl_protocol.py','diagnose_orl_gabor.py']:
        shutil.copy2(BASE/n,ROOT/'source_snapshot'/n)
    dump(ROOT/'protocol.json',{'seeds':SEEDS,'cases':CASES,'normalizations':NORMALIZATIONS,
         'input':'frozen selected inputs from previous audit; original per-class2gallery/8probe unchanged',
         'Tmax':3,'random_init_seed':3314,'LDA_pre_dimension_limit':40,'core_NN':'cosine and Euclidean; mean/origin explicitly labelled',
         'LDA':'unregularized Fisher whitened generalized eigenproblem; independent scipy.eigh(Sb,Sw)',
         'diagnostic_metrics':'six old conventions + QR orthonormal basis; fixed reported dimension and full pre-dimension',
         'QR_role':'alternative metric on the same chosen LDA span; diagnostic only, not asserted original algorithm',
         'normalization_scope':'same feature array for PCA and CSA within each kind; image_zscore filters zero-mean unit-std gray before Gabor',
         'normalization_detail':'tensor_l2 per sample; jet_l2 per pixel Gabor40-vector (gray uses tensor_l2); channel_rms train-only40channel RMS (gray unchanged)',
         'variance_retention':'centered total/between/within energy of encoded training features divided by corresponding original input energy',
         'diagnostic_selection':'none; all fixed outer diagnostic scores retained; historical probes already observed, not blind',
         'training_selection':'fixed initial ranks; mean CSA Gray/Gabor inner accuracy; shared normalization/CSA center/init/LDA policy; only own80gallery',
         'input_sha256':{str(s):{'selection':file_hash(SOURCE/f'seed{s}/selected.json'),
                  'gray':selected(s)['shared_input']['gray_sha256'],'gabor':selected(s)['shared_input']['gabor_sha256']} for s in SEEDS},
         'sources_sha256':{n:file_hash(BASE/n) for n in ['diagnose_csa_lda.py','csa.py','audit_orl_protocol.py','diagnose_orl_gabor.py']}})

def arrays(seed):
    entry=selected(seed)['shared_input']
    return {kind:np.load(BASE/entry[f'{kind}_path'],mmap_mode='r') for kind in ['gray','gabor']}

def normalized(source,seed,variant,train):
    if variant=='baseline':return source
    if variant=='image_zscore':
        gray=np.asarray(source['gray'],dtype=np.float32)
        gray=(gray-gray.mean((1,2),keepdims=True))/np.maximum(gray.std((1,2),keepdims=True),1e-8)
        entry=selected(seed)['shared_input'];path=ROOT/'features'/f'{entry["id"]}_image_zscore.npy'
        if not path.exists():np.save(path,filtered(gray,feature_spec(entry['kernel'])))
        return {'gray':gray,'gabor':np.load(path,mmap_mode='r')}
    result={}
    for kind,array in source.items():
        x=np.asarray(array,dtype=np.float64)
        if variant=='tensor_l2' or variant=='jet_l2' and kind=='gray':
            x=x/np.maximum(np.linalg.norm(x.reshape(len(x),-1),axis=1),1e-12).reshape((-1,)+(1,)*(x.ndim-1))
        elif variant=='jet_l2':x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
        elif variant=='channel_rms' and kind=='gabor':
            scale=np.sqrt(np.mean(x[train]**2,axis=(0,1,2)));x=x/np.maximum(scale,1e-12)
        result[kind]=x
    return result

def fit(data,train,query,algorithm,dimensions,center,init):
    model=PCA(dimensions,center=center,solver='gram') if algorithm=='pca' else CSA(dimensions,center=center,init=init,seed=3314,max_iter=3)
    model.fit(data[train]);g=transform_batches(model,data,train);q=transform_batches(model,data,query)
    return model,g,q

def energies(x,labels):
    x=np.asarray(x,dtype=float).reshape(len(x),-1);mean=x.mean(0)
    total=np.sum((x-mean)**2);between=0.;within=0.
    for c in np.unique(labels):
        a=x[labels==c];mu=a.mean(0);between+=len(a)*np.sum((mu-mean)**2);within+=np.sum((a-mu)**2)
    assert abs(total-between-within)<1e-9*max(1,total)
    return np.array([total,between,within])

def classify(g,q,train,query,metric):
    distance=cdist(q,g,metric);assert np.isfinite(distance).all();near=np.argmin(distance,axis=1)
    if metric=='cosine':ind=1-(q/np.linalg.norm(q,axis=1)[:,None])@(g/np.linalg.norm(g,axis=1)[:,None]).T
    else:ind=np.sqrt(np.maximum(0,np.sum(q*q,axis=1)[:,None]+np.sum(g*g,axis=1)[None,:]-2*q@g.T))
    np.testing.assert_allclose(distance,ind,atol=1e-7,rtol=1e-7)
    np.testing.assert_array_equal(near,np.argmin(ind,axis=1))
    pred=LABELS[train[near]];correct=pred==LABELS[query]
    return pred,near,correct

def diagnose():
    assert not (ROOT/'diagnosis_results.csv').exists()
    results=[];predictions=[];statistics=[];curves=[];solver_checks=[];normalization_hashes=[]
    for seed in SEEDS:
        train,probe=split(LABELS,2,seed);source=arrays(seed);sel=selected(seed);active=sel['selection']['policy']
        for variant in NORMALIZATIONS:
            data=normalized(source,seed,variant,train)
            cases=CASES if variant=='baseline' else [c for c in CASES if c[1] in ['PCA-paper','CSA-paper']]
            for kind,label,algorithm,dimensions,d,center,init in cases:
                x=data[kind];model,g,q=fit(x,train,probe,algorithm,dimensions,center,init)
                key={'seed':seed,'normalization':variant,'kind':kind,'model':label}
                fractions=energies(g,LABELS[train])/energies(x[train],LABELS[train])
                statistics.append({**key,'pre_dimension':g.shape[1],'total_variance_retained':float(fractions[0]),
                     'between_variance_retained':float(fractions[1]),'within_variance_retained':float(fractions[2]),
                     'core_mean_energy_fraction':float(len(g)*np.sum(g.mean(0)**2)/np.sum(g*g)),
                     'sw_rank':'','sw_condition':'','generalized_eigen_residual':'','eigenbasis_condition':''})
                _,_,core_flags=classify(g,q,train,probe,'cosine')
                outputs=[('core_cosine',g,q,'cosine',g.shape[1]),('core_centered_cosine',g-g.mean(0),q-g.mean(0),'cosine',g.shape[1]),
                         ('core_euclidean',g,q,'euclidean',g.shape[1])]
                if d is not None:
                    lda=LDA().fit(g,LABELS[train]);values=eigh(lda.sw,eigvals_only=True)
                    eig,vec=eigh(lda.sb,lda.sw);eig=eig[::-1];vec=vec[:,::-1]
                    np.testing.assert_allclose(eig[:lda.max_components],lda.eigenvalues,atol=1e-5,rtol=1e-7)
                    a=vec[:,:lda.max_components]/np.linalg.norm(vec[:,:lda.max_components],axis=0)
                    b=lda.basis/np.linalg.norm(lda.basis,axis=0)
                    cos=np.abs(np.sum(a*b,axis=0));assert np.min(cos)>1-1e-6
                    residual=np.linalg.norm(lda.sb@lda.basis-(lda.sw@lda.basis)*lda.eigenvalues)/max(1,np.linalg.norm(lda.sb@lda.basis))
                    statistics[-1].update(sw_rank=lda.sw_rank,sw_condition=float(values[-1]/values[0]),generalized_eigen_residual=float(residual),
                                           eigenbasis_condition=float(np.linalg.cond(lda.basis)))
                    solver_checks.append({**key,'minimum_basis_cosine':float(np.min(cos)),'relative_residual':float(residual)})
                    policies=POLICIES if variant=='baseline' else {active:sel['shared_policy']}
                    for name,policy in policies.items():
                        a,b=lda_project(g,q,lda,policy)
                        outputs += [(f'LDA_{name}_paper',a[:,:d],b[:,:d],policy['metric'],d),
                                    (f'LDA_{name}_full',a,b,policy['metric'],a.shape[1])]
                        if variant=='baseline':
                            for k in range(2,a.shape[1]+1):
                                # 1D cosine大量并列不用于曲线结论；使用标准cdist。
                                dist=cdist(b[:,:k],a[:,:k],policy['metric']);pred=LABELS[train[np.argmin(dist,axis=1)]]
                                curves.append({**key,'policy':name,'dimension':k,'accuracy_percent':100*float(np.mean(pred==LABELS[probe]))})
                    for k in [d,lda.max_components]:
                        basis=np.linalg.qr(lda.basis[:,:k])[0];a=(g-lda.mean)@basis;b=(q-lda.mean)@basis
                        outputs.append((f'LDA_QR_{"paper" if k==d else "full"}',a,b,'euclidean',k))
                        if k==g.shape[1]:
                            _,_,qr=classify(a,b,train,probe,'euclidean');_,_,raw=classify(g,q,train,probe,'euclidean')
                            np.testing.assert_array_equal(qr,raw)
                seen=set()
                for stage,a,b,metric,k in outputs:
                    if stage in seen:continue
                    seen.add(stage);pred,near,flags=classify(a,b,train,probe,metric)
                    results.append({**key,'stage':stage,'dimension':k,'correct':int(flags.sum()),'probe_count':320,'accuracy_percent':100*float(flags.mean()),
                         'core_correct':int(core_flags.sum()),'lost_core_correct':int(np.sum(core_flags&~flags)),'gained_core_wrong':int(np.sum(~core_flags&flags))})
                    predictions.extend({**key,'stage':stage,'probe_index':int(i),'neighbor_index':int(train[j]),'true':int(LABELS[i]),'predicted':int(p),'correct':int(c)}
                                       for i,j,p,c in zip(probe,near,pred,flags))
            print(f'Diagnosis seed{seed} {variant} complete',flush=True)
        # 直接记录缓存实际字节哈希，其他归一化由训练索引重新生成。
        if variant=='channel_rms':
            normalization_hashes.append({'seed':seed,'image_zscore_feature_sha256':file_hash(ROOT/'features'/f'{sel["shared_input"]["id"]}_image_zscore.npy')})
    write_csv(ROOT/'diagnosis_results.csv',results);write_csv(ROOT/'diagnosis_predictions.csv',predictions)
    write_csv(ROOT/'representation_statistics.csv',statistics);write_csv(ROOT/'dimension_curves.csv',curves)
    dump(ROOT/'solver_checks.json',solver_checks);dump(ROOT/'normalization_hashes.json',normalization_hashes)
    # 先核对旧主表，不接受换协议后自称复现了旧分数。
    old=json.loads((SOURCE/'evaluation/summary.json').read_text())
    for kind,method in [('gray','CSA-2+LDA'),('gabor','CSA-3+LDA'),('gray','PCA+LDA (Gray)'),('gabor','PCA+LDA (Gabor)')]:
        model='CSA-paper' if method.startswith('CSA') else 'PCA-paper'
        scores=[r['accuracy_percent'] for r in results if r['kind']==kind and r['model']==model and r['normalization']=='baseline'
                  and r['stage']==f'LDA_{selected(r["seed"])["selection"]["policy"]}_paper']
        assert len(scores)==10
        assert abs(np.mean(scores)-next(r['mean_percent'] for r in old if r['profile']=='paper_dimensions' and r['method']==method))<1e-10
    dump(ROOT/'diagnosis_execution.json',{'completed':True,'model_fits':270,'score_records':len(results),'prediction_records':len(predictions),
         'independent_generalized_LDA_checks':len(solver_checks),'old_four_LDA_anchors_reproduced':True,'code_sha256':file_hash(BASE/'diagnose_csa_lda.py')})

def summarize():
    import csv
    with (ROOT/'diagnosis_results.csv').open() as f:rows=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in rows:groups[(r['normalization'],r['kind'],r['model'],r['stage'])].append(r)
    summary=[]
    for key,group in groups.items():
        assert len(group)==10;v=[float(r['accuracy_percent']) for r in group]
        summary.append(dict(zip(['normalization','kind','model','stage'],key))|{'mean_percent':float(np.mean(v)),'std_pp':float(np.std(v,ddof=1)),
              'lost_core_correct_mean':float(np.mean([int(r['lost_core_correct']) for r in group])),
              'gained_core_wrong_mean':float(np.mean([int(r['gained_core_wrong']) for r in group]))})
    write_csv(ROOT/'diagnosis_summary.csv',summary)
    # 主指标按每次已有共同距离规则，不因第28种子的原点不同而混用对照。
    chosen=[]
    for norm in NORMALIZATIONS:
        for kind in ['gray','gabor']:
            for model in ['PCA-paper','CSA-paper']:
                rs=[r for r in rows if r['normalization']==norm and r['kind']==kind and r['model']==model and
                    r['stage']==f'LDA_{selected(int(r["seed"]))["selection"]["policy"]}_paper']
                vals=[float(r['accuracy_percent']) for r in rs];assert len(vals)==10
                chosen.append({'normalization':norm,'kind':kind,'model':model,'mean_percent':float(np.mean(vals)),'std_pp':float(np.std(vals,ddof=1))})
    dump(ROOT/'normalization_summary.json',chosen)
    # 回读所有预测，检查标签、邻居归属和准确率；矩阵NN与独立LDA已逐模型执行。
    with (ROOT/'diagnosis_predictions.csv').open() as f:pred=list(csv.DictReader(f))
    counts=defaultdict(lambda:[0,0]);probe_sets={s:set(split(LABELS,2,s)[1]) for s in SEEDS};train_sets={s:set(split(LABELS,2,s)[0]) for s in SEEDS}
    for p in pred:
        seed=int(p['seed']);i=int(p['probe_index']);j=int(p['neighbor_index'])
        assert i in probe_sets[seed] and j in train_sets[seed] and int(p['true'])==LABELS[i] and int(p['predicted'])==LABELS[j]
        assert int(p['correct'])==int(LABELS[i]==LABELS[j])
        key=tuple(p[k] for k in ['seed','normalization','kind','model','stage']);counts[key][0]+=1;counts[key][1]+=int(p['correct'])
    for r in rows:
        key=tuple(r[k] for k in ['seed','normalization','kind','model','stage']);assert counts[key]==[320,int(r['correct'])]
    dump(ROOT/'diagnosis_verification.json',{'passed':True,'score_records':len(rows),'prediction_records':len(pred),
         'independent_matrix_NN':True,'independent_generalized_LDA':True,'full_QR_vs_raw_Euclidean_invariance':True,
         'four_previous_LDA_means_reproduced':True,'code_sha256':file_hash(BASE/'diagnose_csa_lda.py')})
    print(json.dumps(chosen,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','diagnose','summarize'],required=True)
    p.add_argument('--output',type=Path,default=ROOT);args=p.parse_args();ROOT=args.output.resolve();assert ROOT.is_relative_to(BASE)
    with threadpool_limits(limits=4):
        if args.stage=='prepare':prepare()
        elif args.stage=='diagnose':diagnose()
        else:summarize()
