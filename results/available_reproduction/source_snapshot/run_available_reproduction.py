"""现有 ORL/MNIST 的固定论文配置、描述曲线、收敛和重建。

两种预先声明的协议均完整报告；不使用 probe 选参数或删除低分划分。
"""

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA, PCA, LDA, reconstruction_metrics
from gabor import CONFIG, cached_features
from run_orl_gray import write_csv

BASE=Path(__file__).resolve().parent
SEEDS=[20,21,22]
ORL_CASES=[('PCA+LDA (Gray)','gray','pca',30,18,82.2),
           ('CSA-2+LDA','gray','csa',(11,2),16,84.1),
           ('PCA (Gabor)','gabor','pca',76,None,86.6),
           ('CSA-3','gabor','csa',(15,1,26),None,94.4),
           ('PCA+LDA (Gabor)','gabor','pca',33,25,87.2),
           ('CSA-3+LDA','gabor','csa',(4,2,3),23,90.9)]


def split(labels, gallery_count, seed):
    rng=np.random.default_rng(seed); train=[];probe=[]
    for label in np.unique(labels):
        order=rng.permutation(np.flatnonzero(labels==label))
        train.extend(order[:gallery_count]);probe.extend(order[gallery_count:])
    train,probe=np.array(train),np.array(probe)
    assert not np.intersect1d(train,probe).size
    assert len(np.unique(np.r_[train,probe]))==len(labels)
    return train,probe


def settings(profile, model_type, kind, lda_dim):
    if profile=='euclidean_control':
        return {'center':False,'metric':'euclidean','lda_scaling':'sw_identity'}
    # 依据先前验证证据与幅值特征的尺度性质，在本轮 probe 评分前固定。
    return {'center':model_type=='pca','metric':'euclidean' if kind=='gray' and model_type=='csa' else 'cosine',
            'lda_scaling':'unit_columns'}


def transform_batches(model, data, indices, batch=128):
    return np.concatenate([model.transform(data[indices[start:start+batch]]).reshape(min(batch,len(indices)-start),-1)
                           for start in range(0,len(indices),batch)])


def nn(gallery,probe,labels,metric):
    predictions=[];neighbors=[];distances=[]
    for start in range(0,len(probe),256):
        matrix=cdist(probe[start:start+256],gallery,metric)
        if not np.isfinite(matrix).all():raise ValueError('Nonfinite NN distance')
        index=np.argmin(matrix,axis=1)
        predictions.extend(labels[index]);neighbors.extend(index)
        distances.extend(matrix[np.arange(len(index)),index])
    return np.array(predictions),np.array(neighbors),np.array(distances)


def lda_coordinates(features,lda,dimension,scaling):
    basis=lda.basis[:,:dimension]
    if scaling=='unit_columns':basis=basis/np.linalg.norm(basis,axis=0)
    return (features-lda.mean)@basis


def make_model(algorithm,dimensions,config):
    return PCA(dimensions,center=config['center'],solver='gram') if algorithm=='pca' else CSA(dimensions,center=config['center'],max_iter=3)


def record_recognition(dataset,seed,profile,case,data,labels,names,train,probe,scores,predictions,curves,histories):
    method,kind,algorithm,dimensions,lda_dim,paper=case
    config=settings(profile,algorithm,kind,lda_dim)
    started=time.perf_counter();model=make_model(algorithm,dimensions,config).fit(data[train])
    gallery=transform_batches(model,data,train);queries=transform_batches(model,data,probe)
    lda=LDA().fit(gallery,labels[train]) if lda_dim is not None else None
    if lda:
        dims=range(1,lda.max_components+1) if dataset=='ORL' else [lda_dim]
        for d in dims:
            predicted,_,_=nn(lda_coordinates(gallery,lda,d,config['lda_scaling']),
                             lda_coordinates(queries,lda,d,config['lda_scaling']),labels[train],config['metric'])
            curves.append({'dataset':dataset,'seed':seed,'profile':profile,'method':method,
                           'feature_dim':d,'pre_dims':str(dimensions),'accuracy_percent':100*np.mean(predicted==labels[probe])})
        gallery=lda_coordinates(gallery,lda,lda_dim,config['lda_scaling'])
        queries=lda_coordinates(queries,lda,lda_dim,config['lda_scaling'])
    elif dataset=='ORL' and algorithm=='pca':
        # 该曲线只描述固定预维数以内的前缀；不从 probe 曲线挑主成绩。
        for d in range(1,dimensions+1):
            predicted,_,_=nn(gallery[:,:d],queries[:,:d],labels[train],config['metric'])
            curves.append({'dataset':dataset,'seed':seed,'profile':profile,'method':method,
                           'feature_dim':d,'pre_dims':str(d),'accuracy_percent':100*np.mean(predicted==labels[probe])})
    predicted,neighbors,distances=nn(gallery,queries,labels[train],config['metric'])
    correct=int(np.sum(predicted==labels[probe]));accuracy=100*correct/len(probe)
    row={'dataset':dataset,'seed':seed,'profile':profile,'method':method,'pre_dims':str(dimensions),
         'lda_dim':lda_dim if lda_dim is not None else '',**config,'correct':correct,'probe_count':len(probe),
         'accuracy_percent':accuracy,'paper_percent':paper,'difference_pp':accuracy-paper,
         'wall_seconds':time.perf_counter()-started}
    scores.append(row)
    predictions.extend({'dataset':dataset,'seed':seed,'profile':profile,'method':method,'probe':names[i],
                        'true':int(labels[i]),'predicted':int(p),'nearest_gallery':names[train[j]],
                        'distance':float(d),'correct':int(p==labels[i])} for i,p,j,d in zip(probe,predicted,neighbors,distances))
    if isinstance(model,CSA):
        histories.append({'dataset':dataset,'seed':seed,'profile':profile,'method':method,
                          'history':model.history,'stop_reason':model.stop_reason})
    print(f'{dataset} seed={seed} {profile} {method}: {accuracy:.2f}% (paper {paper:.1f}%)',flush=True)


def reconstruction(data,output):
    # 一人一图 s*/1.pgm；同一输入比较两方法，总存储量包括共享基。
    x=np.asarray(data[::10],dtype=np.float64);n=len(x);D=int(np.prod(x.shape[1:]));rows=[]
    pca=PCA(40,center=False,solver='gram').fit(x)
    encoded=pca.transform(x)
    energy=float(np.sum(x*x))
    for d in range(1,pca.available_rank+1):
        sse=max(0.,energy-float(np.sum(encoded[:,:d]**2)))
        rows.append({'method':'PCA','dimensions':str(d),'coefficients_per_image':d,
                     'compression_ratio':n*D/(d*(n+D)),'paper_rmse':float(np.sqrt(sse/n)),
                     'sse':sse,'stored_scalars':d*(n+D)})
    for r in range(1,41):
        model=CSA((r,r,r),center=False,max_iter=10).fit(x)
        reconstruction=model.inverse_transform(model.transform(x))
        metrics=reconstruction_metrics(x,reconstruction,method='csa',dimensions=(r,r,r))
        rows.append({'method':'CSA-3','dimensions':f'{r}x{r}x{r}','coefficients_per_image':r**3,
                     'compression_ratio':metrics['compression_ratio'],'paper_rmse':metrics['paper_rmse'],
                     'sse':metrics['sse'],'stored_scalars':metrics['stored_scalars']})
        if r%5==0:print(f'ORL reconstruction rank={r}/40',flush=True)
    write_csv(output/'orl_reconstruction.csv',rows)
    convergence=CSA((10,10,10),center=False,max_iter=10,stop_on_convergence=False).fit(x)
    (output/'orl_convergence.json').write_text(json.dumps({'images':'one per subject, s*/1.pgm','early_stop_disabled_for_plot':True,
                                                         'initial_sse':convergence.initial_sse,'history':convergence.history},indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for mode in range(3):
        axes[0].plot(range(1,11),[h['basis_similarity'][mode] for h in convergence.history],marker='o',label=f'Mode {mode+1}')
    axes[0].set(xlabel='Iteration',ylabel='Successive basis similarity',ylim=(0,1.02),title='ORL 10x10x10: convergence diagnostic')
    axes[0].legend()
    for method in ['PCA','CSA-3']:
        points=sorted([r for r in rows if r['method']==method],key=lambda r:r['compression_ratio'])
        axes[1].plot([r['compression_ratio'] for r in points],[r['paper_rmse'] for r in points],marker='.',label=method)
    axes[1].set(xlabel='Compression ratio (including bases)',ylabel='sqrt(SSE / N)',xlim=(0,40),title='ORL reconstruction: group Gabor scale')
    axes[1].legend()
    for axis in axes:axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'orl_convergence_reconstruction.png',dpi=180);plt.close(fig)


def plot_results(output,summary,curves):
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for axis,dataset in zip(axes,['ORL','MNIST G30/P470','MNIST G40/P460']):
        selected=[s for s in summary if s['dataset']==dataset]
        methods=list(dict.fromkeys(s['method'] for s in selected));x=np.arange(len(methods))
        for shift,profile,color in [(-.2,'euclidean_control','#999999'),(0,'group_primary','#d9822b')]:
            points=[next(s for s in selected if s['method']==m and s['profile']==profile) for m in methods]
            axis.bar(x+shift,[s['mean_percent'] for s in points],.2,yerr=[s['std_pp'] for s in points],capsize=2,label=profile,color=color)
        papers=[next(s['paper_percent'] for s in selected if s['method']==m) for m in methods]
        axis.bar(x+.2,papers,.2,label='Paper (different protocol)',color='#2878b5')
        axis.set(xticks=x,xticklabels=[m.replace(' (Gabor)','').replace('+LDA','\n+LDA') for m in methods],ylabel='Accuracy (%)',ylim=(0,100),title=dataset)
        axis.tick_params(axis='x',labelsize=8);axis.legend(fontsize=8,loc='lower right')
    fig.suptitle('Fixed paper dimensions, seeds 20-22; our error bars = split SD')
    fig.tight_layout();fig.savefig(output/'recognition_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for axis,methods in zip(axes,[['PCA+LDA (Gray)','CSA-2+LDA'],['PCA (Gabor)','CSA-3'],['PCA+LDA (Gabor)','CSA-3+LDA']]):
        for method in methods:
            selected=[r for r in curves if r['dataset']=='ORL' and r['profile']=='group_primary' and r['method']==method]
            ds=sorted({r['feature_dim'] for r in selected})
            means=[np.mean([r['accuracy_percent'] for r in selected if r['feature_dim']==d]) for d in ds]
            stds=[np.std([r['accuracy_percent'] for r in selected if r['feature_dim']==d],ddof=1) for d in ds]
            axis.plot(ds,means,label=method);axis.fill_between(ds,np.array(means)-stds,np.array(means)+stds,alpha=.15)
        axis.set(xlabel='LDA dimension',ylabel='Probe accuracy (%)',ylim=(0,100));axis.legend(fontsize=8);axis.grid(alpha=.2)
    fig.suptitle('ORL 2 train / 8 probe: descriptive dimension curves, not parameter selection')
    fig.tight_layout();fig.savefig(output/'orl_lda_curves.png',dpi=180);plt.close(fig)


def pure_csa_curve(data,labels,train,probe,seed,profile,curves):
    ranks_grid=[(2,1,4),(4,1,8),(6,1,10),(8,1,12),(10,1,16),
                (12,1,20),(15,1,26),(16,1,28),(18,1,30)]
    config=settings(profile,'csa','gabor',None)
    for ranks in ranks_grid:
        model=make_model('csa',ranks,config).fit(data[train])
        gallery=transform_batches(model,data,train);queries=transform_batches(model,data,probe)
        predicted,_,_=nn(gallery,queries,labels[train],config['metric'])
        curves.append({'dataset':'ORL','seed':seed,'profile':profile,'method':'CSA-3',
                       'feature_dim':int(np.prod(ranks)),'pre_dims':str(ranks),
                       'accuracy_percent':100*np.mean(predicted==labels[probe])})


def run(output,only):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=BASE/'data/processed/available'
    preprocessing=json.loads((source/'preprocessing.json').read_text())
    if preprocessing['eye_methods'].get('needs_review',0):raise ValueError('Review unresolved eye coordinates before experiments')
    protocol={'time_taipei':datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'scope':'available ORL and balanced MNIST; FERET/PIE not obtained',
              'seeds':SEEDS,'test_used_for_parameter_selection':False,
              'preprocessing':preprocessing,'gabor_group_conventions':CONFIG,
              'profiles':{'euclidean_control':'all uncentered, Sw-normalized LDA, Euclidean NN',
                          'group_primary':'PCA centered, CSA uncentered; unit-column LDA; gray CSA Euclidean, other cosine NN'},
              'recognition_csa':{'max_iter':3,'init':'truncated identity','epsilon':.001},
              'pca_solver':'sample Gram eigendecomposition, conservative numerical-rank check',
              'orl_pure_csa_curve_ranks':[(2,1,4),(4,1,8),(6,1,10),(8,1,12),(10,1,16),(12,1,20),(15,1,26),(16,1,28),(18,1,30)],
              'reconstruction':'s*/1.pgm; uncentered; r=1..40, CSA Tmax=10; PCA d=1..40; no mean storage',
              'limitations':['group eye coordinates, target coordinates and Gabor implementation not confirmed original',
                             'MNIST balanced source indices differ from literal first-half wording',
                             'original random indices unavailable','profiles fixed before this run; ORL was previously explored'],
              'code_sha256':{name:hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                             for name in ['csa.py','gabor.py','prepare_available_data.py','run_available_reproduction.py']},
              'python':platform.python_version()}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    started=time.perf_counter();scores=[];predictions=[];curves=[];histories=[];splits=[]
    if only in ['all','orl']:
        gray=np.load(source/'orl_aligned.npy');labels=np.load(source/'orl_labels.npy')
        names=[f's{p}/{i}.pgm' for p in range(1,41) for i in range(1,11)]
        full=cached_features(gray,source/'orl_gabor_full.npy',1);gabor=full[:,::2,::2,:]
        assert gabor.shape==(400,28,23,40)
        for seed in SEEDS:
            train,probe=split(labels,2,seed)
            splits.extend({'dataset':'ORL','seed':seed,'partition':part,'order':o,'file':names[i],'label':int(labels[i])}
                          for part,indices in [('train_gallery',train),('probe',probe)] for o,i in enumerate(indices))
            for profile in ['euclidean_control','group_primary']:
                for case in ORL_CASES:
                    record_recognition('ORL',seed,profile,case,gray if case[1]=='gray' else gabor,labels,names,train,probe,scores,predictions,curves,histories)
                pure_csa_curve(gabor,labels,train,probe,seed,profile,curves)
        reconstruction(full,output)
    if only in ['all','mnist']:
        gray=np.load(source/'mnist_balanced.npy');labels=np.load(source/'mnist_labels.npy')
        with (source/'mnist_source_indices.csv').open() as handle:index_rows=list(csv.DictReader(handle))
        names=[f"test/{r['test_index_zero_based']}" for r in index_rows]
        gabor=cached_features(gray,source/'mnist_gabor.npy',1)
        assert gabor.shape==(5000,28,28,40)
        for count,pca_dim,ranks,pca_paper,csa_paper in [(30,60,(5,9,18),88.5,92.8),(40,50,(15,5,13),90.5,92.4)]:
            dataset=f'MNIST G{count}/P{500-count}'
            cases=[('PCA (Gabor)','gabor','pca',pca_dim,None,pca_paper),('CSA-3','gabor','csa',ranks,None,csa_paper)]
            for seed in SEEDS:
                train,probe=split(labels,count,seed)
                splits.extend({'dataset':dataset,'seed':seed,'partition':part,'order':o,'file':names[i],'label':int(labels[i])}
                              for part,indices in [('train_gallery',train),('probe',probe)] for o,i in enumerate(indices))
                for profile in ['euclidean_control','group_primary']:
                    for case in cases:
                        record_recognition(dataset,seed,profile,case,gabor,labels,names,train,probe,scores,predictions,curves,histories)
    write_csv(output/'fixed_results.csv',scores);write_csv(output/'predictions.csv',predictions)
    write_csv(output/'splits.csv',splits)
    if curves:write_csv(output/'dimension_curves.csv',curves)
    (output/'recognition_csa_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    summary=[]
    for dataset,profile,method in dict.fromkeys((r['dataset'],r['profile'],r['method']) for r in scores):
        rows=[r for r in scores if (r['dataset'],r['profile'],r['method'])==(dataset,profile,method)]
        values=[r['accuracy_percent'] for r in rows]
        summary.append({'dataset':dataset,'profile':profile,'method':method,'runs':len(rows),
                        'mean_percent':float(np.mean(values)),'std_pp':float(np.std(values,ddof=1)),
                        'min_percent':min(values),'max_percent':max(values),'paper_percent':rows[0]['paper_percent'],
                        'difference_pp':float(np.mean(values))-rows[0]['paper_percent']})
    (output/'summary.json').write_text(json.dumps({'wall_seconds_before_plot':time.perf_counter()-started,'summary':summary},indent=2),encoding='utf-8')
    if only=='all':plot_results(output,summary,curves)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'results/available_reproduction')
    parser.add_argument('--only',choices=['all','orl','mnist'],default='all')
    args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.output,args.only)
