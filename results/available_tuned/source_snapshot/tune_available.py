"""ORL/MNIST训练内两阶段搜索；固定论文维数、原2/8与G30/G40最终划分。"""

import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime,timezone,timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageOps
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import gabor
from csa import PCA,CSA,LDA
from gabor import file_hash
from prepare_available_data import alignment_matrix
from run_available_reproduction import ORL_CASES,SEEDS,split
from run_orl_gray import write_csv
from run_feret import read_csv

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'data/processed/available'
CACHE=BASE/'data/processed/tuning'
sys.path.insert(0,str(BASE/'.deps'))
import cv2

GEOMETRY={'original':[[12.,22.],[34.,22.]],'medium':[[14.,23.],[32.,23.]],'fuller':[[15.,23.],[31.,23.]]}
KERNELS={'standard4':4.,'smaller2':2.}


def sources():
    CACHE.mkdir(parents=True,exist_ok=True)
    eyes=read_csv(BASE/'data/annotations/orl_eye_coordinates.csv')
    result={('ORL','original','gray'):np.load(SOURCE/'orl_aligned.npy')}
    for name,target in GEOMETRY.items():
        if name=='original':continue
        images=[]
        for r in eyes:
            with Image.open(BASE/'data/raw/orl'/r['file']) as im:raw=np.asarray(im.convert('L'))
            points=np.array([[float(r[f'{side}_x']),float(r[f'{side}_y'])] for side in ['left','right']])
            transform=alignment_matrix(points,np.array(target))
            assert np.max(np.abs(np.column_stack([points,np.ones(2)])@transform.T-np.array(target)))<1e-10
            warp=cv2.warpAffine(raw,transform,(46,56),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            images.append(np.asarray(ImageOps.equalize(Image.fromarray(warp)),dtype=np.float32))
        result['ORL',name,'gray']=np.stack(images)
    result['MNIST','unchanged','gray']=np.load(SOURCE/'mnist_balanced.npy')
    return result


def get_features(collection,dataset,geometry,kind,kernel):
    if kind=='gray':return collection[dataset,geometry,'gray']
    key=dataset,geometry,kernel
    if key in collection:return collection[key]
    gray=collection[dataset,geometry,'gray'];destination=CACHE/f'{dataset}_{geometry}_{kernel}.npy'
    metadata=destination.with_suffix('.json')
    spec={'input_sha256':__import__('hashlib').sha256(np.ascontiguousarray(gray).tobytes()).hexdigest(),
          'wavelength_base':KERNELS[kernel],'sigma':2*np.pi,'five_scales':True,'eight_orientations':True,
          'step':2 if dataset=='ORL' else 1,'gabor_sha256':file_hash(BASE/'gabor.py'),
          'tuner_sha256':file_hash(BASE/'tune_available.py')}
    if destination.exists() and metadata.exists() and json.loads(metadata.read_text())['spec']==spec:
        assert file_hash(destination)==json.loads(metadata.read_text())['sha256']
    else:
        if kernel=='standard4' and geometry in ['original','unchanged']:
            old=np.load(SOURCE/('orl_gabor_full.npy' if dataset=='ORL' else 'mnist_gabor.npy'),mmap_mode='r')
            values=old[:,::2,::2,:] if dataset=='ORL' else old
        else:
            # Only this process temporarily changes the parameter dictionary; restore it immediately.
            saved=gabor.CONFIG
            try:
                gabor.CONFIG={**saved,'wavelengths':[float(KERNELS[kernel]*np.sqrt(2)**s) for s in range(5)]}
                values=gabor.features(gray,step=spec['step'])
            finally:gabor.CONFIG=saved
        np.save(destination,values);del values
        metadata.write_text(json.dumps({'spec':spec,'sha256':file_hash(destination)},indent=2),encoding='utf-8')
    collection[key]=np.load(destination,mmap_mode='r')
    print(f'Feature ready: {key}',flush=True)
    return collection[key]


def batch(data,indices,config):
    x=np.asarray(data[indices],dtype=np.float64)
    if config['response']=='sqrt':x=np.sqrt(np.maximum(x,0))
    if config['normalization']=='l2':
        norm=np.linalg.norm(x.reshape(len(x),-1),axis=1)
        if np.any(norm<1e-12):raise ValueError('Zero feature norm')
        x/=norm.reshape((-1,)+(1,)*(x.ndim-1))
    return x


def transform(model,data,indices,config):
    return np.concatenate([model.transform(batch(data,indices[s:s+128],config)).reshape(len(indices[s:s+128]),-1)
                           for s in range(0,len(indices),128)])


def internal_folds(dataset,labels,train,seed):
    rng=np.random.default_rng(603314+seed);folds=[]
    if dataset=='ORL':
        for block,held in enumerate(np.array_split(rng.permutation(np.unique(labels[train])),4)):
            fitting=train[~np.isin(labels[train],held)]
            pairs=[train[labels[train]==c] for c in held]
            for direction in [0,1]:
                query=np.array([p[direction] for p in pairs]);target=np.array([p[1-direction] for p in pairs])
                gallery=np.r_[fitting,target]
                folds.append((2*block+direction,fitting,gallery,query))
    else:
        # Two reproducible 80/20 stratified holdouts inside the original gallery.
        for fold in range(2):
            fitting=[];query=[]
            for c in np.unique(labels[train]):
                order=rng.permutation(train[labels[train]==c]);cut=int(.8*len(order))
                fitting.extend(order[:cut]);query.extend(order[cut:])
            fitting=np.array(fitting);folds.append((fold,fitting,fitting,np.array(query)))
    for _,fit,gallery,query in folds:
        assert set(fit)|set(gallery)|set(query)<=set(train)
        assert not set(fit)&set(query) and not set(gallery)&set(query)
    return folds


def default_config(case):
    method,kind,algorithm,_,lda_dim,_=case
    return {'geometry':'original','kernel':'standard4','response':'linear','normalization':'none',
            'center':algorithm=='pca','init':'identity','scaling':'unit_columns',
            'metric':'euclidean' if kind=='gray' and algorithm=='csa' else 'cosine'}


def candidate_grid(stage,dataset,case,chosen=None):
    base=default_config(case);kind,algorithm=case[1],case[2]
    if dataset=='MNIST':base['geometry']='unchanged'
    configurations=[]
    if stage=='preprocess':
        geometries=list(GEOMETRY) if dataset=='ORL' else ['unchanged']
        kernels=list(KERNELS) if kind=='gabor' else ['standard4']
        responses=['linear','sqrt'] if kind=='gabor' else ['linear']
        for geo,kernel,response,norm,metric in itertools.product(geometries,kernels,responses,['none','l2'],[base['metric'],'cosine' if base['metric']=='euclidean' else 'euclidean']):
            configurations.append({**base,'geometry':geo,'kernel':kernel,'response':response,'normalization':norm,'metric':metric})
    else:
        base={**chosen}
        for center,init,scaling,metric in itertools.product([base['center'],not base['center']],
                    ['identity'] if algorithm=='pca' else ['identity','random'],
                    ['unit_columns','sw_identity'] if case[4] else ['unit_columns'],
                    [base['metric'],'cosine' if base['metric']=='euclidean' else 'euclidean']):
            configurations.append({**base,'center':center,'init':init,'scaling':scaling,'metric':metric})
    return [{**c,'candidate_id':f'{stage}_{i:03d}'} for i,c in enumerate(configurations)]


def fitted(dataset,case,data,labels,fit,config,inner):
    dims=case[3]
    if case[2]=='pca' and inner:
        bound=len(fit)-len(np.unique(labels[fit])) if case[4] else len(fit)-1
        dims=min(dims,bound)
    model=PCA(dims,center=config['center'],solver='gram') if case[2]=='pca' else CSA(dims,center=config['center'],init=config['init'],seed=3314,max_iter=3)
    model.fit(batch(data,fit,config))
    training=transform(model,data,fit,config)
    lda=LDA().fit(training,labels[fit]) if case[4] else None
    if lda:assert case[4]<=lda.max_components
    return model,lda,dims


def coordinates(features,lda,case,config):
    if not lda:return features
    basis=lda.basis[:,:case[4]].copy()
    if config['scaling']=='unit_columns':basis/=np.linalg.norm(basis,axis=0)
    return (features-lda.mean)@basis


def classify(gallery,queries,labels,metric):
    predictions=[];neighbors=[];distances=[]
    for start in range(0,len(queries),256):
        matrix=cdist(queries[start:start+256],gallery,metric)
        if not np.isfinite(matrix).all():raise ValueError('Nonfinite distance')
        index=np.argmin(matrix,axis=1)
        predictions.extend(labels[index]);neighbors.extend(index);distances.extend(matrix[np.arange(len(index)),index])
    return np.array(predictions),np.array(neighbors),np.array(distances)


def choose(dataset,seed,case,labels,train,collection,stage,validation,validation_predictions,candidate_scores,fold_records,chosen=None):
    folds=internal_folds(dataset,labels,train,seed);grid=candidate_grid(stage,dataset,case,chosen)
    if stage=='preprocess':
        fold_records.extend({'dataset':dataset,'seed':seed,'fold':fold,'partition':part,'index':int(i),'label':str(labels[i])}
                            for fold,fit,gallery,query in folds for part,indices in [('inner_fit',fit),('validation_gallery',gallery),('validation_probe',query)] for i in indices)
    cache={}
    for config in grid:
        scores=[]
        source=get_features(collection,'ORL' if dataset=='ORL' else 'MNIST',config['geometry'],case[1],config['kernel'])
        for fold,fit,gallery,query in folds:
            fit_key=(tuple(fit),config['geometry'],config['kernel'],config['response'],config['normalization'],config['center'],config['init'])
            if fit_key not in cache:
                model,lda,dims=fitted(dataset,case,source,labels,fit,config,True)
                # Encode full original gallery once; both ORL directions reuse it.
                cache[fit_key]=(lda,dims,transform(model,source,train,config))
            lda,dims,encoded=cache[fit_key]
            positions={int(i):j for j,i in enumerate(train)}
            g=coordinates(encoded[[positions[int(i)] for i in gallery]],lda,case,config)
            p=coordinates(encoded[[positions[int(i)] for i in query]],lda,case,config)
            predicted,neighbor,_=classify(g,p,labels[gallery],config['metric']);correct=int(np.sum(predicted==labels[query]))
            row={'dataset':dataset,'seed':seed,'method':case[0],'stage':stage,'fold':fold,**config,'inner_pre_dims':str(dims),
                 'correct':correct,'probe_count':len(query),'accuracy_percent':100*correct/len(query)}
            validation.append(row);scores.append(row)
            validation_predictions.extend({'dataset':dataset,'seed':seed,'method':case[0],'stage':stage,'candidate_id':config['candidate_id'],
                                           'fold':fold,'probe_index':int(i),'true':str(labels[i]),'predicted':str(v),
                                           'nearest_gallery_index':int(gallery[j]),'correct':int(v==labels[i])}
                                          for i,v,j in zip(query,predicted,neighbor))
        total=sum(r['correct'] for r in scores);count=sum(r['probe_count'] for r in scores)
        candidate_scores.append({'dataset':dataset,'seed':seed,'method':case[0],'stage':stage,**config,
                                 'validation_correct':total,'validation_probes':count,'validation_percent':100*total/count,
                                 'worst_fold_percent':min(r['accuracy_percent'] for r in scores)})
    eligible=[r for r in candidate_scores if r['dataset']==dataset and r['seed']==seed and r['method']==case[0] and r['stage']==stage]
    winner=min(eligible,key=lambda r:(-r['validation_percent'],-r['worst_fold_percent'],r['candidate_id']))
    print(f'SELECT {dataset} seed={seed} {case[0]} {stage}: {winner}',flush=True)
    return {k:winner[k] for k in grid[0]},winner


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source_files=['tune_available.py','csa.py','gabor.py','prepare_available_data.py','run_available_reproduction.py','run_orl_gray.py','run_feret.py']
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in source_files:shutil.copy2(BASE/name,snapshot/name)
    protocol={'time_taipei':datetime.now(timezone(timedelta(hours=8))).isoformat(),'seeds':SEEDS,'geometry':GEOMETRY,'kernel_bases':KERNELS,
              'fixed_cases':ORL_CASES,'selection':'two-stage training-only coordinate search: preprocessing first, model settings second',
              'tie_break':'highest pooled accuracy, highest worst-fold accuracy, lowest candidate number',
              'ORL_validation':'four held-out identity blocks; two probe/gallery directions; all other training images are distractors',
              'MNIST_validation':'two deterministic80/20 stratified holdouts within each original gallery',
              'validation_seed':'603314+outer_seed','CSA':{'max_iter':3,'epsilon':.001,'random_seed':3314},
              'normalization':'optional per-tensor L2, no learned test statistics','response':'linear magnitude or sqrt magnitude',
              'gabor':'5 scales,8 orientations,sigma2pi,DC-free complex magnitude,reflect; before downsample',
              'original_test_for_selection':False,'chronology':'previous results already observed; this is further development, not a fresh blind test',
              'limits':['finite staged search is not joint global optimization','ORL inner PCA caps and 1-gallery surrogate',
                        'method-specific preprocessing can confound algorithm comparisons','sqrt magnitude is a feature-processing extension'],
              'inputs_sha256':{name:file_hash(SOURCE/name) for name in ['orl_aligned.npy','orl_labels.npy','mnist_balanced.npy','mnist_labels.npy','orl_gabor_full.npy','mnist_gabor.npy']},
              'eye_annotations_sha256':file_hash(BASE/'data/annotations/orl_eye_coordinates.csv'),
              'code_sha256':{name:file_hash(BASE/name) for name in source_files}}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    collection=sources();validation=[];val_predictions=[];candidate_scores=[];fold_records=[];selected=[];units=[];splits=[]
    for dataset,filename,count,cases in [('ORL','orl_labels.npy',2,ORL_CASES),
          ('MNIST G30/P470','mnist_labels.npy',30,[('PCA (Gabor)','gabor','pca',60,None,88.5),('CSA-3','gabor','csa',(5,9,18),None,92.8)]),
          ('MNIST G40/P460','mnist_labels.npy',40,[('PCA (Gabor)','gabor','pca',50,None,90.5),('CSA-3','gabor','csa',(15,5,13),None,92.4)])]:
        labels=np.load(SOURCE/filename)
        for seed in SEEDS:
            train,probe=split(labels,count,seed)
            splits.extend({'dataset':dataset,'seed':seed,'partition':part,'index':int(i),'label':str(labels[i])}
                          for part,indices in [('train_gallery',train),('probe',probe)] for i in indices)
            for case in cases:units.append((dataset,seed,case,labels,train,probe))
    started=time.perf_counter()
    for dataset,seed,case,labels,train,probe in units:
        a,aw=choose(dataset,seed,case,labels,train,collection,'preprocess',validation,val_predictions,candidate_scores,fold_records)
        b,bw=choose(dataset,seed,case,labels,train,collection,'model',validation,val_predictions,candidate_scores,fold_records,a)
        selected.append({'dataset':dataset,'seed':seed,'method':case[0],'config':b,'preprocess_validation_percent':aw['validation_percent'],
                         'validation_percent':bw['validation_percent']})
    write_csv(output/'validation_folds.csv',list({(r['dataset'],r['seed'],r['fold'],r['partition'],r['index']):r for r in fold_records}.values()))
    write_csv(output/'validation_results.csv',validation);write_csv(output/'validation_predictions.csv',val_predictions)
    write_csv(output/'candidate_scores.csv',candidate_scores);write_csv(output/'splits.csv',splits)
    (output/'selected_configs.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    selection_seconds=time.perf_counter()-started
    # Freeze choices before evaluating any final probe in this run.
    rows=[];predictions=[];histories=[]
    for dataset,seed,case,labels,train,probe in units:
        config=next(s['config'] for s in selected if (s['dataset'],s['seed'],s['method'])==(dataset,seed,case[0]))
        data=get_features(collection,'ORL' if dataset=='ORL' else 'MNIST',config['geometry'],case[1],config['kernel'])
        model,lda,dims=fitted(dataset,case,data,labels,train,config,False)
        g=coordinates(transform(model,data,train,config),lda,case,config);p=coordinates(transform(model,data,probe,config),lda,case,config)
        predicted,neighbor,distance=classify(g,p,labels[train],config['metric']);correct=int(np.sum(predicted==labels[probe]))
        accuracy=100*correct/len(probe)
        rows.append({'dataset':dataset,'seed':seed,'method':case[0],'pre_dims':str(dims),'lda_dim':case[4] or '',**config,
                     'correct':correct,'probe_count':len(probe),'accuracy_percent':accuracy,'paper_percent':case[5],'difference_pp':accuracy-case[5]})
        predictions.extend({'dataset':dataset,'seed':seed,'method':case[0],'probe_index':int(i),'true':str(labels[i]),'predicted':str(v),
                            'nearest_gallery_index':int(train[j]),'distance':float(d),'correct':int(v==labels[i])}
                           for i,v,j,d in zip(probe,predicted,neighbor,distance))
        if isinstance(model,CSA):histories.append({'dataset':dataset,'seed':seed,'method':case[0],'history':model.history})
        print(f'FINAL {dataset} seed={seed} {case[0]} {accuracy:.4f}% vs paper {case[5]}%',flush=True)
    write_csv(output/'fixed_results.csv',rows);write_csv(output/'predictions.csv',predictions)
    (output/'csa_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    previous=json.loads((BASE/'results/available_reproduction_v2/summary.json').read_text())['summary'];summary=[]
    for dataset,method in dict.fromkeys((r['dataset'],r['method']) for r in rows):
        rs=[r for r in rows if r['dataset']==dataset and r['method']==method];values=[r['accuracy_percent'] for r in rs]
        old=next(r for r in previous if r['dataset']==dataset and r['method']==method and r['profile']=='group_primary')
        summary.append({'dataset':dataset,'method':method,'mean_percent':float(np.mean(values)),'std_pp':float(np.std(values,ddof=1)),
                        'paper_percent':rs[0]['paper_percent'],'previous_percent':old['mean_percent'],
                        'improvement_pp':float(np.mean(values))-old['mean_percent'],'difference_pp':float(np.mean(values))-rs[0]['paper_percent']})
    (output/'summary.json').write_text(json.dumps({'selection_seconds':selection_seconds,'wall_seconds_before_plot':time.perf_counter()-started,'summary':summary},indent=2),encoding='utf-8')
    (output/'feature_manifests.json').write_text(json.dumps({p.name:json.loads(p.read_text()) for p in CACHE.glob('*.json')},indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for axis,dataset in zip(axes,['ORL','MNIST G30/P470','MNIST G40/P460']):
        points=[r for r in summary if r['dataset']==dataset];x=np.arange(len(points))
        axis.bar(x-.25,[r['previous_percent'] for r in points],.25,label='Previous declared profile',color='#999999')
        axis.bar(x,[r['mean_percent'] for r in points],.25,yerr=[r['std_pp'] for r in points],capsize=2,label='Training-selected',color='#d9822b')
        axis.bar(x+.25,[r['paper_percent'] for r in points],.25,label='Paper reference',color='#2878b5')
        axis.set(xticks=x,xticklabels=[r['method'].replace(' (Gabor)','').replace('+LDA','\n+LDA') for r in points],title=dataset,ylabel='Accuracy (%)',ylim=(0,100))
        axis.tick_params(axis='x',labelsize=8);axis.legend(loc='lower right',fontsize=8)
    fig.suptitle('Fixed paper dimensions, original splits; training-only staged tuning')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'results/available_tuned')
    with threadpool_limits(limits=4):run(parser.parse_args().output)
