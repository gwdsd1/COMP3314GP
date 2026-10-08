"""FERET Table I / Fig6：训练身份与未见身份的Fa/Fb测试严格隔离。"""

import argparse
import csv
import json
import platform
import shutil
import time
from datetime import datetime,timezone,timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA,PCA,LDA
from gabor import file_hash
from run_orl_gray import write_csv
from prepare_feret import load_pairs,variant_images,EYE_TARGETS,OUTPUT as PREPARED,ROOT

BASE=Path(__file__).resolve().parent
CASES={50:[('PCA+LDA (Gray)',45,25,84.6),('CSA-2+LDA',(10,4),35,88.2)],
       100:[('PCA+LDA (Gray)',95,40,90.6),('CSA-2+LDA',(11,5),44,92.2)],
       200:[('PCA+LDA (Gray)',112,69,93.4),('CSA-2+LDA',(13,7),62,95.2)]}
PROFILES={'group_primary':{'pca_center':True,'csa_center':False,'lda_scaling':'unit_columns','metric':'euclidean'},
          'sw_euclidean_control':{'pca_center':False,'csa_center':False,'lda_scaling':'sw_identity','metric':'euclidean'},
          'cosine_control':{'pca_center':True,'csa_center':False,'lda_scaling':'unit_columns','metric':'cosine'}}


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as handle:return list(csv.DictReader(handle))


def all_prefix_nn(gallery,probes,labels,metric):
    """所有LDA前缀维数的最近邻，逐列累积并按probe分块，不重复矩阵乘法。"""
    gallery=np.asarray(gallery,dtype=float);probes=np.asarray(probes,dtype=float)
    n,d=len(probes),gallery.shape[1]
    neighbors=np.empty((n,d),dtype=int);distances=np.empty((n,d))
    for start in range(0,n,128):
        batch=probes[start:start+128]
        accumulated=np.zeros((len(batch),len(gallery)));gnorm=np.zeros(len(gallery));pnorm=np.zeros(len(batch))
        for k in range(d):
            if metric=='euclidean':
                accumulated+=(batch[:,k,None]-gallery[None,:,k])**2
                matrix=accumulated
            elif metric=='cosine':
                accumulated+=batch[:,k,None]*gallery[None,:,k]
                gnorm+=gallery[:,k]**2;pnorm+=batch[:,k]**2
                denom=np.sqrt(pnorm[:,None]*gnorm[None,:])
                if np.any(denom<=0):raise ValueError('Zero norm in cosine prefix')
                matrix=np.clip(1-accumulated/denom,0,2)
            else:raise ValueError(metric)
            index=np.argmin(matrix,axis=1);value=matrix[np.arange(len(batch)),index]
            neighbors[start:start+len(batch),k]=index
            distances[start:start+len(batch),k]=np.sqrt(value) if metric=='euclidean' else value
    return labels[neighbors],neighbors,distances


def plot(output,rows,curves,primary='group_primary'):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,q in zip(axes,CASES):
        for method in ['PCA+LDA (Gray)','CSA-2+LDA']:
            selected=[r for r in curves if r['q']==q and r['profile']==primary and r['method']==method]
            ax.plot([r['lda_dim'] for r in selected],[r['accuracy_percent'] for r in selected],label=method)
        ax.set(title=f'Train identities={q}; unseen identities={1195-q}',xlabel='LDA dimension',ylabel='Top-one accuracy (%)',ylim=(0,100))
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle(f'FERET official pair order: {primary}; descriptive probe curves')
    fig.tight_layout();fig.savefig(output/'lda_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,q in zip(axes,CASES):
        methods=['PCA+LDA (Gray)','CSA-2+LDA'];x=np.arange(2)
        profiles=[(primary,'Validation selected' if primary=='validation_selected' else 'Fixed baseline','#d9822b')]
        if primary!='group_primary':profiles.append(('group_primary','Fixed baseline','#ab7197'))
        profiles.extend([('sw_euclidean_control','Sw Euclidean control','#aaaaaa'),('cosine_control','Cosine control','#78954a')])
        width=.8/(len(profiles)+1)
        for j,(profile,label,color) in enumerate(profiles):
            offset=(j-len(profiles)/2)*width
            values=[next(r['accuracy_percent'] for r in rows if r['q']==q and r['method']==m and r['profile']==profile) for m in methods]
            ax.bar(x+offset,values,width,label=label,color=color)
        values=[next(r['paper_percent'] for r in rows if r['q']==q and r['method']==m) for m in methods]
        ax.bar(x+len(profiles)*width/2,values,width,label='Paper reference',color='#2878b5')
        ax.set(xticks=x,xticklabels=['PCA+LDA','CSA-2+LDA'],title=f'q={q}',ylabel='Accuracy (%)',ylim=(0,100));ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('Table I fixed dimensions; protocols differ, no test-selected profile')
    fig.tight_layout();fig.savefig(output/'paper_comparison.png',dpi=180);plt.close(fig)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pairs=load_pairs();records=read_csv(PREPARED/'records.csv')
    metadata=json.loads((PREPARED/'preprocessing.json').read_text())
    x=np.load(PREPARED/'aligned.npy');labels=np.load(PREPARED/'labels.npy')
    assert x.shape==(2390,41,27) and np.array_equal(labels[::2],[p['subject_id'] for p in pairs]) and np.array_equal(labels[::2],labels[1::2])
    assert all(file_hash(PREPARED/name)==digest for name,digest in metadata['prepared_sha256'].items())
    source_files=['csa.py','gabor.py','prepare_available_data.py','prepare_feret.py','run_feret.py','run_orl_gray.py']
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in source_files:shutil.copy2(BASE/name,snapshot/name)
    protocol={'time_taipei':datetime.now(timezone(timedelta(hours=8))).isoformat(),'dataset':'original grayscale FERET official1195 Fa/Fb pairs',
              'preprocessing':metadata,'cases':CASES,'profiles':PROFILES,'primary_profile':'group_primary',
              'split':'first q paired identities in preserved official gallery order; BOTH Fa/Fb train; rest Fa gallery and Fb probe',
              'no_test_parameter_selection':True,'CSA':{'max_iter':3,'epsilon':.001,'init':'identity'},
              'curve_role':'descriptive only; fixed Table I dimensions used for reported scores',
              'test_gallery_lda_origin':'training-feature mean, never unseen-gallery mean',
              'limitations':['original first-q ordering unconfirmed','target eyes and exact normalization unconfirmed',
                             'LDA scaling, distance and centering are group conventions','single deterministic partition, no split SD'],
              'code_sha256':{name:file_hash(BASE/name) for name in source_files},'python':platform.python_version(),'BLAS_threads':4}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    started=time.perf_counter();rows=[];predictions=[];curves=[];histories=[];splits=[]
    for q,cases in CASES.items():
        train=np.arange(2*q);gallery=np.arange(2*q,len(x),2);probe=gallery+1
        assert not set(labels[train])&set(labels[gallery]) and len(train)==2*q
        for part,indices in [('subspace_lda_train',train),('unseen_gallery',gallery),('unseen_probe',probe)]:
            splits.extend({'q':q,'partition':part,'index':int(i),'subject_id':str(labels[i]),'role':records[i]['role'],'file':records[i]['file']} for i in indices)
        for profile,config in PROFILES.items():
            for method,pre_dims,lda_dim,paper in cases:
                began=time.perf_counter();is_pca=method.startswith('PCA')
                center=config['pca_center'] if is_pca else config['csa_center']
                model=PCA(pre_dims,center=center,solver='gram') if is_pca else CSA(pre_dims,center=center,max_iter=3)
                model.fit(x[train]);training=model.transform(x[train]).reshape(len(train),-1)
                assert training.shape[1]<=len(train)-q
                lda=LDA().fit(training,labels[train]);assert lda_dim<=lda.max_components
                basis=lda.basis.copy()
                if config['lda_scaling']=='unit_columns':basis/=np.linalg.norm(basis,axis=0)
                g=(model.transform(x[gallery]).reshape(len(gallery),-1)-lda.mean)@basis
                p=(model.transform(x[probe]).reshape(len(probe),-1)-lda.mean)@basis
                predicted,neighbors,distances=all_prefix_nn(g,p,labels[gallery],config['metric'])
                # Fixed score independently checked against SciPy distances.
                matrix=cdist(p[:,:lda_dim],g[:,:lda_dim],config['metric']);reference=np.argmin(matrix,axis=1)
                assert np.array_equal(reference,neighbors[:,lda_dim-1])
                assert np.allclose(matrix[np.arange(len(probe)),reference],distances[:,lda_dim-1],atol=1e-7)
                for d in range(1,lda.max_components+1):
                    curves.append({'q':q,'profile':profile,'method':method,'pre_dims':str(pre_dims),'lda_dim':d,
                                   'accuracy_percent':100*float(np.mean(predicted[:,d-1]==labels[probe]))})
                chosen=predicted[:,lda_dim-1];correct=int(np.sum(chosen==labels[probe]));accuracy=100*correct/len(probe)
                rows.append({'q':q,'training_images':len(train),'test_identities':len(probe),'profile':profile,'method':method,
                             'pre_dims':str(pre_dims),'lda_dim':lda_dim,'center':center,'metric':config['metric'],
                             'lda_scaling':config['lda_scaling'],'sw_rank':lda.sw_rank,'correct':correct,'probe_count':len(probe),
                             'accuracy_percent':accuracy,'paper_percent':paper,'difference_pp':accuracy-paper,
                             'wall_seconds':time.perf_counter()-began})
                predictions.extend({'q':q,'profile':profile,'method':method,'probe_index':int(i),'probe':records[i]['file'],
                                    'true':str(labels[i]),'predicted':str(label),'nearest_gallery_index':int(gallery[n]),
                                    'nearest_gallery':records[gallery[n]]['file'],'distance':float(distance),'correct':int(label==labels[i])}
                                   for i,label,n,distance in zip(probe,chosen,neighbors[:,lda_dim-1],distances[:,lda_dim-1]))
                if not is_pca:histories.append({'q':q,'profile':profile,'method':method,'history':model.history,'stop_reason':model.stop_reason})
                print(f'q={q} {profile} {method}: {accuracy:.4f}% vs paper {paper}%',flush=True)
    write_csv(output/'fixed_results.csv',rows);write_csv(output/'predictions.csv',predictions)
    write_csv(output/'splits.csv',splits);write_csv(output/'lda_dimension_curves.csv',curves)
    (output/'csa_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps({'wall_seconds_before_plot':time.perf_counter()-started,'results':rows},indent=2),encoding='utf-8')
    plot(output,rows,curves)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'results/feret_official_pairs')
    with threadpool_limits(limits=4):run(parser.parse_args().output)
