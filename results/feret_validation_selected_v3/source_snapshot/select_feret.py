"""仅在前q训练身份内五折验证选最高准确率，再重训并一次评估未见身份。"""

import argparse
import itertools
import json
import platform
import shutil
import time
from pathlib import Path
from datetime import datetime,timezone,timedelta

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import PCA,CSA,LDA
from gabor import file_hash
from prepare_feret import variant_images,EYE_TARGETS,OUTPUT as PREPARED
from run_feret import CASES,read_csv,all_prefix_nn,plot,BASE
from run_orl_gray import write_csv

FOLD_COUNTS={50:10,100:20,200:10}


def candidates(method):
    is_pca=method.startswith('PCA');result=[]
    centers=[True,False] if is_pca else [False,True]
    inits=['identity'] if is_pca else ['identity','random']
    for eye,norm,center,init,scaling,metric in itertools.product(EYE_TARGETS,['equalized','zscore'],centers,inits,
                                                               ['unit_columns','sw_identity'],['euclidean','cosine']):
        result.append({'candidate_id':f'c{len(result):03d}','eye_target':eye,'normalization':norm,
                       'center':center,'init':init,'lda_scaling':scaling,'metric':metric})
    return result


def fit_and_encode(x,labels,fit_indices,gallery,probe,method,pre_dims,config):
    is_pca=method.startswith('PCA')
    model=PCA(pre_dims,center=config['center'],solver='gram') if is_pca else CSA(pre_dims,center=config['center'],init=config['init'],seed=3314,max_iter=3)
    model.fit(x[fit_indices]);features=model.transform(x[fit_indices]).reshape(len(fit_indices),-1)
    lda=LDA().fit(features,labels[fit_indices]);basis=lda.basis.copy()
    if config['lda_scaling']=='unit_columns':basis/=np.linalg.norm(basis,axis=0)
    g=(model.transform(x[gallery]).reshape(len(gallery),-1)-lda.mean)@basis
    p=(model.transform(x[probe]).reshape(len(probe),-1)-lda.mean)@basis
    return model,lda,g,p


def evaluate_selected(output,q,case,config,x,labels,records,profile,rows,predictions,curves,histories):
    method,pre_dims,lda_dim,paper=case
    train=np.arange(2*q);gallery=np.arange(2*q,len(labels),2);probe=gallery+1
    began=time.perf_counter();model,lda,g,p=fit_and_encode(x,labels,train,gallery,probe,method,pre_dims,config)
    predicted,neighbors,distances=all_prefix_nn(g,p,labels[gallery],config['metric'])
    reference=cdist(p[:,:lda_dim],g[:,:lda_dim],config['metric']);neighbor=np.argmin(reference,axis=1)
    assert np.array_equal(neighbor,neighbors[:,lda_dim-1])
    assert np.allclose(reference[np.arange(len(probe)),neighbor],distances[:,lda_dim-1],atol=1e-7)
    for d in range(1,lda.max_components+1):
        curves.append({'q':q,'profile':profile,'method':method,'pre_dims':str(pre_dims),'lda_dim':d,
                       'accuracy_percent':100*float(np.mean(predicted[:,d-1]==labels[probe]))})
    chosen=predicted[:,lda_dim-1];correct=int(np.sum(chosen==labels[probe]));accuracy=100*correct/len(probe)
    rows.append({'q':q,'training_images':2*q,'test_identities':len(probe),'profile':profile,'method':method,
                 'pre_dims':str(pre_dims),'lda_dim':lda_dim,**config,'sw_rank':lda.sw_rank,'correct':correct,
                 'probe_count':len(probe),'accuracy_percent':accuracy,'paper_percent':paper,'difference_pp':accuracy-paper,
                 'wall_seconds':time.perf_counter()-began})
    predictions.extend({'q':q,'profile':profile,'method':method,'probe_index':int(i),'probe':records[i]['file'],
                        'true':str(labels[i]),'predicted':str(label),'nearest_gallery_index':int(gallery[j]),
                        'nearest_gallery':records[gallery[j]]['file'],'distance':float(distance),'correct':int(label==labels[i])}
                       for i,label,j,distance in zip(probe,chosen,neighbors[:,lda_dim-1],distances[:,lda_dim-1]))
    if isinstance(model,CSA):histories.append({'q':q,'profile':profile,'method':method,'history':model.history,'stop_reason':model.stop_reason})
    print(f'FINAL q={q} {profile} {method}: {accuracy:.4f}% (paper {paper}%)',flush=True)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    labels=np.load(PREPARED/'labels.npy');records=read_csv(PREPARED/'records.csv')
    preprocess=json.loads((PREPARED/'preprocessing.json').read_text())
    assert all(file_hash(PREPARED/name)==digest for name,digest in preprocess['prepared_sha256'].items())
    source_files=['csa.py','gabor.py','prepare_available_data.py','prepare_feret.py','run_feret.py','select_feret.py','run_orl_gray.py']
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in source_files:shutil.copy2(BASE/name,snapshot/name)
    grids={method:candidates(method) for method in ['PCA+LDA (Gray)','CSA-2+LDA']}
    protocol={'time_taipei':datetime.now(timezone(timedelta(hours=8))).isoformat(),'preprocessing':preprocess,'cases':CASES,
              'selection':'maximize pooled training-only identity-validation accuracy; ties maximize worst-fold accuracy, then prefer lowest candidate number',
              'fold_seed':'3314+q; permute first q identities; split using fold_counts',
              'fold_counts':FOLD_COUNTS,
              'candidate_grids':grids,'eye_targets':EYE_TARGETS,'CSA_seed':3314,'CSA_max_iter':3,'CSA_epsilon':.001,
              'PCA_validation_cap':'none required: inner training identities45/95/180 support paper dimensions45/95/112',
              'validation_gallery':'ALL first-q Fa images; inner-trained identity Fa added as distractors; target probe identities excluded from fitting',
              'tie_break':'highest worst-fold accuracy, then lowest candidate number; margin recorded descriptively, not optimized',
              'chronology':'v3 follows prior v1/v2 evaluations: removes validation PCA dimension surrogate and uses conservative tie-break; prior results retained; no final test scores enter candidate objective',
              'primary_profile':'validation_selected','test_used_for_parameter_selection':False,
              'test_protocol':'first q pairs train using Fa+Fb; remaining identities Fa gallery/Fb probe',
              'curve_role':'descriptive only AFTER selection; fixed paper final LDA dimensions',
              'comparison_profiles':['group_primary fixed baseline','sw_euclidean_control','cosine_control'],
              'limitations':['finite candidate grid, not guaranteed global optimum','small training-only validation gallery',
                             'original first-q ordering and normalization details unconfirmed','validation-selected PCA/CSA may differ in metric/preprocessing',
                             'validation gallery includes trained-identity distractors; final gallery has only unseen identities',
                             'v3 is not a fresh blind test after previous runs; first-q training remains the only selection data'],
              'python':platform.python_version(),'BLAS_threads':4,'code_sha256':{name:file_hash(BASE/name) for name in source_files}}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    began=time.perf_counter();validation=[];candidate_scores=[];validation_predictions=[];fold_records=[];selected=[]
    # Only first200 identities enter preprocessing candidates before selection.
    variants={(eye,norm):variant_images(eye,norm,limit=400) for eye in EYE_TARGETS for norm in ['equalized','zscore']}
    for q,cases in CASES.items():
        fold_subjects=np.array_split(np.random.default_rng(3314+q).permutation(q),FOLD_COUNTS[q])
        for fold,held in enumerate(fold_subjects):
            fit_people=np.setdiff1d(np.arange(q),held)
            fit=np.sort(np.r_[2*fit_people,2*fit_people+1]);gallery=2*np.arange(q);probe=2*held+1
            assert not set(labels[fit])&set(labels[probe]) and max(probe)<2*q
            fold_records.extend({'q':q,'fold':fold,'partition':part,'index':int(i),'subject_id':str(labels[i]),'role':records[i]['role']}
                                for part,indices in [('inner_train',fit),('validation_gallery',gallery),('validation_probe',probe)] for i in indices)
            for method,paper_dims,lda_dim,paper in cases:
                dims=paper_dims
                cache={}
                for config in grids[method]:
                    cache_key=(config['eye_target'],config['normalization'],config['center'],config['init'])
                    if cache_key not in cache:
                        cache[cache_key]=fit_and_encode(variants[config['eye_target'],config['normalization']],labels,
                                                        fit,gallery,probe,method,dims,{**config,'lda_scaling':'sw_identity'})
                    model,lda,g,p=cache[cache_key]
                    assert lda_dim<=lda.max_components
                    if config['lda_scaling']=='unit_columns':
                        lengths=np.linalg.norm(lda.basis,axis=0);g_use=g[:,:lda_dim]/lengths[:lda_dim];p_use=p[:,:lda_dim]/lengths[:lda_dim]
                    else:g_use,p_use=g[:,:lda_dim],p[:,:lda_dim]
                    matrix=cdist(p_use,g_use,config['metric'])
                    if not np.isfinite(matrix).all():raise ValueError('Nonfinite validation distances')
                    neighbor=np.argmin(matrix,axis=1);predicted=labels[gallery[neighbor]]
                    correct=int(np.sum(predicted==labels[probe]))
                    target=np.argmax(labels[probe,None]==labels[gallery][None,:],axis=1)
                    positive=matrix[np.arange(len(probe)),target]
                    wrong=matrix.copy();wrong[np.arange(len(probe)),target]=np.inf
                    negative=wrong.min(axis=1)
                    margin=(negative-positive)/(negative+positive+1e-12)
                    validation.append({'q':q,'fold':fold,'method':method,**config,'inner_pre_dims':str(dims),'lda_dim':lda_dim,
                                       'correct':correct,'probe_count':len(probe),'accuracy_percent':100*correct/len(probe),
                                       'margin_sum':float(margin.sum())})
                    validation_predictions.extend({'q':q,'fold':fold,'method':method,'candidate_id':config['candidate_id'],
                                                   'probe_index':int(i),'true':str(labels[i]),'predicted':str(value),
                                                   'nearest_gallery_index':int(gallery[j]),'correct':int(value==labels[i]),'margin':float(m)}
                                                  for i,value,j,m in zip(probe,predicted,neighbor,margin))
                if (fold+1)%5==0:print(f'validation q={q} fold={fold+1}/{FOLD_COUNTS[q]} {method} complete',flush=True)
        for method,_,_,_ in cases:
            for config in grids[method]:
                rows=[r for r in validation if r['q']==q and r['method']==method and r['candidate_id']==config['candidate_id']]
                total=sum(r['correct'] for r in rows);count=sum(r['probe_count'] for r in rows)
                assert len(rows)==FOLD_COUNTS[q] and count==q
                candidate_scores.append({'q':q,'method':method,**config,'validation_correct':total,'validation_probes':count,
                                         'validation_percent':100*total/count,'validation_mean_margin':sum(r['margin_sum'] for r in rows)/count,
                                         'worst_fold_percent':min(r['accuracy_percent'] for r in rows)})
            scores=[r for r in candidate_scores if r['q']==q and r['method']==method]
            winner=min(scores,key=lambda r:(-r['validation_correct'],-r['worst_fold_percent'],r['candidate_id']))
            selected.append(winner)
            print(f'SELECTED q={q} {method}: {winner}',flush=True)
    # Persist every choice before any final test classification.
    write_csv(output/'validation_folds.csv',fold_records);write_csv(output/'validation_results.csv',validation)
    write_csv(output/'validation_predictions.csv',validation_predictions);write_csv(output/'candidate_scores.csv',candidate_scores)
    (output/'selected_configs.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    selection_seconds=time.perf_counter()-began
    # Final scoring uses no further hyperparameter choices.
    final_variants={};rows=[];predictions=[];curves=[];histories=[];splits=[]
    baseline={'candidate_id':'fixed_baseline','eye_target':'wide','normalization':'equalized','center':False,'init':'identity','lda_scaling':'unit_columns','metric':'euclidean'}
    for q,cases in CASES.items():
        for part,indices in [('subspace_lda_train',np.arange(2*q)),('unseen_gallery',np.arange(2*q,2390,2)),('unseen_probe',np.arange(2*q+1,2390,2))]:
            splits.extend({'q':q,'partition':part,'index':int(i),'subject_id':str(labels[i]),'role':records[i]['role'],'file':records[i]['file']} for i in indices)
        for case in cases:
            method=case[0];winner=next(r for r in selected if r['q']==q and r['method']==method)
            chosen={k:winner[k] for k in baseline}
            configs=[('validation_selected',chosen),('group_primary',{**baseline,'center':method.startswith('PCA')}),
                     ('sw_euclidean_control',{**baseline,'lda_scaling':'sw_identity'}),
                     ('cosine_control',{**baseline,'center':method.startswith('PCA'),'metric':'cosine'})]
            for profile,config in configs:
                variant=(config['eye_target'],config['normalization'])
                if variant not in final_variants:
                    data=variant_images(*variant);filename=f"features_{variant[0]}_{variant[1]}.npy"
                    np.save(output/filename,data);final_variants[variant]=data
                evaluate_selected(output,q,case,config,final_variants[variant],labels,records,profile,rows,predictions,curves,histories)
    write_csv(output/'fixed_results.csv',rows);write_csv(output/'predictions.csv',predictions)
    write_csv(output/'splits.csv',splits);write_csv(output/'lda_dimension_curves.csv',curves)
    (output/'csa_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps({'validation_seconds':selection_seconds,'wall_seconds_before_plot':time.perf_counter()-began,
                                                 'results':rows},indent=2),encoding='utf-8')
    (output/'feature_hashes.json').write_text(json.dumps({p.name:file_hash(p) for p in output.glob('features_*.npy')},indent=2),encoding='utf-8')
    plot(output,rows,curves,primary='validation_selected')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'results/feret_validation_selected_v3')
    with threadpool_limits(limits=4):run(parser.parse_args().output)
