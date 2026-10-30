"""Consolidate the two completed gray recovery batches without refitting."""
import json
from pathlib import Path

import numpy as np

import run_orl_gray_recovery as R


def main():
    root=R.BASE/'results/orl_gray_recovery_20261030'
    sources=[('initial',root),('refinement',root/'refinement')]
    summaries=[];scores=[]
    for stage,folder in sources:
        assert R.read(folder/'verification.json')['passed']
        summaries.extend(dict(stage=stage,**row) for row in R.rows(folder/'summary.csv'))
        scores.extend(dict(stage=stage,**row) for row in R.rows(folder/'results.csv'))
    R.table(root/'combined_summary.csv',summaries)
    eligible=[r for r in summaries if r['start'] in R.STARTS]
    best=max(eligible,key=lambda r:float(r['mean_percent']))
    single=max([r for r in scores if r['start'] in R.STARTS],key=lambda r:float(r['accuracy_percent']))
    old=R.rows(R.BASE/'results/orl_protocol_audit/evaluation/fixed_results.csv')
    paired=[]
    for seed in R.SEEDS:
        previous=next(r for r in old if int(r['seed'])==seed and r['method']=='CSA-2+LDA')
        a=next(r for r in scores if r['stage']==best['stage'] and r['input']==best['input']
               and r['start']==best['start'] and r['policy']==best['policy'] and int(r['seed'])==seed)
        controls=[next(r for r in scores if r['stage']==best['stage'] and r['input']==best['input']
               and r['start']==method and r['policy']==best['policy'] and int(r['seed'])==seed)
                  for method in R.CONTROLS]
        old_accuracy=float(previous['accuracy_percent']);new_accuracy=float(a['accuracy_percent'])
        paired.append(dict(seed=seed,old_shared_CSA=old_accuracy,new_CSA=new_accuracy,
            change_pp=new_accuracy-old_accuracy,new_correct=int(a['correct']),probe_count=320,
            matched_PCA30_LDA18=float(controls[0]['accuracy_percent']),
            matched_PCA22_LDA16=float(controls[1]['accuracy_percent'])))
    R.table(root/'final_paired_comparison.csv',paired)
    matched=[r for r in summaries if r['stage']==best['stage'] and r['input']==best['input'] and r['policy']==best['policy']]
    per_input=[]
    for stage,input_id in dict.fromkeys((r['stage'],r['input']) for r in eligible):
        per_input.append(max([r for r in eligible if r['stage']==stage and r['input']==input_id],
                             key=lambda r:float(r['mean_percent'])))
    R.table(root/'per_input_best_exploratory.csv',per_input)
    change=np.array([r['change_pp'] for r in paired])
    initialization=[]
    folder=root/'refinement' if best['stage']=='refinement' else root
    policy_index=next(i for i,r in enumerate(R.POLICIES) if r['id']==best['policy'])
    for seed in R.SEEDS:
        with np.load(folder/'predictions'/f"{best['input']}_s{seed}_rectangular_identity.npz") as x, \
             np.load(folder/'predictions'/f"{best['input']}_s{seed}_hosvd.npz") as y:
            labels=R.LABELS[x['probe']]
            a,b=x['predicted'][policy_index],y['predicted'][policy_index]
            initialization.append(dict(seed=seed,predicted_labels_differ=int(np.sum(a!=b)),
                correctness_flags_differ=int(np.sum((a==labels)!=(b==labels))),
                identity_correct=int(np.sum(a==labels)),hosvd_correct=int(np.sum(b==labels))))
    result=dict(best_global_mean=best,matched_settings=matched,highest_single_run=single,
        old_shared_mean=float(np.mean([r['old_shared_CSA'] for r in paired])),
        old_shared_std=float(np.std([r['old_shared_CSA'] for r in paired],ddof=1)),
        mean_change_pp=float(change.mean()),improved_splits=int(np.sum(change>0)),
        tied_splits=int(np.sum(change==0)),lower_splits=int(np.sum(change<0)),
        model_count=960,CSA_models=600,PCA_models=360,score_count=len(scores),
        prediction_count=320*len(scores),candidate_CSA_means=len(eligible),input_bundles=18,
        all_ten_splits_retained=True,all_batches_verified=True,
        selection='Global configuration selected using already-observed probes; no per-split oracle mean.',
        paper_aggregation_unknown=True,author_settings_confirmed=False,
        fixed=dict(input_size=[56,46],CSA_ranks=[11,2],LDA=16,Tmax=3,G=2,P=8,
                   regularization='none',additional_training_images=False),
        best_settings=dict(eyes='YuNet frozen annotation',target_left=[15.5,24.],target_right=[30.5,24.],
            interpolation='OpenCV INTER_LINEAR',boundary='BORDER_REPLICATE',histogram='PIL equalize',
            CSA_center=False,CSA_start=best['start'],LDA_scale='unit columns',NN='gallery-centered cosine'),
        identity_hosvd_comparison=initialization,
        limits=R.read(root/'protocol.json')['limits'])
    R.dump(root/'final_analysis.json',result)
    frozen={}
    for stage,folder in sources:
        for name in ['protocol.json','splits.json','models.csv','summary.csv','results.csv','verification.json']:
            frozen[(Path(stage)/name).as_posix()]=R.file_hash(folder/name)
    R.dump(root/'final_analysis_provenance.json',dict(analyzer_sha256=R.file_hash(Path(__file__)),
             source_sha256=frozen,outputs_sha256={name:R.file_hash(root/name) for name in
             ['combined_summary.csv','final_paired_comparison.csv','per_input_best_exploratory.csv','final_analysis.json']}))
    print(json.dumps({k:result[k] for k in ['best_global_mean','mean_change_pp','improved_splits',
          'tied_splits','lower_splits','model_count','score_count','prediction_count']},indent=2))


if __name__=='__main__':main()
