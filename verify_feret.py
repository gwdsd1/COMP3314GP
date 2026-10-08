"""Read back FERET results: official pairs, training/test isolation, training-only selection, all predictions, and hashes."""

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from gabor import file_hash
from prepare_feret import load_pairs,OUTPUT as PREPARED,ROOT
from run_feret import read_csv,BASE,CASES


def verify(output):
    output=Path(output);protocol=json.loads((output/'protocol.json').read_text())
    metadata=protocol['preprocessing'];pairs=load_pairs();labels=np.load(PREPARED/'labels.npy');records=read_csv(PREPARED/'records.csv')
    assert np.array_equal(labels[::2],[r['subject_id'] for r in pairs]) and np.array_equal(labels[1::2],labels[::2])
    assert file_hash(ROOT/'official_fa_fb_pairs.csv')==metadata['pair_table_sha256']
    assert file_hash(ROOT/'fa_fb_candidates.csv')==metadata['candidate_table_sha256']
    assert file_hash(ROOT/'file_manifest.json')==metadata['source_manifest_sha256']
    for name,digest in metadata['prepared_sha256'].items():assert file_hash(PREPARED/name)==digest
    for name,digest in protocol['code_sha256'].items():assert file_hash(output/'source_snapshot'/name)==digest
    feature_hashes=json.loads((output/'feature_hashes.json').read_text())
    for name,digest in feature_hashes.items():
        assert file_hash(output/name)==digest
        array=np.load(output/name,mmap_mode='r');assert array.shape==(2390,41,27) and np.isfinite(array).all()
        if 'zscore' in name:assert np.allclose(array.mean(axis=(1,2)),0,atol=1e-8) and np.allclose(array.std(axis=(1,2)),1)
    checks=['official pairs, prepared data, source snapshots and final feature hashes/shapes']

    partitions=defaultdict(set);split_rows=read_csv(output/'splits.csv')
    for r in split_rows:
        i=int(r['index']);q=int(r['q']);partitions[q,r['partition']].add(i)
        assert r['subject_id']==labels[i] and r['file']==records[i]['file'] and r['role']==records[i]['role']
    fold_counts={int(q):int(n) for q,n in protocol.get('fold_counts',{50:5,100:5,200:5}).items()}
    for q in CASES:
        train=partitions[q,'subspace_lda_train'];gallery=partitions[q,'unseen_gallery'];probe=partitions[q,'unseen_probe']
        assert train==set(range(2*q)) and gallery==set(range(2*q,2390,2)) and probe==set(range(2*q+1,2390,2))
        assert not set(labels[list(train)])&set(labels[list(gallery)])
        assert set(labels[list(gallery)])==set(labels[list(probe)]) and len(gallery)==len(probe)==1195-q
    checks.append('first-q Fa+Fb fitting, remaining Fa gallery/Fb probe, strict identity isolation')

    validation_folds=read_csv(output/'validation_folds.csv');fold_sets=defaultdict(set)
    for r in validation_folds:
        q,fold,i=int(r['q']),int(r['fold']),int(r['index'])
        assert i<2*q and r['subject_id']==labels[i] and r['role']==records[i]['role']
        fold_sets[q,fold,r['partition']].add(i)
    for q in CASES:
        heldout=[]
        for fold,people in enumerate(np.array_split(np.random.default_rng(3314+q).permutation(q),fold_counts[q])):
            fitting=np.setdiff1d(np.arange(q),people)
            assert fold_sets[q,fold,'inner_train']==set(np.r_[2*fitting,2*fitting+1])
            assert fold_sets[q,fold,'validation_gallery']==set(range(0,2*q,2))
            assert fold_sets[q,fold,'validation_probe']==set(2*people+1)
            assert not set(labels[list(fold_sets[q,fold,'inner_train'])])&set(labels[2*people+1])
            heldout.extend(people.tolist())
        assert sorted(heldout)==list(range(q))
    checks.append('declared identity folds contain only training subjects; validation targets excluded from fitting')

    val_predictions=read_csv(output/'validation_predictions.csv');vgroups=defaultdict(list)
    for r in val_predictions:
        q,fold,i,j=int(r['q']),int(r['fold']),int(r['probe_index']),int(r['nearest_gallery_index'])
        assert i in fold_sets[q,fold,'validation_probe'] and j in fold_sets[q,fold,'validation_gallery']
        assert r['true']==labels[i] and r['predicted']==labels[j] and int(r['correct'])==int(r['true']==r['predicted'])
        assert np.isfinite(float(r['margin'])) and abs(float(r['margin']))<=1+1e-8
        vgroups[q,fold,r['method'],r['candidate_id']].append(r)
    validation=read_csv(output/'validation_results.csv')
    for r in validation:
        q,fold=int(r['q']),int(r['fold']);ps=vgroups[q,fold,r['method'],r['candidate_id']]
        assert len(ps)==len({p['probe_index'] for p in ps})==int(r['probe_count'])==q//fold_counts[q]
        assert sum(int(p['correct']) for p in ps)==int(r['correct'])
        assert np.isclose(sum(float(p['margin']) for p in ps),float(r['margin_sum']))
        inner_people=q-q//fold_counts[q]
        dims=ast.literal_eval(r['inner_pre_dims']);assert np.prod(dims)<=inner_people
        assert int(r['lda_dim'])<=min(np.prod(dims),inner_people-1)
    candidates=read_csv(output/'candidate_scores.csv');selected=json.loads((output/'selected_configs.json').read_text())
    for r in candidates:
        q=int(r['q']);vr=[v for v in validation if int(v['q'])==q and v['method']==r['method'] and v['candidate_id']==r['candidate_id']]
        assert len(vr)==fold_counts[q] and sum(int(v['probe_count']) for v in vr)==q
        assert sum(int(v['correct']) for v in vr)==int(r['validation_correct'])
        assert np.isclose(sum(float(v['margin_sum']) for v in vr)/q,float(r['validation_mean_margin']))
    for choice in selected:
        eligible=[r for r in candidates if int(r['q'])==choice['q'] and r['method']==choice['method']]
        field='worst_fold_percent' if 'fold_counts' in protocol else 'validation_mean_margin'
        winner=min(eligible,key=lambda r:(-int(r['validation_correct']),-float(r[field]),r['candidate_id']))
        assert winner['candidate_id']==choice['candidate_id']
    checks.append('all validation predictions, pooled scores and declared winner rule independently recomputed')

    predictions=read_csv(output/'predictions.csv');groups=defaultdict(list)
    for r in predictions:
        q,i,j=int(r['q']),int(r['probe_index']),int(r['nearest_gallery_index'])
        assert i in partitions[q,'unseen_probe'] and j in partitions[q,'unseen_gallery']
        assert r['true']==labels[i] and r['predicted']==labels[j]
        assert r['probe']==records[i]['file'] and r['nearest_gallery']==records[j]['file']
        assert int(r['correct'])==int(r['true']==r['predicted']) and np.isfinite(float(r['distance']))
        groups[q,r['profile'],r['method']].append(r)
    fixed=read_csv(output/'fixed_results.csv')
    for r in fixed:
        q=int(r['q']);ps=groups[q,r['profile'],r['method']]
        assert len(ps)==len({p['probe_index'] for p in ps})==int(r['probe_count'])==1195-q
        assert sum(int(p['correct']) for p in ps)==int(r['correct'])
        assert np.isclose(100*int(r['correct'])/(1195-q),float(r['accuracy_percent']))
        case=next(c for c in CASES[q] if c[0]==r['method'])
        assert ast.literal_eval(r['pre_dims'])==case[1] and int(r['lda_dim'])==case[2] and float(r['paper_percent'])==case[3]
        assert np.prod(case[1])<=q and int(r['sw_rank'])>=case[2]
        if r['profile']=='validation_selected':
            choice=next(s for s in selected if s['q']==q and s['method']==r['method'])
            for field in ['candidate_id','eye_target','normalization','lda_scaling','metric','init']:assert r[field]==choice[field]
            assert (r['center']=='True')==choice['center']
    for item in json.loads((output/'csa_history.json').read_text()):
        assert len(item['history'])==3
        errors=[v for h in item['history'] for v in h['mode_sse']]
        assert np.all(np.diff(errors)<=1e-5*max(errors))
    counts={'fixed_results':len(fixed),'predictions':len(predictions),'splits':len(split_rows),
            'validation_folds':len(validation_folds),'validation_results':len(validation),
            'validation_predictions':len(val_predictions),'candidates':len(candidates),
            'curves':len(read_csv(output/'lda_dimension_curves.csv'))}
    expected_folds=sum((3*q-q//fold_counts[q])*fold_counts[q] for q in CASES)
    assert counts==dict(fixed_results=24,predictions=25880,splits=7170,validation_folds=expected_folds,
                        validation_results=sum(144*n for n in fold_counts.values()),validation_predictions=50400,candidates=432,curves=1752),counts
    checks.append('all final predictions and counts, exact Table I dimensions, three-iteration monotonic CSA')
    result={'status':'passed','counts':counts,'checks':checks,
            'limits':'checks validate bookkeeping and selection isolation; previous experiments were seen, so this is not a fresh blind test'}
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'results/feret_validation_selected_v3')
    verify(parser.parse_args().output)
