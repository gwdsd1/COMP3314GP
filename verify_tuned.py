"""Read back training-only search and final outputs; verify splits, selection, every prediction, summaries, and provenance."""
import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d
import gabor
from gabor import file_hash
from run_feret import read_csv

BASE=Path(__file__).resolve().parent


def groups(rows,keys):
    result=defaultdict(list)
    for row in rows:result[tuple(row[k] for k in keys)].append(row)
    return result


def verify(output):
    output=Path(output);protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['original_test_for_selection'] is False
    for name,digest in protocol['code_sha256'].items():
        assert file_hash(output/'source_snapshot'/name)==digest
    for name,digest in protocol['inputs_sha256'].items():
        assert file_hash(BASE/'data/processed/available'/name)==digest
    assert file_hash(BASE/'data/annotations/orl_eye_coordinates.csv')==protocol['eye_annotations_sha256']
    manifests=json.loads((output/'feature_manifests.json').read_text())
    for name,meta in manifests.items():
        path=BASE/'data/processed/tuning'/Path(name).with_suffix('.npy')
        assert file_hash(path)==meta['sha256']
        array=np.load(path,mmap_mode='r')
        assert array.shape==((400,28,23,40) if name.startswith('ORL') else (5000,28,28,40))
        for start in range(0,len(array),128):assert np.isfinite(array[start:start+128]).all()
        assert meta['spec']['gabor_sha256']==protocol['code_sha256']['gabor.py']
        assert meta['spec']['tuner_sha256']==protocol['code_sha256']['tune_available.py']
    checks=['source snapshots, inputs, eye annotations and eight feature caches match recorded hashes']

    labels={d:np.load(BASE/'data/processed/available'/('orl_labels.npy' if d=='ORL' else 'mnist_labels.npy'))
            for d in ['ORL','MNIST G30/P470','MNIST G40/P460']}
    splits=read_csv(output/'splits.csv');outer=groups(splits,['dataset','seed','partition'])
    for dataset,y in labels.items():
        count=2 if dataset=='ORL' else 30 if 'G30' in dataset else 40
        for seed in protocol['seeds']:
            rng=np.random.default_rng(seed);train=[];probe=[]
            for label in np.unique(y):
                order=rng.permutation(np.flatnonzero(y==label));train.extend(order[:count]);probe.extend(order[count:])
            for part,indices in [('train_gallery',train),('probe',probe)]:
                rs=outer[dataset,str(seed),part]
                assert [int(r['index']) for r in rs]==indices
                assert all(r['label']==str(y[int(r['index'])]) for r in rs)
    folds=read_csv(output/'validation_folds.csv');fg=groups(folds,['dataset','seed','fold','partition'])
    index_cache={}
    def indices(mapping,key):
        cache_key=(id(mapping),key)
        if cache_key not in index_cache:index_cache[cache_key]={int(r['index']) for r in mapping[key]}
        return index_cache[cache_key]
    for dataset in labels:
        for seed in protocol['seeds']:
            train=indices(outer,(dataset,str(seed),'train_gallery'));seen=[]
            for fold in range(8 if dataset=='ORL' else 2):
                key=(dataset,str(seed),str(fold));fit=indices(fg,(*key,'inner_fit'))
                gallery=indices(fg,(*key,'validation_gallery'));query=indices(fg,(*key,'validation_probe'))
                assert fit|gallery|query<=train and not query&fit and not query&gallery
                y=labels[dataset]
                for part in ['inner_fit','validation_gallery','validation_probe']:
                    assert all(r['label']==str(y[int(r['index'])]) for r in fg[(*key,part)])
                if dataset=='ORL':
                    assert (len(fit),len(gallery),len(query))==(60,70,10)
                    assert not set(y[list(fit)])&set(y[list(query)])
                    assert set(y[list(query)])<=set(y[list(gallery)])
                else:
                    count=24 if 'G30' in dataset else 32
                    assert gallery==fit and len(fit)==10*count
                    assert all(np.sum(y[list(fit)]==c)==count for c in np.unique(y))
                seen.extend(query)
            if dataset=='ORL':assert len(seen)==len(set(seen))==80 and set(seen)==train
    checks.append('original seeds and 2/8, G30/P470, G40/P460 splits; every validation index belongs to original training/gallery')

    vp=read_csv(output/'validation_predictions.csv')
    keys=['dataset','seed','method','stage','candidate_id','fold'];pg=groups(vp,keys)
    for r in vp:
        key=(r['dataset'],r['seed'],r['fold']);i,j=int(r['probe_index']),int(r['nearest_gallery_index']);y=labels[r['dataset']]
        assert i in indices(fg,(*key,'validation_probe')) and j in indices(fg,(*key,'validation_gallery'))
        assert r['true']==str(y[i]) and r['predicted']==str(y[j])
        assert int(r['correct'])==int(y[i]==y[j])
    vr=read_csv(output/'validation_results.csv');vg=groups(vr,keys[:-1])
    for r in vr:
        ps=pg[tuple(r[k] for k in keys)];n=int(r['probe_count']);correct=int(r['correct'])
        assert len(ps)==len({p['probe_index'] for p in ps})==n
        assert sum(int(p['correct']) for p in ps)==correct
        assert np.isclose(float(r['accuracy_percent']),100*correct/n)
        dims=ast.literal_eval(r['inner_pre_dims'])
        if r['dataset']=='ORL' and r['method'].startswith('PCA'):
            expected=30 if 'Gray' in r['method'] or '+LDA' in r['method'] else 59
            assert dims==expected
    candidates=read_csv(output/'candidate_scores.csv');cg=groups(candidates,keys[:4])
    for r in candidates:
        rs=vg[tuple(r[k] for k in keys[:-1])]
        correct=sum(int(v['correct']) for v in rs);n=sum(int(v['probe_count']) for v in rs)
        assert correct==int(r['validation_correct']) and n==int(r['validation_probes'])
        assert np.isclose(float(r['validation_percent']),100*correct/n)
        assert np.isclose(float(r['worst_fold_percent']),min(float(v['accuracy_percent']) for v in rs))
    selected=json.loads((output/'selected_configs.json').read_text())
    for s in selected:
        key=(s['dataset'],str(s['seed']),s['method'])
        def winner(stage):return min(cg[(*key,stage)],key=lambda r:(-float(r['validation_percent']),-float(r['worst_fold_percent']),r['candidate_id']))
        a,b=winner('preprocess'),winner('model')
        for name in ['geometry','kernel','response','normalization']:
            assert a[name]==b[name] and all(r[name]==a[name] for r in cg[(*key,'model')])
        for name,value in s['config'].items():assert b[name]==str(value)
        assert np.isclose(s['validation_percent'],float(b['validation_percent']))
    checks.append('all validation predictions and pooled/worst-fold scores; both-stage winners replay without final-probe scores')

    final=read_csv(output/'fixed_results.csv');predictions=read_csv(output/'predictions.csv');fp=groups(predictions,keys[:3])
    for r in predictions:
        y=labels[r['dataset']];i,j=int(r['probe_index']),int(r['nearest_gallery_index'])
        assert i in indices(outer,(r['dataset'],r['seed'],'probe')) and j in indices(outer,(r['dataset'],r['seed'],'train_gallery'))
        assert r['true']==str(y[i]) and r['predicted']==str(y[j]) and int(r['correct'])==int(y[i]==y[j])
        assert np.isfinite(float(r['distance'])) and float(r['distance'])>=-1e-10
    orl_dims={'PCA+LDA (Gray)':30,'CSA-2+LDA':(11,2),'PCA (Gabor)':76,'CSA-3':(15,1,26),'PCA+LDA (Gabor)':33,'CSA-3+LDA':(4,2,3)}
    for r in final:
        key=tuple(r[k] for k in keys[:3]);ps=fp[key]
        assert len(ps)==len({p['probe_index'] for p in ps})==int(r['probe_count'])
        correct=sum(int(p['correct']) for p in ps);assert correct==int(r['correct'])
        assert np.isclose(float(r['accuracy_percent']),100*correct/len(ps))
        expected=orl_dims[r['method']] if r['dataset']=='ORL' else (60 if 'G30' in r['dataset'] else 50) if r['method'].startswith('PCA') else (5,9,18) if 'G30' in r['dataset'] else (15,5,13)
        assert ast.literal_eval(r['pre_dims'])==expected
        s=next(s for s in selected if (s['dataset'],str(s['seed']),s['method'])==key)
        assert all(r[k]==str(v) for k,v in s['config'].items())
    summary=json.loads((output/'summary.json').read_text())['summary'];old=json.loads((BASE/'results/available_reproduction_v2/summary.json').read_text())['summary']
    for s in summary:
        rows=[r for r in final if r['dataset']==s['dataset'] and r['method']==s['method']]
        values=[float(r['accuracy_percent']) for r in rows];assert len(values)==3
        assert np.isclose(s['mean_percent'],np.mean(values)) and np.isclose(s['std_pp'],np.std(values,ddof=1))
        previous=next(r for r in old if r['dataset']==s['dataset'] and r['method']==s['method'] and r['profile']=='group_primary')
        assert s['previous_percent']==previous['mean_percent'] and np.isclose(s['improvement_pp'],np.mean(values)-s['previous_percent'])
        assert np.isclose(s['difference_pp'],s['mean_percent']-s['paper_percent'])
    histories=json.loads((output/'csa_history.json').read_text())
    for model in histories:
        h=model['history'];assert 1<=len(h)<=3
        errors=[v for step in h for v in step['mode_sse']]
        assert np.isfinite(errors).all() and min(errors)>=-1e-5
        assert np.all(np.diff(errors)<=max(errors)*1e-10+1e-7)
    checks.append('final predictions, selected configurations, paper dimensions, three-seed mean/sample-SD and monotone CSA histories')

    # Newly searched wavelength base 2: independent direct spatial convolution.
    saved=gabor.CONFIG;rng=np.random.default_rng(732);image=rng.normal(size=(1,9,7)).astype(np.float32);maximum=0.
    try:
        gabor.CONFIG={**saved,'wavelengths':[float(2*np.sqrt(2)**s) for s in range(5)]}
        actual=gabor.features(image)
        for scale in [0,2,4]:
            for orientation in [0,3,7]:
                kernel=gabor.kernel(gabor.CONFIG['wavelengths'][scale],gabor.CONFIG['orientations'][orientation]);radius=kernel.shape[0]//2
                padded=np.pad(image[0],((radius,radius),(radius,radius)),mode='reflect')
                expected=np.abs(convolve2d(padded,kernel,mode='valid'))
                maximum=max(maximum,float(np.max(np.abs(expected-actual[0,:,:,8*scale+orientation]))))
                np.testing.assert_allclose(actual[0,:,:,8*scale+orientation],expected,atol=2e-6,rtol=3e-5)
        np.testing.assert_allclose(gabor.features(image,step=2),actual[:,::2,::2],atol=0)
    finally:gabor.CONFIG=saved
    assert gabor.CONFIG is saved
    checks.append('new base-2 Gabor FFT matches independent spatial convolution for nine kernels; sampling and configuration restoration verified')
    counts={'final_results':len(final),'final_predictions':len(predictions),'outer_split_rows':len(splits),
            'validation_fold_rows':len(folds),'validation_results':len(vr),'validation_predictions':len(vp),
            'candidates':len(candidates),'selected_units':len(selected),'summary_groups':len(summary),'csa_histories':len(histories)}
    assert counts=={'final_results':30,'final_predictions':61560,'outer_split_rows':31200,'validation_fold_rows':10920,
                    'validation_results':7152,'validation_predictions':103200,'candidates':1092,'selected_units':30,'summary_groups':10,'csa_histories':15}
    result={'passed':True,'counts':counts,'checks':checks,'new_gabor_max_absolute_error':maximum,
            'limits':['audit checks recorded outputs, not an independent full model reimplementation','previous final scores were already observed; development evaluation, not a fresh blind test']}
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE/'results/available_tuned')
    verify(parser.parse_args().output)
