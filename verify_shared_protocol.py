"""独立回读预测/嵌入、划分、统计、有效区域和直接 Gabor 卷积。"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d
from threadpoolctl import threadpool_limits

from run_shared_protocol import ROOT, BASE, LABELS, NAMES, feature_spec, dump
from run_feret import read_csv
from gabor import file_hash
from diagnose_orl_gabor import filtered

def direct_checks():
    image=np.random.default_rng(3314).normal(size=(8,6)).astype(np.float32);results=[]
    for name in ['adaptive_reflect','adaptive_zero','fixed39_reflect','fixed39_zero']:
        s=feature_spec(name);actual=filtered(image[None],s)[0];expected=np.empty_like(actual)
        sizes=[]
        for scale in range(5):
            wavelength=4*np.sqrt(2)**scale;k=2*np.pi/wavelength;sigma=2*np.pi
            radius=19 if name.startswith('fixed39') else int(np.ceil(np.round(3*wavelength,12)))
            sizes.append(2*radius+1);y,x=np.mgrid[-radius:radius+1,-radius:radius+1]
            padded=np.pad(image.astype(np.float64),((radius,radius),(radius,radius)),mode=s['boundary'])
            for orientation in range(8):
                theta=orientation*np.pi/8
                kernel=(k*k/sigma**2)*np.exp(-k*k*(x*x+y*y)/(2*sigma*sigma))*(np.exp(1j*k*(x*np.cos(theta)+y*np.sin(theta)))-np.exp(-sigma*sigma/2))
                kernel-=kernel.mean()
                expected[:,:,scale*8+orientation]=np.abs(convolve2d(padded,kernel,mode='valid'))[::2,::2]
        error=float(np.max(np.abs(actual-expected)));assert error<2e-6,(name,error)
        results.append({'variant':name,'all_40_channels_max_abs_error':error,'kernel_sizes':sizes})
    return results

def verify(stage):
    out=ROOT/f'step{stage}';protocol=json.loads((out/'protocol.json').read_text())
    for name,value in protocol['code_sha256'].items():assert file_hash(out/'source_snapshot'/name)==value
    scores=read_csv(out/'fixed_results.csv');predictions=read_csv(out/'predictions.csv');splits=read_csv(out/'splits.csv')
    expected_models={1:18,2:72,3:48,4:80}[stage]
    assert len(scores)==expected_models and len(predictions)==expected_models*320
    expected_seeds=list(range(20,30)) if stage==4 else [20,21,22]
    assert len(splits)==len(expected_seeds)*400
    part={}
    for seed in expected_seeds:
        group=[r for r in splits if int(r['seed'])==seed]
        gallery=[int(r['index']) for r in group if r['partition']=='gallery'];probe=[int(r['index']) for r in group if r['partition']=='probe']
        assert len(gallery)==80 and len(probe)==320 and set(gallery).isdisjoint(probe) and set(gallery+probe)==set(range(400))
        for indices,count in [(gallery,2),(probe,8)]:
            assert all(np.sum(LABELS[indices]==label)==count for label in range(1,41))
        part[seed]=(np.array(gallery),np.array(probe))
        for r in group:assert NAMES[int(r['index'])]==r['file'] and LABELS[int(r['index'])]==int(r['label'])
    groups=defaultdict(list)
    for r in predictions:groups[(r['variant'],r['seed'],r['method'])].append(r)
    keys=set();inputs=defaultdict(set)
    for row in scores:
        key=(row['variant'],row['seed'],row['method']);assert key not in keys;keys.add(key)
        records=groups[key];assert len(records)==320
        embeddings=np.load(out/row['embedding']);gallery=embeddings['gallery'];queries=embeddings['probe']
        train,probe=part[int(row['seed'])]
        np.testing.assert_array_equal(train,embeddings['train_indices']);np.testing.assert_array_equal(probe,embeddings['probe_indices'])
        assert np.isfinite(gallery).all() and np.isfinite(queries).all()
        # 独立矩阵点积形式的 cosine NN，不调用运行入口的 cdist/nn。
        norms=np.linalg.norm(gallery,axis=1);qnorms=np.linalg.norm(queries,axis=1)
        assert min(norms)>0 and min(qnorms)>0
        distances=1-(queries/qnorms[:,None])@(gallery/norms[:,None]).T
        positions=np.argmin(distances,axis=1);correct=0
        for j,(i,position,r) in enumerate(zip(probe,positions,records)):
            assert int(r['probe_index'])==i and int(r['neighbor_index'])==train[position]
            assert int(r['predicted'])==LABELS[train[position]] and int(r['true'])==LABELS[i]
            assert r['probe']==NAMES[i] and r['nearest_gallery']==NAMES[train[position]]
            assert abs(float(r['distance'])-distances[j,position])<1e-10
            flag=int(LABELS[i]==LABELS[train[position]]);assert int(r['correct'])==flag;correct+=flag
        assert int(row['correct'])==correct and abs(float(row['accuracy_percent'])-100*correct/320)<1e-10
        assert row['metric']=='cosine' and row['lda_scaling']=='unit_columns'
        kind='gray' if row['method'] in ['PCA+LDA (Gray)','CSA-2+LDA'] else 'gabor'
        inputs[(row['variant'],kind)].add(row['input_sha256'])
    assert all(len(v)==1 for v in inputs.values())
    assert set(groups)==keys
    manifests=json.loads((out/'feature_manifests.json').read_text())
    for m in manifests:
        for kind,shape in [('gray',(400,56,46)),('gabor',(400,28,23,40))]:
            path=BASE/m[f'{kind}_path'];assert file_hash(path)==m[f'{kind}_sha256']
            array=np.load(path,mmap_mode='r');assert array.shape==shape and np.isfinite(array).all()
            assert inputs[(m['variant'],kind)]=={m[f'{kind}_sha256']} if (m['variant'],kind) in inputs else True
        if stage in [2,3]:
            name='legacy_reflect' if stage==2 else m['variant']
            gray=np.load(BASE/m['gray_path'],mmap_mode='r')
            saved=np.load(BASE/m['gabor_path'],mmap_mode='r')
            regenerated=filtered(gray[[0,59]],feature_spec(name))
            np.testing.assert_allclose(regenerated,saved[[0,59]],rtol=2e-6,atol=2e-6)
    summary=json.loads((out/'summary.json').read_text())
    for r in summary:
        values=[float(s['accuracy_percent']) for s in scores if s['method']==r['method'] and s['variant']==r['variant']]
        assert len(values)==r['n_splits']==len(expected_seeds)
        assert abs(np.mean(values)-r['mean_percent'])<1e-10 and abs(np.std(values,ddof=1)-r['std_pp'])<1e-10
    histories=json.loads((out/'csa_history.json').read_text())
    for h in histories:
        assert len(h['history'])==3
        errors=[h['initial_sse']]+[r['sse'] for r in h['history']]
        assert all(b<=a+1e-8*max(1,a) for a,b in zip(errors,errors[1:]))
    extra={}
    if stage==1:
        for source in json.loads((out/'external_sources.json').read_text()):
            folder=BASE/source['directory']
            for name,value in source['sha256'].items():assert file_hash(folder/name)==value
            assert json.loads((folder/'verification.json').read_text())['status']=='passed'
            original=read_csv(folder/'fixed_results.csv');original_predictions=read_csv(folder/'predictions.csv')
            selected=[r for r in original if r['profile']==source['profile'] and (r.get('dataset','FERET').startswith('MNIST') or 'q' in r)]
            for row in selected:
                matching=[r for r in original_predictions if r['profile']==row['profile'] and r['method']==row['method'] and
                          (r['q']==row['q'] if 'q' in row else r['dataset']==row['dataset'] and r['seed']==row['seed'])]
                assert len(matching)==int(row['probe_count'])
                assert sum(int(r['true']==r['predicted']) for r in matching)==int(row['correct'])
        assert len(read_csv(out/'external_shared_results.csv'))==18
        extra['external_reused_scores']=18
    if stage==2:
        quality=read_csv(out/'image_quality.csv');assert len(quality)==800
        assert max(float(r['eye_error']) for r in quality)<1e-10
        for variant in ['valid_zscore','valid_zscore_lanczos']:
            array=np.load(out/f'{variant}.npy');mask=np.load(out/f'{variant}_mask.npy')
            for image,valid in zip(array,mask):
                assert np.all(image[~valid]==0)
                assert abs(np.mean(image[valid].astype(np.float64)))<1e-6
                assert abs(np.std(image[valid].astype(np.float64))-1)<1e-6
        baseline=np.load(out/'whole_hist.npy');np.testing.assert_array_equal(baseline,np.load(BASE/'data/processed/orl_branches_v2/eyes_only_seed20.npy'))
        old=read_csv(ROOT/'step1/fixed_results.csv')
        for r in scores:
            if r['variant']=='whole_hist':
                matching=next(s for s in old if s['seed']==r['seed'] and s['method']==r['method'])
                assert r['correct']==matching['correct']
        extra['preprocessing_selection']=json.loads((out/'preprocessing_selection.json').read_text())
    if stage==3:extra['independent_convolution_checks']=direct_checks()
    if stage==4:
        assert file_hash(ROOT/'step3/protocol.json')==protocol['prior_protocol_sha256']
        old=read_csv(ROOT/'step3/fixed_results.csv')
        for r in scores:
            if int(r['seed'])<=22 and r['method'] not in ['PCA+LDA (Gray)','CSA-2+LDA','PCA-64 (Gabor)','CSA-64 (Gabor)']:
                matching=next(s for s in old if s['variant']==protocol['frozen_gabor'] and s['seed']==r['seed'] and s['method']==r['method'])
                assert r['correct']==matching['correct']
        for r in read_csv(out/'paired_differences.csv'):
            selected=[s for s in scores if s['seed']==r['seed']]
            p=next(s for s in selected if s['method']==r['pca']);c=next(s for s in selected if s['method']==r['csa'])
            assert abs(float(r['csa_minus_pca_pp'])-(float(c['accuracy_percent'])-float(p['accuracy_percent'])))<1e-10
        table=read_csv(ROOT/'shared_comparison.csv');assert len(table)==18
        for r in table:
            if r['dataset']=='ORL':
                s=next(x for x in summary if x['method']==r['method'])
                assert abs(float(r['mean_percent'])-s['mean_percent'])<1e-10
            else:
                source=read_csv(ROOT/'step1/external_shared_results.csv')
                values=[float(x['accuracy_percent']) for x in source if (x['dataset'],x['method'])==(r['dataset'],r['method'])]
                assert len(values)==int(r['n_splits'])
                assert abs(np.mean(values)-float(r['mean_percent']))<1e-10
                sd=np.std(values,ddof=1) if len(values)>1 else 0.
                assert abs(sd-float(r['std_pp']))<1e-10
        extra['frozen_protocol_sha256']=file_hash(out/'protocol.json')
    result={'passed':True,'stage':stage,'models':len(scores),'predictions':len(predictions),'splits':len(splits),
            'verifier_sha256':file_hash(BASE/'verify_shared_protocol.py'),
            'checks':['snapshot/feature hashes','shared input hashes across methods','gallery/probe disjoint and class counts',
                      'independent matrix cosine NN on saved embeddings','prediction-derived accuracy and sample SD',
                      'CSA three iterations and monotone SSE','prior-stage anchor checks'],**extra}
    dump(out/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,required=True,choices=[1,2,3,4])
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    ROOT=args.output.resolve()
    with threadpool_limits(limits=4):verify(args.stage)
