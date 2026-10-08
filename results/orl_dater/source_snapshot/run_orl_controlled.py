"""Fixed ORL sigma x context factorial and saved-model diagnostics.

No gallery/probe selection: twenty declared splits, four inputs, four methods.
Historical error analysis reuses saved predictions; eye shifts never refit.
"""
import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import optimize_orl_unknowns as E
import optimize_orl_ta_crop as T
from gabor import file_hash
from run_orl_fixed_inputs import NAMES

BASE = Path(__file__).resolve().parent
ROOT = BASE/'results/orl_controlled'
SEEDS = list(range(4200, 4220))
SHIFTS = [('x_minus1', -1., 0.), ('x_plus1', 1., 0.),
          ('y_minus1', 0., -1.), ('y_plus1', 0., 1.)]
SOURCES = ['run_orl_controlled.py', 'optimize_orl_ta_crop.py']+E.SOURCES
read, dump, table, csv_rows = E.read, E.dump, E.table, E.csv_rows


def specs():
    common = dict(wavelength=4., support=19, finite_dc=True, distance=20.,
                  eye_y=24., histogram='PIL', downsample_first=False)
    return [dict(id=f'{sigma}_{place}', sigma=float(value), context=context, **common)
            for sigma, value in [('2pi', 2*np.pi), ('pi', np.pi)]
            for place, context in [('crop', False), ('context', True)]]


def prepare():
    assert not ROOT.exists(), 'Use a new output directory; no historical overwrite'
    ROOT.mkdir(parents=True)
    for folder in ['models', 'predictions', 'robust_predictions', 'source_snapshot']:
        (ROOT/folder).mkdir()
    hashes = {n: file_hash(BASE/n) for n in SOURCES}
    for n in SOURCES:
        shutil.copy2(BASE/n, ROOT/'source_snapshot'/n)
    old = {e['id']: e for e in read(BASE/'results/orl_unknown_parameters/feature_manifest.json')}
    crop = {e['id']: e for e in read(BASE/'results/orl_ta_crop_optimized/feature_manifest.json')}
    mapping = {'2pi_crop': old['legacy_crop'], '2pi_context': old['context_anchor'],
               'pi_crop': crop['crop_pi_zero'], 'pi_context': old['context_sigma_pi']}
    raw, eyes = E.raw_inputs()
    checks, manifest = [], []
    for spec in specs():
        item = mapping[spec['id']]
        assert file_hash(BASE/item['path']) == item['sha256']
        data = np.load(BASE/item['path'], mmap_mode='r')
        assert data.shape == (400, 28, 23, 40) and np.isfinite(data).all()
        gray, centers, margin = E.make_gray(raw, eyes, spec)
        for i in (0, 197):
            direct = E.direct_features(gray[i], spec, margin)
            err = float(np.max(np.abs(direct-data[i])))
            np.testing.assert_allclose(direct, data[i], atol=2e-4, rtol=3e-5)
            checks.append(dict(input=spec['id'], image_index=i, max_abs_error=err))
        manifest.append(dict(id=spec['id'], path=item['path'], sha256=item['sha256'], spec=spec))
    splits = [dict(seed=s, learn=E.split(E.LABELS, 2, s)[0].tolist(),
                   query=E.split(E.LABELS, 2, s)[1].tolist()) for s in SEEDS]
    for s in splits:
        assert not set(s['learn']) & set(s['query'])
        np.testing.assert_array_equal(np.bincount(E.LABELS[s['learn']])[1:], np.full(40, 2))
        np.testing.assert_array_equal(np.bincount(E.LABELS[s['query']])[1:], np.full(40, 8))
    historical_root=BASE/'results/orl_unknown_parameters'
    historical_files=[historical_root/'evaluation/results.csv',historical_root/'verification.json',historical_root/'feature_manifest.json']
    for row in csv_rows(historical_files[0]):
        if row['stage']=='stage_C' or (row['stage']=='stage0_context' and row['method']=='CSA24_LDA23'):
            historical_files.extend([BASE/row['model_file'],BASE/row['prediction_file']])
    dump(ROOT/'protocol.json', dict(date='2026-10-08', seeds=SEEDS, inputs=specs(),
        cases=E.CASES, outer_fits=320, own_gallery_count=80, probe_count=320,
        CSA=dict(center=False, init='identity', Tmax=3), PCA=dict(center=True, solver='gram'),
        LDA=dict(alpha=0, direction_scale='unit'), NN='gallery-centered cosine',
        Gabor='5x8,lambda4*sqrt(2)^s,39x39 finite mean removed,magnitude power1,stride00,zero boundary',
        alignment='YuNet eyes ->(13,24)/(33,24),linear replicate56x46,PIL equalize central crop',
        robustness=dict(shifts=SHIFTS, units='source-image pixels', both_eyes_same_translation=True,
                        only_probe_transformed=True, extra_fits=0, model_and_gallery_fixed=True),
        error_analysis='saved stage_C versus stage0_context, same3200 repeated probe records; not independent images',
        benchmark='saved stage_C, seeds20/25/29,CSA24 andPCA24; 1warmup+7runs; projection+LDA+NN320probes,excludesGabor',
        source_sha256=hashes,historical_sha256={p.relative_to(BASE).as_posix():file_hash(p) for p in historical_files},
        eyes_sha256=file_hash(BASE/'results/orl_protocol_audit/yunet_eye_coordinates.csv'),
        raw_sha256={n:file_hash(BASE/'data/raw/orl'/n) for n in NAMES},
        limits=['context is a field-of-view extension','unknown settings not author-confirmed',
                'same400images reused,not blind','paper best versus repeated mean differs',
                'twenty overlapping splits are not independent subject samples',
                'no data-dependent input/seed selection; descriptive paired effects only']))
    dump(ROOT/'splits.json', splits)
    dump(ROOT/'feature_manifest.json', manifest)
    dump(ROOT/'feature_verification.json', dict(passed=True, direct_channel_checks=320, checks=checks))
    dump(ROOT/'freeze.json', {n:file_hash(ROOT/n) for n in ['protocol.json', 'splits.json', 'feature_manifest.json']})
    print('Frozen four inputs,20splits,320fits and four probe-only shifts before scoring.', flush=True)


def check():
    for n,h in read(ROOT/'freeze.json').items():
        assert file_hash(ROOT/n)==h, n
    p=read(ROOT/'protocol.json')
    for n,h in p['source_sha256'].items():
        assert file_hash(BASE/n)==h, n
    for e in read(ROOT/'feature_manifest.json'):
        assert file_hash(BASE/e['path'])==e['sha256']
    assert file_hash(BASE/'results/orl_protocol_audit/yunet_eye_coordinates.csv')==p['eyes_sha256']
    for n,h in p['raw_sha256'].items():
        assert file_hash(BASE/'data/raw/orl'/n)==h
    for n,h in p['historical_sha256'].items():
        assert file_hash(BASE/n)==h,n
    return p


def encode(m, x):
    x=np.asarray(x, dtype=float)
    if 'pca_basis' in m:
        return (x.reshape(len(x),-1)-m['mean'])@m['pca_basis']
    return np.einsum('nhwc,ha,wb,cd->nabd', x, m['axis0'], m['axis1'], m['axis2'], optimize=True).reshape(len(x),-1)


def nn(a,b,learn, independent=False):
    if independent:
        near=np.argmax((b/np.linalg.norm(b,axis=1,keepdims=True))@(a/np.linalg.norm(a,axis=1,keepdims=True)).T,axis=1)
    else:
        d=cdist(b,a,'cosine'); assert np.isfinite(d).all(); near=np.argmin(d,axis=1)
    return E.LABELS[learn[near]], learn[near]


def evaluate():
    check(); assert not (ROOT/'results.csv').exists()
    rows=[]; started=time.perf_counter()
    for entry in read(ROOT/'feature_manifest.json'):
        data=np.load(BASE/entry['path'],mmap_mode='r')
        for seed in SEEDS:
            learn,query=E.split(E.LABELS,2,seed); x,y=data[learn],data[query]
            for name,alg,dims,ld,paper in E.CASES:
                tic=time.perf_counter()
                model=E.PCA(dims,center=True,solver='gram').fit(x) if alg=='pca' else E.fit_csa(x,dims,'identity')
                g=model.transform(x).reshape(80,-1); q=model.transform(y).reshape(320,-1)
                arrays=dict(learn=learn,query=query,gallery_core=g,query_core=q,mean=model.mean)
                if alg=='pca': arrays['pca_basis']=model.basis
                else: arrays.update({f'axis{k}':u for k,u in enumerate(model.bases)})
                w=None
                if ld is not None:
                    lda=E.LDA().fit(g,E.LABELS[learn]); assert ld<=lda.max_components
                    w=lda.basis[:,:ld]
                    arrays.update(lda_basis=w,lda_mean=lda.mean,sw=lda.sw,sb=lda.sb,eigenvalues=lda.eigenvalues[:ld])
                a,b=E.outputs(g,q,w,E.POLICIES[0]); pred,near=nn(a,b,learn)
                elapsed=time.perf_counter()-tic
                stem=f'{entry["id"]}_s{seed}_{name}'
                mf=ROOT/'models'/f'{stem}.npz'; pf=ROOT/'predictions'/f'{stem}.npz'
                np.savez_compressed(mf,**arrays)
                np.savez_compressed(pf,predictions=pred,neighbor_index=near,learn=learn,query=query)
                rows.append(dict(input=entry['id'],seed=seed,method=name,correct=int(np.sum(pred==E.LABELS[query])),
                    accuracy_percent=100*float(np.mean(pred==E.LABELS[query])),paper_percent=paper,
                    fit_encode_NN_seconds=elapsed,model_file=mf.relative_to(BASE).as_posix(),prediction_file=pf.relative_to(BASE).as_posix()))
            print(f'{entry["id"]} seed{seed}:4fixed models complete',flush=True)
    assert len(rows)==320
    table(ROOT/'results.csv',rows)
    dump(ROOT/'execution.json',dict(outer_fits=320,prediction_records=102400,elapsed_seconds=time.perf_counter()-started,BLAS_threads=2))
    dump(ROOT/'model_freeze.json',{p.relative_to(BASE).as_posix():file_hash(p)
        for p in [ROOT/'results.csv']+[BASE/r[k] for r in rows for k in ['model_file','prediction_file']]})


def check_models():
    for n,h in read(ROOT/'model_freeze.json').items():
        assert file_hash(BASE/n)==h,n


def verify():
    check();check_models(); rows=csv_rows(ROOT/'results.csv'); assert len(rows)==320
    arrays={e['id']:np.load(BASE/e['path'],mmap_mode='r') for e in read(ROOT/'feature_manifest.json')}
    expected={(e['id'],s,c[0]) for e in specs() for s in SEEDS for c in E.CASES}
    assert {(r['input'],int(r['seed']),r['method']) for r in rows}==expected
    err=residual=0.; records=0
    for r in rows:
        with np.load(BASE/r['model_file']) as m,np.load(BASE/r['prediction_file']) as p:
            learn,query=E.split(E.LABELS,2,int(r['seed']))
            np.testing.assert_array_equal(m['learn'],learn); np.testing.assert_array_equal(m['query'],query)
            g,q,e,f=E.check_model(m,arrays[r['input']],learn,query,1.,'identity','pca_basis' in m)
            err=max(err,e); residual=max(residual,f)
            a,b=E.outputs(g,q,m['lda_basis'] if 'lda_basis' in m else None,E.POLICIES[0])
            pred,near=nn(a,b,learn,True)
            np.testing.assert_array_equal(p['query'],query); np.testing.assert_array_equal(p['learn'],learn)
            np.testing.assert_array_equal(p['predictions'],pred); np.testing.assert_array_equal(p['neighbor_index'],near)
            assert int(np.sum(pred==E.LABELS[query]))==int(r['correct'])
            assert abs(100*np.mean(pred==E.LABELS[query])-float(r['accuracy_percent']))<1e-10
            records+=len(query)
        if r['method']=='PCA33_LDA25': print(f'Verified {r["input"]} seed{r["seed"]}',flush=True)
    dump(ROOT/'verification.json',dict(passed=True,models_checked=320,CSA_independent_refits=160,
        PCA_eigen_checks=160,prediction_records_checked=records,max_CSA_projector_error=err,
        max_Fisher_residual=residual,freeze_hashes_unchanged=True))


def analyze():
    check(); assert read(ROOT/'verification.json')['passed']
    rows=csv_rows(ROOT/'results.csv'); summary=[]; effects=[]
    for name,_,_,_,paper in E.CASES:
        scores={v['id']:np.array([float(next(r['accuracy_percent'] for r in rows if r['input']==v['id'] and int(r['seed'])==s and r['method']==name)) for s in SEEDS]) for v in specs()}
        for v,values in scores.items():
            delta=values-scores['2pi_crop']
            summary.append(dict(input=v,method=name,n_splits=20,mean_percent=float(values.mean()),std_pp=float(values.std(ddof=1)),
                min_percent=float(values.min()),max_percent=float(values.max()),baseline_change_pp=float(delta.mean()),
                wins=int(np.sum(delta>0)),ties=int(np.sum(delta==0)),losses=int(np.sum(delta<0)),paper_percent=paper,
                paper_difference_pp=None if paper is None else float(values.mean()-paper)))
        contrasts={'sigma_in_crop':scores['pi_crop']-scores['2pi_crop'],
            'sigma_in_context':scores['pi_context']-scores['2pi_context'],
            'context_at_2pi':scores['2pi_context']-scores['2pi_crop'],
            'context_at_pi':scores['pi_context']-scores['pi_crop'],
            'sigma_context_interaction':scores['pi_context']-scores['2pi_context']-scores['pi_crop']+scores['2pi_crop']}
        for label,d in contrasts.items():
            effects.append(dict(method=name,contrast=label,mean_change_pp=float(d.mean()),std_pp=float(d.std(ddof=1)),
                min_pp=float(d.min()),max_pp=float(d.max()),positive=int(np.sum(d>0)),zero=int(np.sum(d==0)),negative=int(np.sum(d<0))))
    table(ROOT/'summary.csv',summary); dump(ROOT/'summary.json',summary); table(ROOT/'factorial_effects.csv',effects)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for i,s in enumerate(specs()):
        group=[next(r for r in summary if r['input']==s['id'] and r['method']==c[0]) for c in E.CASES]
        ax.errorbar(np.arange(4)+(i-1.5)*.13,[r['mean_percent'] for r in group],yerr=[r['std_pp'] for r in group],fmt='o',capsize=3,label=s['id'])
    for i,c in enumerate(E.CASES):
        if c[4] is not None: ax.scatter(i,c[4],marker='*',c='black',s=95)
    ax.set(xticks=range(4),xticklabels=[c[0] for c in E.CASES],ylabel='Accuracy (%)',ylim=(65,100),title='ORL fixed sigma x context; 20 matched splits\nMean +/- sample SD; stars: paper best (different aggregation)')
    ax.grid(alpha=.2);ax.legend(ncol=2);fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


def errors():
    check(); assert not (ROOT/'historical_errors.csv').exists()
    old=BASE/'results/orl_unknown_parameters'
    assert read(old/'verification.json')['passed']
    rows=csv_rows(old/'evaluation/results.csv'); records=[]
    for seed in range(20,30):
        selected={}
        for stage,name in [('stage0_context','CSA24_LDA23'),('stage_C','CSA24_LDA23'),('stage_C','PCA24_LDA23')]:
            r=next(r for r in rows if int(r['seed'])==seed and r['stage']==stage and r['method']==name)
            with np.load(BASE/r['prediction_file']) as p:
                selected[(stage,name)]={k:p[k].copy() for k in p.files}
        base=selected[('stage0_context','CSA24_LDA23')]; csa=selected[('stage_C','CSA24_LDA23')]; pca=selected[('stage_C','PCA24_LDA23')]
        for p in [base,pca]: np.testing.assert_array_equal(p['query'],csa['query'])
        for j,index in enumerate(csa['query']):
            truth=int(E.LABELS[index]); b=int(base['predictions'][j]); c=int(csa['predictions'][j]); p=int(pca['predictions'][j])
            records.append(dict(seed=seed,image_index=int(index),file=NAMES[index],truth=truth,
                old_context_prediction=b,CSA_prediction=c,PCA_prediction=p,
                old_context_neighbor=NAMES[base['neighbor_index'][j]],CSA_neighbor=NAMES[csa['neighbor_index'][j]],PCA_neighbor=NAMES[pca['neighbor_index'][j]],
                corrected=int(b!=truth and c==truth),new_error=int(b==truth and c!=truth),
                CSA_only=int(c==truth and p!=truth),PCA_only=int(p==truth and c!=truth),
                both_correct=int(c==truth and p==truth),both_wrong=int(c!=truth and p!=truth)))
    assert len(records)==3200
    table(ROOT/'historical_errors.csv',records)
    counts={k:sum(r[k] for r in records) for k in ['corrected','new_error','CSA_only','PCA_only','both_correct','both_wrong']}
    assert counts==dict(corrected=224,new_error=111,CSA_only=158,PCA_only=249,both_correct=2653,both_wrong=140)
    photos=[]
    for index in sorted({r['image_index'] for r in records}):
        group=[r for r in records if r['image_index']==index]
        photos.append(dict(image_index=index,file=NAMES[index],probe_occurrences=len(group),
            **{k:sum(r[k] for r in group) for k in counts}))
    table(ROOT/'historical_photo_counts.csv',photos)
    raw,eyes=E.raw_inputs(); gallery=[]
    for category in ['corrected','new_error','CSA_only','PCA_only']:
        chosen=[]
        for r in records:
            if r[category] and r['image_index'] not in {x['image_index'] for x in chosen}:
                chosen.append(r)
                if len(chosen)==2:break
        gallery.extend((category,r) for r in chosen)
    canvas=Image.new('RGB',(660,len(gallery)*145),'white'); d=ImageDraw.Draw(canvas)
    for i,(category,r) in enumerate(gallery):
        d.text((5,i*145+4),f'{category}: seed{r["seed"]} {r["file"]} truth{r["truth"]} / CSA{r["CSA_prediction"]} PCA{r["PCA_prediction"]}',fill='black')
        for j,(label,name) in enumerate([('probe',r['file']),('CSA neighbor',r['CSA_neighbor']),('PCA neighbor',r['PCA_neighbor'])]):
            k=NAMES.index(name); im=Image.fromarray(raw[k]).convert('RGB'); mark=ImageDraw.Draw(im)
            for x,y in eyes[k]: mark.ellipse((x-2,y-2,x+2,y+2),outline='red',width=1)
            canvas.paste(im,(15+j*215,i*145+25));d.text((115+j*215,i*145+38),label+'\n'+name,fill='black')
    canvas.save(ROOT/'historical_error_montage.png')
    dump(ROOT/'historical_error_summary.json',dict(passed=True,repeated_probe_records=3200,
        unique_probe_photos=len(photos),counts=counts,extra_fits=0,
        montage_selection='first2 distinct photos in seed/query order per category; illustrative,not representative sample',
        interpretation='eye/pose/crop explanations require visual inspection; counts do not identify causal factors'))
    print(counts,flush=True)


def costs():
    check(); assert not (ROOT/'cost_benchmark.csv').exists()
    old=BASE/'results/orl_unknown_parameters'; rows=csv_rows(old/'evaluation/results.csv')
    manifest={e['id']:e for e in read(old/'feature_manifest.json')}; result=[]
    for seed in (20,25,29):
        for name in ['CSA24_LDA23','PCA24_LDA23']:
            r=next(r for r in rows if int(r['seed'])==seed and r['stage']=='stage_C' and r['method']==name)
            with np.load(BASE/r['model_file']) as m,np.load(BASE/r['prediction_file']) as saved:
                source=np.load(BASE/manifest[r['input']]['path'],mmap_mode='r')
                x=E.tensors(source,m['query'],float(r['power'])); policy=next(p for p in E.POLICIES if p['id']==r['policy'])
                a,_=E.outputs(m['gallery_core'],m['query_core'],m['lda_basis'],policy)
                trials=[]
                for repeat in range(8):
                    tic=time.perf_counter(); q=encode(m,x)
                    _,b=E.outputs(m['gallery_core'],q,m['lda_basis'],policy); pred,near=nn(a,b,m['learn'])
                    elapsed=time.perf_counter()-tic
                    np.testing.assert_array_equal(pred,saved['predictions']);np.testing.assert_array_equal(near,saved['neighbor_index'])
                    if repeat:trials.append(elapsed)
                weights=[m['pca_basis']] if 'pca_basis' in m else [m[f'axis{k}'] for k in range(3)]
                weights.append(m['lda_basis']); weight_elements=sum(w.size for w in weights)
                # Runtime arrays exclude training scatters,core/probe caches and indices.
                runtime_bytes=sum(w.nbytes for w in weights)+m['mean'].nbytes+m['lda_mean'].nbytes+a.nbytes+m['learn'].nbytes
                result.append(dict(seed=seed,method=name,input=r['input'],power=float(r['power']),projection_weight_elements=weight_elements,
                    projection_weight_bytes=sum(w.nbytes for w in weights),runtime_arrays_bytes=runtime_bytes,
                    median_320probe_projection_LDA_NN_seconds=float(np.median(trials)),minimum_seconds=min(trials),maximum_seconds=max(trials),
                    warmups=1,timed_repeats=7,extra_fits=0))
    table(ROOT/'cost_benchmark.csv',result)
    dump(ROOT/'cost_verification.json',dict(passed=True,models=6,prediction_replay_records=6*8*320,
        CSA24_LDA23_weights=830,PCA24_LDA23_weights=618792,
        definition='actual float64 ndarray nbytes; not compressed archive size,peak memory,or total Python process memory',
        timing='precomputed feature batch; excludes Gabor/preprocessing/training/disk writes; per-case 1warmup+7runs;2BLASthreads',
        platform=platform.platform(),python=platform.python_version()))
    print(result,flush=True)


def robustness():
    check();check_models(); assert read(ROOT/'verification.json')['passed']; assert not (ROOT/'robustness.csv').exists()
    raw,eyes=E.raw_inputs(); rows=csv_rows(ROOT/'results.csv'); result=[]; feature_checks=[]; started=time.perf_counter()
    for spec in specs():
        original=next(e for e in read(ROOT/'feature_manifest.json') if e['id']==spec['id'])
        gray,_,margin=E.make_gray(raw,eyes,spec)
        # Zero-shift construction check before changing any eye coordinate.
        rebuilt=E.feature_array(gray[:2],spec,margin)
        np.testing.assert_array_equal(rebuilt,np.load(BASE/original['path'],mmap_mode='r')[:2])
        for shift,dx,dy in SHIFTS:
            perturbed=eyes.copy();perturbed[:,:,0]+=dx;perturbed[:,:,1]+=dy
            gray,_,margin=E.make_gray(raw,perturbed,spec); data=E.feature_array(gray,spec,margin)
            direct=E.direct_features(gray[0],spec,margin)
            np.testing.assert_allclose(direct,data[0],atol=2e-4,rtol=3e-5)
            feature_checks.append(dict(input=spec['id'],shift=shift,max_abs_error=float(np.max(np.abs(direct-data[0])))))
            for r in (r for r in rows if r['input']==spec['id']):
                with np.load(BASE/r['model_file']) as m,np.load(BASE/r['prediction_file']) as clean:
                    q=encode(m,data[m['query']]); w=m['lda_basis'] if 'lda_basis' in m else None
                    a,b=E.outputs(m['gallery_core'],q,w,E.POLICIES[0]);pred,near=nn(a,b,m['learn'])
                    pred2,near2=nn(a,b,m['learn'],True)
                    np.testing.assert_array_equal(pred,pred2);np.testing.assert_array_equal(near,near2)
                    truth=E.LABELS[m['query']];bc=clean['predictions']==truth;pc=pred==truth
                    pf=ROOT/'robust_predictions'/f'{spec["id"]}_{shift}_s{r["seed"]}_{r["method"]}.npz'
                    np.savez_compressed(pf,query=m['query'],learn=m['learn'],predictions=pred,neighbor_index=near)
                    result.append(dict(input=spec['id'],shift=shift,seed=int(r['seed']),method=r['method'],
                        clean_percent=float(r['accuracy_percent']),perturbed_percent=100*float(pc.mean()),change_pp=100*float(pc.mean())-float(r['accuracy_percent']),
                        corrected=int(np.sum(~bc&pc)),new_error=int(np.sum(bc&~pc)),prediction_file=pf.relative_to(BASE).as_posix()))
            print(f'Probe-only {spec["id"]} {shift}:80saved models,no refits',flush=True)
    assert len(result)==1280
    table(ROOT/'robustness.csv',result)
    summary=[]
    for spec in specs():
        for name,*_ in E.CASES:
            for shift,_,_ in SHIFTS:
                g=[r for r in result if r['input']==spec['id'] and r['method']==name and r['shift']==shift]
                delta=np.array([r['change_pp'] for r in g])
                summary.append(dict(input=spec['id'],method=name,shift=shift,n_splits=20,
                    mean_clean_percent=float(np.mean([r['clean_percent'] for r in g])),mean_perturbed_percent=float(np.mean([r['perturbed_percent'] for r in g])),
                    mean_change_pp=float(delta.mean()),std_change_pp=float(delta.std(ddof=1)),corrected=sum(r['corrected'] for r in g),new_error=sum(r['new_error'] for r in g)))
    table(ROOT/'robustness_summary.csv',summary)
    check();check_models()
    dump(ROOT/'robustness_verification.json',dict(passed=True,extra_fits=0,scores=1280,prediction_records_checked=409600,
        independent_normalized_dot_NN=True,zero_shift_feature_checks='first2images all40channels for all4inputs exact',
        direct_channel_checks=640,checks=feature_checks,source_input_model_settings_unchanged=True,
        elapsed_seconds=time.perf_counter()-started,limits='common translations of both eyes in source pixels only; no independent eye displacement or rotation'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['prepare','evaluate','verify','analyze','errors','costs','robustness'])
    parser.add_argument('--output',default='results/orl_controlled');args=parser.parse_args();ROOT=BASE/args.output
    with threadpool_limits(limits=2):globals()[args.stage]()
