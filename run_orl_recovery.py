"""Bounded ORL 4x2x3 -> ordinary LDA23 recovery study.

All ten historical splits and all candidates are retained. Probe-ranked means
and maxima are exploratory; the fixed default and gallery-SSE starts are also
reported separately. This does not recover unknown author settings.
"""
import argparse
import csv
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA, PCA, LDA, mode_product, unfold, leading_vectors
from gabor import file_hash
import optimize_orl_unknowns as E

BASE = Path(__file__).resolve().parent
SEEDS = list(range(20, 30))
RANKS = (4, 2, 3)
STARTS = ['rectangular_identity', 'full_identity', 'hosvd']
POLICIES = [dict(id=f'{scale}_{origin}_{metric}_g0', scale=scale,
                 origin=origin, metric=metric, gamma=0.)
            for scale in ['unit', 'sw']
            for origin, metric in [('gallery', 'cosine'), ('linear', 'cosine'),
                                   ('gallery', 'euclidean')]]
POLICIES += [dict(id=f'unit_gallery_{metric}_g{gamma}', scale='unit',
                  origin='gallery', metric=metric, gamma=gamma)
             for metric in ['cosine', 'euclidean'] for gamma in [.25, .5, 1.]]
SOURCES = ['run_orl_recovery.py', 'csa.py', 'gabor.py',
           'optimize_orl_unknowns.py', 'run_orl_fixed_inputs.py',
           'prepare_available_data.py', 'run_available_reproduction.py']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_fit(x, start):
    """Explicit sequential covariance updates; full start is an ambiguity study."""
    model = CSA(RANKS, center=False, max_iter=3)
    model.shape = x.shape[1:]
    model.mean = np.zeros(model.shape)
    if start == 'full_identity':
        model.bases = [np.eye(m) for m in model.shape]
    elif start == 'hosvd':
        model.bases = []
        for k, rank in enumerate(RANKS):
            b = unfold(x, k)
            model.bases.append(leading_vectors(np.einsum('nac,nbc->ab', b, b), rank))
    else:
        model.bases = [np.eye(m, r) for m, r in zip(model.shape, RANKS)]
    initial = [u.copy() for u in model.bases]
    energy = float(np.sum(x*x))
    trajectory = []
    for iteration in range(3):
        for k, rank in enumerate(RANKS):
            z = x
            for j, u in enumerate(model.bases):
                if j != k:
                    z = mode_product(z, u.T, j)
            b = unfold(z, k)
            cov = np.einsum('nac,nbc->ab', b, b, optimize=True)
            model.bases[k] = leading_vectors(cov, rank)
            # Independent covariance accumulation, not the fitting einsum.
            direct = sum(v @ v.T for v in b)
            np.testing.assert_allclose(cov, direct, rtol=2e-12, atol=1e-4)
            residual = np.linalg.norm(cov @ model.bases[k] -
                       model.bases[k] @ (model.bases[k].T @ cov @ model.bases[k]))
            assert residual / max(1., np.linalg.norm(cov)) < 1e-10
        retained = float(np.sum(model.transform(x)**2))
        trajectory.append(dict(iteration=iteration+1, retained_energy=retained,
                               sse=max(0., energy-retained)))
    assert all(trajectory[i+1]['retained_energy'] >= trajectory[i]['retained_energy'] - 1e-5
               for i in range(2))
    for u in model.bases:
        np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-10)
    if start != 'full_identity':
        other = CSA(RANKS, center=False, max_iter=3, initial_bases=initial).fit(x)
        for u, v in zip(model.bases, other.bases):
            np.testing.assert_allclose(u @ u.T, v @ v.T, atol=2e-9, rtol=2e-9)
    return model, trajectory


def prepare(root):
    root.mkdir(exist_ok=False)
    for folder in ['features', 'models', 'source_snapshot']:
        (root/folder).mkdir()
    for filename in SOURCES:
        shutil.copy2(BASE/filename, root/'source_snapshot'/filename)
    base = next(s for s in E.inputs() if s['id'] == 'context_sigma_pi')
    specs = [dict(base, id=name, distance=distance, eye_y=y)
             for name, distance, y in [('d20_y24', 20., 24.),
                                      ('d18_y25', 18., 25.), ('d22_y22', 22., 22.)]]
    splits = [dict(seed=s, gallery=E.split(E.LABELS, 2, s)[0].tolist(),
                   probe=E.split(E.LABELS, 2, s)[1].tolist()) for s in SEEDS]
    protocol = dict(date='2026-10-27', seeds=SEEDS, inputs=specs,
        ranks=RANKS, LDA=23, Tmax=3, center=False, magnitude_power=1,
        starts=STARTS, policies=POLICIES, source_sha256={f:file_hash(BASE/f) for f in SOURCES},
        paper_target=90.9, gallery_count=80, probe_count=320,
        planned_CSA_fits=90, planned_PCA24_controls=30,
        primary_policy=POLICIES[0]['id'],
        selection='All settings frozen before scoring. Best ten-split means and single maxima are probe-selected exploration. Gallery-SSE initialization is training-only, but its input/metric must remain fixed.',
        limits=['same400images and previously observed splits, not blind',
                'context changes field of view', 'full-square start is an initialization interpretation, not an author-confirmed fix',
                'gamma>0 is Fisher-eigenvalue metric weighting, an extension',
                'no shrinkage, new ranks, extra gallery images or discarded splits',
                'best-result wording does not establish author seed cherry-picking'])
    dump(root/'protocol.json', protocol)
    dump(root/'splits.json', splits)
    raw, eyes = E.raw_inputs()
    checks = []
    for spec in specs:
        gray, _, margin = E.make_gray(raw, eyes, spec)
        features = E.feature_array(gray, spec, margin)
        for i in [0, 197]:
            reference = E.direct_features(gray[i], spec, margin)
            np.testing.assert_allclose(features[i], reference, atol=2e-4, rtol=3e-5)
            checks.append(dict(input=spec['id'], image_index=i,
                               max_abs=float(np.max(np.abs(features[i]-reference)))))
        np.save(root/'features'/f"{spec['id']}.npy", features)
    historical = next(e for e in read(BASE/'results/orl_unknown_parameters/feature_manifest.json')
                      if e['id']=='context_sigma_pi')
    assert file_hash(root/'features/d20_y24.npy') == historical['sha256']
    dump(root/'feature_checks.json', dict(passed=True, checks=checks,
         baseline_feature_hash_matches=True))
    dump(root/'input_freeze.json', {p.relative_to(root).as_posix():file_hash(p)
         for p in [root/'protocol.json', root/'splits.json', *sorted((root/'features').glob('*.npy'))]})
    print('Frozen 3 geometries, 3 CSA starts, 12 NN policies, ten splits.', flush=True)


def fit(root):
    p = read(root/'protocol.json')
    for f, sha in p['source_sha256'].items():
        assert file_hash(BASE/f)==sha
    records = []
    for spec in p['inputs']:
        data = np.load(root/'features'/f"{spec['id']}.npy", mmap_mode='r')
        for split in read(root/'splits.json'):
            g = np.array(split['gallery'])
            assert len(g)==80 and not set(g)&set(split['probe'])
            assert np.all(np.bincount(E.LABELS[g])[1:]==2)
            x = np.asarray(data[g], dtype=float)
            for start in STARTS+['PCA24']:
                if start == 'PCA24':
                    model = PCA(24, center=True, solver='gram').fit(x)
                    history = []
                    saved = dict(basis=model.basis, mean=model.mean)
                else:
                    model, history = reference_fit(x, start)
                    saved = {f'u{k}':u for k,u in enumerate(model.bases)}
                    saved['mean'] = model.mean
                core = model.transform(x).reshape(80, 24)
                lda = LDA().fit(core, E.LABELS[g])
                assert lda.max_components>=23
                values, _ = eigh(lda.sb, lda.sw)
                np.testing.assert_allclose(values[::-1][:23], lda.eigenvalues[:23], rtol=1e-7, atol=1e-7)
                w = lda.basis[:, :23]
                residual = np.linalg.norm(lda.sb@w-(lda.sw@w)*lda.eigenvalues[:23]) / max(1.,np.linalg.norm(lda.sb@w))
                assert residual<1e-9
                saved.update(lda_basis=w, lda_mean=lda.mean, eigenvalues=lda.eigenvalues[:23],
                             gallery_core=core, gallery=g)
                filename = f"{spec['id']}_{split['seed']}_{start}.npz"
                np.savez(root/'models'/filename, **saved)
                records.append(dict(input=spec['id'], seed=split['seed'], start=start,
                    model='models/'+filename, model_sha256=file_hash(root/'models'/filename),
                    gallery_sse=history[-1]['sse'] if history else '', fisher_residual=residual,
                    trajectory=json.dumps(history)))
            print('Fitted',spec['id'],split['seed'],flush=True)
    table(root/'models.csv',records)
    dump(root/'model_freeze.json', dict(model_count=len(records),
         models_csv_sha256=file_hash(root/'models.csv'), all_fits_before_any_probe_scores=True))


def score(root):
    p=read(root/'protocol.json')
    rows=E.csv_rows(root/'models.csv')
    assert file_hash(root/'models.csv')==read(root/'model_freeze.json')['models_csv_sha256']
    splits={s['seed']:s for s in read(root/'splits.json')}
    results=[]
    predictions=[]
    for spec in p['inputs']:
        data=np.load(root/'features'/f"{spec['id']}.npy",mmap_mode='r')
        for row in [r for r in rows if r['input']==spec['id']]:
            assert file_hash(root/row['model'])==row['model_sha256']
            saved=np.load(root/row['model']);g=saved['gallery'];q=np.array(splits[int(row['seed'])]['probe'])
            if row['start']=='PCA24':
                query=(np.asarray(data[q],dtype=float).reshape(320,-1)-saved['mean'])@saved['basis']
            else:
                query=np.asarray(data[q],dtype=float)
                for k in range(3):query=mode_product(query,saved[f'u{k}'].T,k)
                query=query.reshape(320,24)
            for policy in p['policies']:
                w=saved['lda_basis'].copy()
                if policy['scale']=='unit':w/=np.linalg.norm(w,axis=0)
                if policy['gamma']:
                    ev=np.maximum(saved['eigenvalues'],0)
                    w*=np.maximum(ev/max(ev),1e-12)**policy['gamma']
                origin=saved['lda_mean'] if policy['origin']=='gallery' else 0.
                gallery=(saved['gallery_core']-origin)@w;probe=(query-origin)@w
                distances=cdist(probe,gallery,metric=policy['metric'])
                winner=distances.argmin(axis=1)
                if policy['metric']=='cosine':
                    a=probe/np.linalg.norm(probe,axis=1,keepdims=True)
                    b=gallery/np.linalg.norm(gallery,axis=1,keepdims=True)
                    independent=(a@b.T).argmax(axis=1)
                else:
                    independent=np.array([np.sum((gallery-v)**2,axis=1).argmin() for v in probe])
                np.testing.assert_array_equal(winner,independent)
                pred=E.LABELS[g[winner]];correct=int(np.sum(pred==E.LABELS[q]))
                result=dict(input=row['input'],seed=int(row['seed']),start=row['start'],
                    policy=policy['id'],gamma=policy['gamma'],correct=correct,probe_count=320,
                    accuracy_percent=100*correct/320)
                results.append(result)
                predictions.extend(dict(input=row['input'],seed=int(row['seed']),start=row['start'],
                    policy=policy['id'],probe_index=int(i),neighbor_index=int(j),
                    predicted_label=int(label),correct=int(label==E.LABELS[i]))
                    for i,j,label in zip(q,g[winner],pred))
    table(root/'results.csv',results)
    table(root/'predictions.csv',predictions)
    summary=[]
    for key in itertools.product([s['id'] for s in p['inputs']],STARTS+['PCA24'],[x['id'] for x in p['policies']]):
        matching=[r for r in results if (r['input'],r['start'],r['policy'])==key]
        assert sorted(r['seed'] for r in matching)==SEEDS
        values=np.array([r['accuracy_percent'] for r in matching])
        summary.append(dict(input=key[0],start=key[1],policy=key[2],n_splits=10,
            mean_percent=float(values.mean()),std_pp=float(values.std(ddof=1)),
            min_percent=float(values.min()),max_percent=float(values.max()),
            mean_minus_paper_pp=float(values.mean()-90.9),at_least_paper=int(np.sum(values>=90.9))))
    table(root/'summary.csv',summary)
    eligible=[r for r in summary if r['start']!='PCA24']
    best=max(eligible,key=lambda r:r['mean_percent'])
    primary=[r for r in summary if r['policy']==p['primary_policy']]
    dump(root/'analysis.json',dict(selection='Exploratory best over all fixed candidates on ten already-seen probe splits',
        best_mean=best,best_unweighted=max([r for r in eligible if r['policy'].endswith('_g0')],key=lambda r:r['mean_percent']),
        primary_policy_results=primary,
        global_single_max=max(r['accuracy_percent'] for r in results if r['start']!='PCA24'),
        limits=p['limits']))
    dump(root/'verification.json',dict(passed=True,model_count=len(rows),score_count=len(results),
        prediction_count=len(predictions),all_ten_splits_present=True,independent_NN_predictions_match=True,
        independent_covariance_and_fisher_checks=True,source_and_model_hashes_unchanged=True,
        heldout_selection_claim=False))
    print(json.dumps(dict(best_mean=best),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/orl_recovery_20261027')
    parser.add_argument('--stage',choices=['prepare','fit','score','all'],default='all')
    args=parser.parse_args();root=BASE/args.output
    with threadpool_limits(limits=2):
        if args.stage in ['prepare','all']:prepare(root)
        if args.stage in ['fit','all']:fit(root)
        if args.stage in ['score','all']:score(root)


if __name__=='__main__':
    main()
