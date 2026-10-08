"""Four crop-only Gabor inputs, original dimensions, own-gallery selection.

Reuse the frozen model/verification helpers without editing their source.
No context, response power, regularized LDA, dimension or seed search.
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import optimize_orl_unknowns as E

BASE = E.BASE
ROOT = BASE/'results/orl_ta_crop_optimized'
SEEDS, LABELS, CASES, POLICIES = E.SEEDS, E.LABELS, E.CASES, E.POLICIES
STARTS = ['identity', 'hosvd']
SOURCES = ['optimize_orl_ta_crop.py']+E.SOURCES
read, dump, table, csv_rows, file_hash = E.read, E.dump, E.table, E.csv_rows, E.file_hash


def inputs():
    common = {'wavelength': 4., 'support': 19, 'finite_dc': True, 'distance': 20.,
              'eye_y': 24., 'histogram': 'PIL', 'context': False, 'downsample_first': False}
    return [{'id': f'crop_{name}_{boundary}', **common, 'sigma': float(sigma), 'boundary': boundary}
            for name, sigma in [('2pi', 2*np.pi), ('pi', np.pi)]
            for boundary in ['zero', 'reflect']]


def rank(row):
    return (row['inner_correct'], row['worst_fold_correct'],
            -[s['id'] for s in inputs()].index(row['input']), -STARTS.index(row['init']),
            -[p['id'] for p in POLICIES].index(row['policy']))


def features(gray, spec):
    result = np.empty((len(gray), 28, 23, 40), np.float32)
    mode = 'constant' if spec['boundary'] == 'zero' else 'reflect'
    r = 19; shape = (next_fast_len(56+4*r), next_fast_len(46+4*r))
    for s in range(5):
        kernels = [fftn(E.kernel(spec, s, o), s=shape) for o in range(8)]
        for start in range(0, len(gray), 32):
            padded = np.pad(gray[start:start+32], ((0, 0), (r, r), (r, r)), mode=mode)
            spectrum = fftn(padded, s=shape, axes=(-2, -1))
            for o, kernel in enumerate(kernels):
                response = np.abs(ifftn(spectrum*kernel, axes=(-2, -1)))
                result[start:start+len(padded), :, :, s*8+o] = response[:, 2*r:2*r+56:2, 2*r:2*r+46:2]
    return result


def direct_features(gray, spec):
    mode = 'constant' if spec['boundary'] == 'zero' else 'reflect'
    result = np.empty((28, 23, 40))
    for s in range(5):
        for o in range(8):
            kernel = E.kernel(spec, s, o).astype(np.complex128)
            response = np.abs(convolve2d(np.pad(gray.astype(float), 19, mode=mode), kernel, mode='valid'))
            result[:, :, s*8+o] = response[::2, ::2]
    return result


def prepare():
    ROOT.mkdir(exist_ok=False)
    for name in ['features', 'cv', 'selection', 'evaluation', 'source_snapshot']:
        (ROOT/name).mkdir()
    prior = BASE/'results/orl_fixed_input_extensions'
    assert read(prior/'verification.json')['passed']
    anchor = read(prior/'feature_manifest.json')['baseline']
    protocol = {'date': '2026-10-08', 'seeds': SEEDS,
        'outer': 'all400 ORL images,40classes,2gallery/8probe per class',
        'inner': 'own80gallery only,16folds of75learning/5validation,all40classes; each validated once',
        'inputs': inputs(), 'powers': [1.], 'starts': STARTS, 'policies': POLICIES,
        'geometry': 'YuNet; target(13,24)/(33,24),cv2 linear warp/replicate,56x46; PIL equalize',
        'Gabor': '5scales wavelength4*sqrt(2)^s,8orientations o*pi/8,39x39 finite-mean-zero complex kernels; magnitude then stride00',
        'boundary_definition': 'zero=np.pad constant0; reflect=np.pad reflect,edge excluded; no raw-image context',
        'CSA_center': False, 'Tmax': 3, 'LDA_alpha': 0.,
        'dimensions': 'CSA4x2x3->LDA23,CSA15x1x26,PCA24->LDA23,PCA33->LDA25',
        'LDA_limit': '24<=75-40 inside;24/33<=80-40 outside;390 has no LDA',
        'target': 'CSA24+ordinaryLDA23; four outer methods share input/init(where applicable)/NN policy',
        'stage_A': '4inputs,identity,unit/gallery/cosine; complete16folds,keep2',
        'stage_B': 'best2inputs x2starts x6policies=24 complete-CV configurations',
        'tie_rule': 'inner correct,worst fold,declared input order,identity,declared policy order',
        'freeze': 'all10 choices and CV models hashed; independently verify before any outer scoring',
        'baseline': {'root': str(prior.relative_to(BASE)), 'feature': anchor,
                     'score_sha256': file_hash(prior/'results.csv'),
                     'verification_sha256': file_hash(prior/'verification.json')},
        'raw_manifest_sha256': file_hash(prior/'raw_manifest.csv'),
        'eyes_sha256': file_hash(BASE/'results/orl_protocol_audit/yunet_eye_coordinates.csv'),
        'source_sha256': {n: file_hash(BASE/n) for n in SOURCES},
        'limits': ['author numeric settings not confirmed; TA explicitly allows remaining alignment/Gabor choices with disclosure',
                   'NN/init/scaling are disclosed implementations,not individually TA-confirmed',
                   'historical probes observed; not blind', 'paper best versus repeated mean differs',
                   '4input pruning by default policy may omit a better input under another policy',
                   'per-gallery validation winners,not global optimality; controls not separately optimized'],
        'stop': 'all10 selections and40new outer fits; retain negatives; no more candidates/seeds'}
    dump(ROOT/'protocol.json', protocol)
    for n in SOURCES:
        shutil.copy2(BASE/n, ROOT/'source_snapshot'/n)
    raw, eyes = E.raw_inputs()
    gray, center, margin = E.make_gray(raw, eyes, inputs()[0])
    assert margin == 0 and gray.shape == (400, 56, 46)
    np.testing.assert_array_equal(gray, center)
    source_gray = next(e for e in read(BASE/'results/orl_protocol_audit/catalogue.json')
                       if e['id'] == 'yunet_d20_replicate_fixed39_zero')
    assert file_hash(BASE/source_gray['gray_path']) == source_gray['gray_sha256']
    oldgray = np.load(BASE/source_gray['gray_path'], mmap_mode='r')
    np.testing.assert_array_equal(gray, oldgray)
    np.save(ROOT/'features/gray56.npy', gray)
    entries, errors = [], []
    for spec in inputs():
        if spec['id'] == 'crop_2pi_zero':
            path = BASE/anchor['path']; assert file_hash(path) == anchor['sha256']
            a = np.load(path, mmap_mode='r')
            np.testing.assert_array_equal(features(gray[:2], spec), a[:2])
        else:
            path = ROOT/'features'/f"{spec['id']}.npy"
            a = features(gray, spec); np.save(path, a)
        assert a.shape == (400, 28, 23, 40) and np.isfinite(a).all()
        for index in [0, 197]:
            error = float(np.max(np.abs(direct_features(gray[index], spec)-a[index])))
            assert error < 3e-4, (spec['id'], index, error)
            errors.append({'input': spec['id'], 'image_index': index, 'max_abs_error': error})
        entries.append({'id': spec['id'], 'path': path.relative_to(BASE).as_posix(),
                        'sha256': file_hash(path), 'spec': spec})
        print(f"Prepared {spec['id']}; original crop and magnitude retained", flush=True)
    dump(ROOT/'feature_manifest.json', entries)
    dump(ROOT/'feature_verification.json', {'passed': True, 'all400_gray_centers_exact': True,
         'baseline_first2_features_exact': True, 'channel_checks': 320, 'checks': errors})
    dump(ROOT/'input_freeze.json', {n: file_hash(ROOT/n) for n in ['protocol.json', 'feature_manifest.json']})


def data_arrays():
    return {e['id']: np.load(BASE/e['path'], mmap_mode='r') for e in read(ROOT/'feature_manifest.json')}


def paths(seed, row):
    return [ROOT/'cv'/f'seed{seed}'/E.key(row['input'], 1., row['init'])/f'fold{i:02d}.npz' for i in range(16)]


def select():
    E.check_inputs(); assert not (ROOT/'selection_freeze.json').exists()
    data = data_arrays(); execution = []; started = time.perf_counter()
    for seed in SEEDS:
        folder = ROOT/'selection'/f'seed{seed}'; folder.mkdir(exist_ok=False)
        a = []
        for spec in inputs():
            row = E.candidate(seed, spec['id'], 1., 'identity', POLICIES[0], data[spec['id']])
            a.append(row); print(f"seed{seed} input {spec['id']}: {row['inner_correct']}/80", flush=True)
        top2 = sorted(a, key=rank, reverse=True)[:2]; b = []
        for r in top2:
            for init in STARTS:
                cv = E.cv_models(seed, r['input'], 1., init, data[r['input']])
                for policy in POLICIES:
                    n, worst, pred, near = E.score_cv(cv, policy)
                    b.append({'input': r['input'], 'power': 1., 'init': init, 'policy': policy['id'],
                              'inner_correct': n, 'inner_total': 80, 'worst_fold_correct': worst,
                              'predictions': pred, 'neighbors': near})
        chosen = {'baseline': a[0], 'input_A': max(a, key=rank), 'final_B': max(b, key=rank)}
        assert [r['inner_correct'] for r in chosen.values()] == sorted(r['inner_correct'] for r in chosen.values())
        dump(folder/'choices.json', {'seed': seed, 'top2': [r['input'] for r in top2], 'choices': chosen})
        for stage, rows in [('A', a), ('B', b)]:
            dump(folder/f'candidates_{stage}.json', rows)
            table(folder/f'candidates_{stage}.csv', [{k: v for k, v in r.items() if k not in ['predictions', 'neighbors']} for r in rows])
        execution.extend({'seed': seed, 'stage': s, **{k: v for k, v in r.items() if k not in ['predictions', 'neighbors']}}
                         for s, r in chosen.items())
        print(f"seed{seed} frozen: {[r['inner_correct'] for r in chosen.values()]}", flush=True)
    table(ROOT/'selected_settings.csv', execution)
    files = list((ROOT/'selection').rglob('*'))+list((ROOT/'cv').rglob('*.npz'))+[ROOT/'selected_settings.csv']
    dump(ROOT/'selection_freeze.json', {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files if p.is_file()})
    fits = len(list((ROOT/'cv').rglob('*.npz'))); assert fits == 960
    dump(ROOT/'selection_execution.json', {'CV_model_fits': fits, 'candidate_rows': 280,
        'elapsed_seconds': time.perf_counter()-started, 'outer_probe_scored': False, 'all10_choices_frozen': True})


def verify_selection():
    E.check_selections(); data = data_arrays(); checked = set(); records = 0; error = residual = 0.
    for seed in SEEDS:
        folder = ROOT/'selection'/f'seed{seed}'; choice = read(folder/'choices.json')['choices']
        for stage, selected in [('A', 'input_A'), ('B', 'final_B')]:
            rows = read(folder/f'candidates_{stage}.json'); assert max(rows, key=rank) == choice[selected]
            for row in rows:
                policy = next(p for p in POLICIES if p['id'] == row['policy'])
                n, worst, pred, near = E.score_cv(paths(seed, row), policy, independent=True)
                assert (n, worst, pred, near) == (row['inner_correct'], row['worst_fold_correct'], row['predictions'], row['neighbors'])
                records += 80
        train, probe = E.split(LABELS, 2, seed)
        for row in choice.values():
            for path, (learn, query) in zip(paths(seed, row), E.folds(train, seed)):
                if path in checked:
                    continue
                with np.load(path) as m:
                    np.testing.assert_array_equal(m['learn'], learn); np.testing.assert_array_equal(m['query'], query)
                    assert not np.intersect1d(learn, probe).size and not np.intersect1d(query, probe).size
                    _, _, e, r = E.check_model(m, data[row['input']], learn, query, 1., row['init'])
                error, residual = max(error, e), max(residual, r); checked.add(path)
        print(f'All candidate NN and selected CV checks passed seed{seed}', flush=True)
    assert records == 22400
    dump(ROOT/'selection_verification.json', {'passed': True, 'candidate_rows_checked': 280,
        'validation_prediction_records_checked': records, 'selected_CV_independent_refits': len(checked),
        'max_CSA_projector_abs_error': error, 'max_Fisher_relative_residual': residual,
        'own_gallery_only': True, 'all_frozen_hashes_unchanged': True})


def evaluate():
    E.check_selections(); assert read(ROOT/'selection_verification.json')['passed']
    assert not (ROOT/'evaluation/results.csv').exists()
    for name in ['models', 'predictions']:
        (ROOT/'evaluation'/name).mkdir()
    data = data_arrays(); rows = []; started = time.perf_counter()
    prior = BASE/'results/orl_fixed_input_extensions'
    protocol = read(ROOT/'protocol.json')
    assert file_hash(prior/'results.csv') == protocol['baseline']['score_sha256']
    for r in csv_rows(prior/'results.csv'):
        if r['variant'] != 'baseline':
            continue
        rows.append({'seed': int(r['seed']), 'stage': 'baseline', 'method': r['method'],
            'input': 'crop_2pi_zero', 'power': 1., 'init': 'identity', 'policy': POLICIES[0]['id'],
            'correct': int(r['correct']), 'accuracy_percent': float(r['accuracy_percent']),
            'paper_percent': float(r['paper_percent']) if r['paper_percent'] else None,
            'model_file': (prior/r['model_file']).relative_to(BASE).as_posix(),
            'prediction_file': (prior/r['prediction_file']).relative_to(BASE).as_posix(), 'reused': True})
    for seed in SEEDS:
        row = read(ROOT/'selection'/f'seed{seed}'/'choices.json')['choices']['final_B']
        learn, query = E.split(LABELS, 2, seed); x, y = data[row['input']][learn], data[row['input']][query]
        policy = next(p for p in POLICIES if p['id'] == row['policy'])
        for name, alg, dims, ld, paper in CASES:
            model = E.PCA(dims, center=True, solver='gram').fit(x) if alg == 'pca' else E.fit_csa(x, dims, row['init'])
            g = model.transform(x).reshape(80, -1); q = model.transform(y).reshape(320, -1)
            arrays = {'learn': learn, 'query': query, 'gallery_core': g, 'query_core': q, 'mean': model.mean}
            if alg == 'pca':
                arrays['pca_basis'] = model.basis
            else:
                arrays.update({f'axis{k}': u for k, u in enumerate(model.bases)})
            w = None
            if ld is not None:
                lda = E.LDA().fit(g, LABELS[learn]); assert ld <= lda.max_components
                w = lda.basis[:, :ld]
                arrays.update(lda_basis=w, lda_mean=lda.mean, sw=lda.sw, sb=lda.sb, eigenvalues=lda.eigenvalues[:ld])
            a, b = E.outputs(g, q, w, policy)
            distance = cdist(b, a, policy['metric']); assert np.isfinite(distance).all()
            near = np.argmin(distance, axis=1); pred = LABELS[learn[near]]
            stem = f'final_seed{seed}_{name}'; mf = ROOT/'evaluation/models'/f'{stem}.npz'; pf = ROOT/'evaluation/predictions'/f'{stem}.npz'
            np.savez_compressed(mf, **arrays)
            np.savez_compressed(pf, learn=learn, query=query, predictions=pred, neighbor_index=learn[near])
            rows.append({'seed': seed, 'stage': 'final_B', 'method': name, 'input': row['input'], 'power': 1.,
                'init': row['init'], 'policy': row['policy'], 'correct': int(np.sum(pred == LABELS[query])),
                'accuracy_percent': 100*float(np.mean(pred == LABELS[query])), 'paper_percent': paper,
                'model_file': mf.relative_to(BASE).as_posix(), 'prediction_file': pf.relative_to(BASE).as_posix(), 'reused': False})
        print(f'Outer score seed{seed}: all4 shared methods', flush=True)
    assert len(rows) == 80
    table(ROOT/'evaluation/results.csv', rows)
    dump(ROOT/'execution.json', {'new_outer_fits': 40, 'historical_baseline_scores_reused': 40,
        'score_rows': 80, 'elapsed_seconds': time.perf_counter()-started, 'BLAS_threads': 2})


def verify():
    E.check_selections(); rows = csv_rows(ROOT/'evaluation/results.csv'); assert len(rows) == 80
    data = data_arrays(); checked = records = 0; error = residual = 0.
    for row in rows:
        with np.load(BASE/row['model_file']) as m, np.load(BASE/row['prediction_file']) as pred:
            if row['reused'] == 'True':
                a, b, learn, query = m['gallery'], m['probe'], m['train'], m['query']
            else:
                learn, query = E.split(LABELS, 2, int(row['seed']))
                np.testing.assert_array_equal(m['learn'], learn); np.testing.assert_array_equal(m['query'], query)
                g, q, e, r = E.check_model(m, data[row['input']], learn, query, 1., row['init'], 'pca_basis' in m)
                error, residual = max(error, e), max(residual, r); checked += 1
                policy = next(p for p in POLICIES if p['id'] == row['policy'])
                a, b = E.outputs(g, q, m['lda_basis'] if 'lda_basis' in m else None, policy)
            metric = next(p['metric'] for p in POLICIES if p['id'] == row['policy'])
            if metric == 'cosine':
                near = np.argmax((b/np.linalg.norm(b, axis=1, keepdims=True))@(a/np.linalg.norm(a, axis=1, keepdims=True)).T, axis=1)
            else:
                near = np.argmin(np.sum(b*b, axis=1)[:, None]+np.sum(a*a, axis=1)[None, :]-2*b@a.T, axis=1)
            prediction = LABELS[learn[near]]
            np.testing.assert_array_equal(pred['query'], query)
            np.testing.assert_array_equal(pred['predictions'], prediction); np.testing.assert_array_equal(pred['neighbor_index'], learn[near])
            assert int(np.sum(prediction == LABELS[query])) == int(row['correct'])
            assert abs(100*np.mean(prediction == LABELS[query])-float(row['accuracy_percent'])) < 1e-10
            records += len(query)
    assert checked == 40 and records == 25600
    dump(ROOT/'verification.json', {'passed': True, 'new_models_checked': 40,
        'new_CSA_independent_refits': 20, 'new_PCA_eigen_checks': 20,
        'historical_baseline_NN_replayed': 40, 'prediction_records_checked': records,
        'max_CSA_projector_abs_error': error, 'max_Fisher_relative_residual': residual,
        'all_source_input_selection_hashes_unchanged': True})


def analyze():
    E.check_selections(); assert read(ROOT/'verification.json')['passed']
    rows = csv_rows(ROOT/'evaluation/results.csv'); summary = []
    for name, _, _, _, paper in CASES:
        baseline = {int(r['seed']): float(r['accuracy_percent']) for r in rows if r['method'] == name and r['stage'] == 'baseline'}
        for stage in ['baseline', 'final_B']:
            group = sorted([r for r in rows if r['method'] == name and r['stage'] == stage], key=lambda r: int(r['seed']))
            values = np.array([float(r['accuracy_percent']) for r in group]); delta = values-np.array([baseline[s] for s in SEEDS])
            summary.append({'method': name, 'stage': stage, 'mean_percent': float(values.mean()), 'std_pp': float(values.std(ddof=1)),
                'min_percent': float(values.min()), 'max_percent': float(values.max()), 'baseline_change_pp': float(delta.mean()),
                'wins': int(np.sum(delta > 0)), 'ties': int(np.sum(delta == 0)), 'losses': int(np.sum(delta < 0)),
                'paper_percent': paper, 'paper_difference_pp': None if paper is None else float(values.mean()-paper),
                'paper_threshold_count': None if paper is None else int(np.sum(values >= paper))})
    table(ROOT/'comparison.csv', summary); dump(ROOT/'summary.json', summary)
    chosen = [read(ROOT/'selection'/f'seed{s}'/'choices.json')['choices']['final_B'] for s in SEEDS]
    counts = {k: {str(v): sum(r[k] == v for r in chosen) for v in sorted({r[k] for r in chosen})} for k in ['input', 'init', 'policy']}
    dump(ROOT/'final_setting_counts.json', counts)
    history = next(r for r in csv_rows(BASE/'results/orl_protocol_audit/shared_comparison.csv')
                   if r['dataset'] == 'ORL' and r['profile'] == 'paper_dimensions' and r['method'] == 'CSA-3+LDA')
    dump(ROOT/'historical_shared_reference.json', {'row': history,
        'source_sha256': file_hash(BASE/'results/orl_protocol_audit/shared_comparison.csv'),
        'comparison_limit': 'historic shared79.66 uses another input configuration; paired gain calculated against81.25 baseline only'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, names in zip(axes, [[c[0] for c in CASES[:2]], [c[0] for c in CASES[2:]]]):
        for name in names:
            group = [r for r in summary if r['method'] == name]
            artist = ax.errorbar([0, 1], [r['mean_percent'] for r in group], yerr=[r['std_pp'] for r in group], marker='o', capsize=3, label=name)
            paper = group[0]['paper_percent']
            if paper is not None:
                ax.axhline(paper, linestyle='--', alpha=.5, color=artist[0].get_color(), label=f'paper best {paper}%')
        ax.set(xticks=[0, 1], xticklabels=['Fixed crop baseline', 'Gallery-selected crop'], ylabel='Accuracy (%)')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('ORL crop-only, original dimensions, G2/P8, Tmax=3\n10matched splits, mean ± sample SD; historical probes already observed')
    fig.tight_layout(); fig.savefig(ROOT/'comparison.png', dpi=180); plt.close(fig)
    print({'summary': summary, 'selected_counts': counts}, flush=True)


def configure(root):
    global ROOT
    ROOT = root; E.ROOT = root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['prepare', 'select', 'verify_selection', 'evaluate', 'verify', 'analyze'])
    parser.add_argument('--output', type=Path, default=Path('results/orl_ta_crop_optimized'))
    args = parser.parse_args(); output = (BASE/args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    assert output.is_relative_to(BASE), 'Output must stay within project'
    configure(output)
    with threadpool_limits(limits=2):
        globals()[args.stage]()
