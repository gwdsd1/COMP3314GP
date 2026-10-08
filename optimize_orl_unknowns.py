"""Bounded gallery-only ORL protocol search; original Table II dimensions.

12 prespecified inputs -> best2 x3 powers -> best2 x3 powers x2 starts
x6 shared NN/LDA policies. All10 selections freeze before any outer scoring.
This is a pruned search, not a proof of a global optimum or author settings.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.linalg import eigh
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA, PCA, LDA
from gabor import file_hash
from audit_orl_protocol import folds
from run_available_reproduction import split
from run_orl_fixed_inputs import raw_inputs, cv2, alignment_matrix, energy, CASES, LABELS, SEEDS

BASE = Path(__file__).resolve().parent
ROOT = BASE/'results/orl_unknown_parameters'
POWERS = [1., .75, .5]
STARTS = ['identity', 'hosvd']
POLICIES = [{'id': f'{scale}_{origin}_{metric}', 'scale': scale, 'origin': origin, 'metric': metric}
            for scale in ('unit', 'sw')
            for origin, metric in [('gallery', 'cosine'), ('linear', 'cosine'), ('gallery', 'euclidean')]]
DEFAULT_POLICY = POLICIES[0]['id']
SOURCES = ['optimize_orl_unknowns.py', 'csa.py', 'gabor.py', 'audit_orl_protocol.py',
           'run_available_reproduction.py', 'run_orl_fixed_inputs.py', 'prepare_available_data.py']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def csv_rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def inputs():
    common = {'wavelength': 4., 'sigma': float(2*np.pi), 'support': 19, 'finite_dc': True,
              'distance': 20., 'eye_y': 24., 'histogram': 'PIL', 'context': True, 'downsample_first': False}
    variants = [('legacy_crop', {'context': False}), ('context_anchor', {}),
                ('context_w8', {'wavelength': 8.}), ('context_sigma_pi', {'sigma': float(np.pi)}),
                ('context_w8_sigma_pi', {'wavelength': 8., 'sigma': float(np.pi)}),
                ('context_analytic_dc', {'finite_dc': False}),
                ('resize_first_w4', {'context': False, 'downsample_first': True}),
                ('resize_first_w2', {'context': False, 'downsample_first': True, 'wavelength': 2.}),
                ('context_d18_y25', {'distance': 18., 'eye_y': 25.}),
                ('context_d22_y22', {'distance': 22., 'eye_y': 22.}),
                ('context_adaptive', {'support': None}), ('context_cdfmin', {'histogram': 'CDF_min'})]
    return [{'id': n, **common, **overrides} for n, overrides in variants]


def kernel(spec, scale, orientation):
    wavelength = spec['wavelength']*np.sqrt(2)**scale
    k, sigma = 2*np.pi/wavelength, spec['sigma']
    r = spec['support']
    if r is None:
        r = int(np.ceil(np.round(3*sigma/k, 12)))
    y, x = np.mgrid[-r:r+1, -r:r+1]
    a = (k*k/sigma**2)*np.exp(-k*k*(x*x+y*y)/(2*sigma*sigma))*(
        np.exp(1j*k*(x*np.cos(orientation*np.pi/8)+y*np.sin(orientation*np.pi/8)))-np.exp(-sigma*sigma/2))
    if spec['finite_dc']:
        a -= a.mean()
    return a.astype(np.complex64)


def make_gray(raw, eyes, spec):
    kernels = [kernel(spec, s, 0) for s in range(5)]
    margin = max(k.shape[0]//2 for k in kernels) if spec['context'] else 0
    target = np.array([[23-spec['distance']/2, spec['eye_y']], [23+spec['distance']/2, spec['eye_y']]])
    result, centers = [], []
    for image, eye in zip(raw, eyes):
        matrix = alignment_matrix(eye, target)
        central = cv2.warpAffine(image, matrix, (46, 56), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        shifted = matrix.copy(); shifted[:, 2] += margin
        canvas = cv2.warpAffine(image, shifted, (46+2*margin, 56+2*margin),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        canvas[margin:margin+56, margin:margin+46] = central
        if spec['histogram'] == 'PIL':
            mask = np.zeros_like(canvas); mask[margin:margin+56, margin:margin+46] = 255
            equalized = np.asarray(ImageOps.equalize(Image.fromarray(canvas), Image.fromarray(mask)), dtype=np.float32)
        else:
            counts = np.bincount(central.ravel(), minlength=256); cdf = counts.cumsum()
            minimum = cdf[np.flatnonzero(counts)[0]]
            lut = np.rint(np.clip((cdf-minimum)*255/max(1, central.size-minimum), 0, 255)).astype(np.uint8)
            equalized = lut[canvas].astype(np.float32)
        center = equalized[margin:margin+56, margin:margin+46]
        centers.append(center)
        if spec['downsample_first']:
            equalized = cv2.resize(center, (23, 28), interpolation=cv2.INTER_AREA)
        result.append(equalized)
    return np.stack(result), np.stack(centers), margin


def feature_array(grays, spec, margin):
    result = np.empty((len(grays), 28, 23, 40), dtype=np.float32)
    h, w = grays.shape[1:]
    stride = 1 if spec['downsample_first'] else 2
    ch, cw = (28, 23) if stride == 1 else (56, 46)
    for s in range(5):
        ks = [kernel(spec, s, o) for o in range(8)]
        r = ks[0].shape[0]//2
        shape = (next_fast_len(h+4*r), next_fast_len(w+4*r))
        spectra = [fftn(k, s=shape) for k in ks]
        for start in range(0, len(grays), 32):
            padded = np.pad(grays[start:start+32], ((0, 0), (r, r), (r, r)))
            spectrum = fftn(padded, s=shape, axes=(-2, -1))
            y0, x0 = 2*r+margin, 2*r+margin
            for o, k in enumerate(spectra):
                response = np.abs(ifftn(spectrum*k, axes=(-2, -1))[:, y0:y0+ch, x0:x0+cw])
                result[start:start+len(padded), :, :, s*8+o] = response[:, ::stride, ::stride]
    return result


def direct_features(gray, spec, margin):
    # Independent float64 spatial reference and explicit crop/downsample.
    result = np.empty((28, 23, 40))
    stride = 1 if spec['downsample_first'] else 2
    h, w = (28, 23) if stride == 1 else (56, 46)
    for s in range(5):
        for o in range(8):
            k = kernel(spec, s, o).astype(np.complex128); r = k.shape[0]//2
            z = np.abs(convolve2d(np.pad(gray.astype(float), r), k, mode='valid'))
            result[:, :, s*8+o] = z[margin:margin+h:stride, margin:margin+w:stride]
    return result


def prepare():
    ROOT.mkdir(exist_ok=False)
    for name in ('features', 'cv', 'selection', 'evaluation', 'source_snapshot'):
        (ROOT/name).mkdir()
    assert read(BASE/'results/orl_fixed_input_extensions/verification.json')['passed']
    specs = inputs()
    protocol = {'seeds': SEEDS, 'outer': '40classes; exactly2 gallery and8 probe/class',
                'inner': '16folds,75learning/5validation; all40classes; own80gallery each held out once',
                'inputs': specs, 'powers': POWERS, 'starts': STARTS, 'policies': POLICIES,
                'target': 'ordinary CSA4x2x3 + LDA23 only; PCA/CSA390 controls share chosen input/power/NN',
                'CSA_center': False, 'Tmax': 3, 'LDA_alpha': 0.,
                'dimensions': 'CSA24->23, CSA390, PCA24->23(equal budget), PCA33->25(paper)',
                'stage_A': '12 inputs, identity,power1,unit/gallery/cosine; keep2 by complete inner CV',
                'stage_B': 'best2 inputs x3 powers, same identity and policy',
                'stage_C': 'best2 inputs x3 powers x2 starts x6 policies; best final72 configurations',
                'tie_rule': 'correct count, worst fold correct count, context_anchor priority, input order, power1/.75/.5, identity, policy order',
                'report': 'stage0 frozen context plus A/B/C; all candidate scores and negative stages retained',
                'stop': 'complete all10 seeds and3 stages; never add candidates from probe scores',
                'limits': ['pruning can omit a global best input', 'historical probes seen; not blind',
                           'means and paper best results differ', 'author numeric settings not confirmed',
                           'final parameters vary with gallery; no pooling other seeds gallery to choose own parameters'],
                'references': ['https://cenl.ucsd.edu/pubs/ieee-1997.pdf', 'https://www.kolda.net/publication/TensorReview.pdf'],
                'raw_manifest_sha256': file_hash(BASE/'results/orl_fixed_input_extensions/raw_manifest.csv'),
                'eyes_sha256': file_hash(BASE/'results/orl_protocol_audit/yunet_eye_coordinates.csv'),
                'source_sha256': {n: file_hash(BASE/n) for n in SOURCES}}
    dump(ROOT/'protocol.json', protocol)
    for n in SOURCES:
        shutil.copy2(BASE/n, ROOT/'source_snapshot'/n)
    raw, eyes = raw_inputs(); entries, errors = [], []
    anchor_manifest = read(BASE/'results/orl_fixed_input_extensions/feature_manifest.json')
    for spec in specs:
        gray, center, margin = make_gray(raw, eyes, spec)
        if spec['id'] in ('legacy_crop', 'context_anchor'):
            anchor = anchor_manifest['baseline' if spec['id'] == 'legacy_crop' else 'crop_context']
            path = BASE/anchor['path']; assert file_hash(path) == anchor['sha256']
            feature = np.load(path, mmap_mode='r')
            regenerated = feature_array(gray[:2], spec, margin)
            np.testing.assert_array_equal(regenerated, feature[:2])
        else:
            path = ROOT/'features'/f"{spec['id']}.npy"
            feature = feature_array(gray, spec, margin)
            np.save(path, feature)
        assert feature.shape == (400, 28, 23, 40) and np.isfinite(feature).all()
        reference = direct_features(gray[0], spec, margin)
        error = float(np.max(np.abs(reference-feature[0])))
        assert error < 3e-4, (spec['id'], error)
        errors.append({'input': spec['id'], 'direct40channel_max_abs_error': error})
        np.save(ROOT/'features'/f"{spec['id']}_gray_center.npy", center)
        entries.append({'id': spec['id'], 'path': str(path.relative_to(BASE)).replace('\\', '/'),
                        'sha256': file_hash(path), 'margin': margin, 'spec': spec})
        print(f"Prepared {spec['id']}: margin={margin}; spatial reference error={error:.3g}", flush=True)
    dump(ROOT/'feature_manifest.json', entries)
    dump(ROOT/'feature_verification.json', {'passed': True, 'channel_checks': 480,
                                            'anchor_first2_exact': True, 'checks': errors})
    dump(ROOT/'input_freeze.json', {n: file_hash(ROOT/n) for n in ('protocol.json', 'feature_manifest.json')})


def check_inputs():
    for n, h in read(ROOT/'input_freeze.json').items():
        assert file_hash(ROOT/n) == h
    protocol = read(ROOT/'protocol.json')
    for n, h in protocol['source_sha256'].items():
        assert file_hash(BASE/n) == h, n
    for entry in read(ROOT/'feature_manifest.json'):
        assert file_hash(BASE/entry['path']) == entry['sha256']
    assert file_hash(BASE/'results/orl_protocol_audit/yunet_eye_coordinates.csv') == protocol['eyes_sha256']


def initial_bases(x, ranks):
    result = []
    for k, r in enumerate(ranks):
        z = np.moveaxis(x, k+1, 1).reshape(len(x), x.shape[k+1], -1)
        _, u = eigh(np.einsum('nac,nbc->ab', z, z, optimize=True))
        result.append(u[:, -r:][:, ::-1])
    return result


def key(input_id, power, init):
    return f'{input_id}_p{int(round(power*100)):03d}_{init}'


def tensors(data, indices, power):
    return np.asarray(data[indices], dtype=np.float32)**power if power != 1 else np.asarray(data[indices])


def fit_csa(x, ranks, init):
    bases = initial_bases(x.astype(float), ranks) if init == 'hosvd' else None
    return CSA(ranks, center=False, init='identity', initial_bases=bases, max_iter=3).fit(x)


def cv_models(seed, input_id, power, init, data):
    folder = ROOT/'cv'/f'seed{seed}'/key(input_id, power, init)
    folder.mkdir(parents=True, exist_ok=True)
    train, _ = split(LABELS, 2, seed)
    paths = []
    for fi, (learn, query) in enumerate(folds(train, seed)):
        path = folder/f'fold{fi:02d}.npz'; paths.append(path)
        if path.exists():
            continue
        x, y = tensors(data, learn, power), tensors(data, query, power)
        model = fit_csa(x, (4, 2, 3), init)
        g = model.transform(x).reshape(75, 24); q = model.transform(y).reshape(5, 24)
        lda = LDA().fit(g, LABELS[learn])
        if lda.max_components < 23:
            raise ValueError('LDA rank below fixed23; invalid candidate')
        np.savez_compressed(path, learn=learn, query=query, gallery_core=g, query_core=q,
                            axis0=model.bases[0], axis1=model.bases[1], axis2=model.bases[2],
                            lda_mean=lda.mean, lda_basis=lda.basis[:, :23], sw=lda.sw, sb=lda.sb,
                            eigenvalues=lda.eigenvalues[:23])
    return paths


def outputs(g, q, lda_basis, policy):
    mean = g.mean(0) if policy['origin'] == 'gallery' else np.zeros(g.shape[1])
    a, b = g-mean, q-mean
    if lda_basis is not None:
        w = lda_basis.copy()
        if policy['scale'] == 'unit':
            w /= np.linalg.norm(w, axis=0)
        a, b = a@w, b@w
    return a, b


def score_cv(paths, policy, independent=False):
    correct, fold_counts, predictions, neighbors = 0, [], [], []
    for path in paths:
        m = np.load(path)
        a, b = outputs(m['gallery_core'], m['query_core'], m['lda_basis'], policy)
        if independent and policy['metric'] == 'cosine':
            near = np.argmax((b/np.linalg.norm(b, axis=1, keepdims=True))@(a/np.linalg.norm(a, axis=1, keepdims=True)).T, axis=1)
        elif independent:
            distances = np.sum(b*b, axis=1)[:, None]+np.sum(a*a, axis=1)[None, :]-2*b@a.T
            near = np.argmin(distances, axis=1)
        else:
            distances = cdist(b, a, policy['metric']); assert np.isfinite(distances).all()
            near = np.argmin(distances, axis=1)
        pred = LABELS[m['learn'][near]]
        n = int(np.sum(pred == LABELS[m['query']]))
        correct += n; fold_counts.append(n)
        predictions.extend(pred.tolist()); neighbors.extend(m['learn'][near].tolist())
    return correct, min(fold_counts), predictions, neighbors


def candidate(seed, input_id, power, init, policy, data):
    paths = cv_models(seed, input_id, power, init, data)
    count, worst, pred, near = score_cv(paths, policy)
    return {'input': input_id, 'power': power, 'init': init, 'policy': policy['id'],
            'inner_correct': count, 'inner_total': 80, 'worst_fold_correct': worst,
            'predictions': pred, 'neighbors': near}


def rank(row):
    priority = ['context_anchor']+[s['id'] for s in inputs() if s['id'] != 'context_anchor']
    return (row['inner_correct'], row['worst_fold_correct'], -priority.index(row['input']),
            -POWERS.index(row['power']), -STARTS.index(row['init']),
            -[p['id'] for p in POLICIES].index(row['policy']))


def select():
    check_inputs()
    assert not (ROOT/'selection_freeze.json').exists()
    entries = read(ROOT/'feature_manifest.json')
    data = {e['id']: np.load(BASE/e['path'], mmap_mode='r') for e in entries}
    execution = []; started = time.perf_counter()
    for seed in SEEDS:
        folder = ROOT/'selection'/f'seed{seed}'; folder.mkdir(exist_ok=False)
        phase_a = []
        for spec in inputs():
            row = candidate(seed, spec['id'], 1., 'identity', POLICIES[0], data[spec['id']])
            phase_a.append(row)
            print(f"seed{seed} A {spec['id']}: {row['inner_correct']}/80", flush=True)
        top2 = sorted(phase_a, key=rank, reverse=True)[:2]
        phase_b = [candidate(seed, r['input'], p, 'identity', POLICIES[0], data[r['input']]) for r in top2 for p in POWERS]
        print(f'seed{seed} B complete', flush=True)
        phase_c = []
        for r in top2:
            for p in POWERS:
                for init in STARTS:
                    paths = cv_models(seed, r['input'], p, init, data[r['input']])
                    for policy in POLICIES:
                        n, worst, pred, near = score_cv(paths, policy)
                        phase_c.append({'input': r['input'], 'power': p, 'init': init, 'policy': policy['id'],
                                        'inner_correct': n, 'inner_total': 80, 'worst_fold_correct': worst,
                                        'predictions': pred, 'neighbors': near})
                    print(f"seed{seed} C {r['input']} power{p} {init} complete", flush=True)
        anchor = next(r for r in phase_a if r['input'] == 'context_anchor')
        chosen = {'stage0_context': anchor, 'stage_A': max(phase_a, key=rank),
                  'stage_B': max(phase_b, key=rank), 'stage_C': max(phase_c, key=rank)}
        assert [chosen[n]['inner_correct'] for n in chosen] == sorted(chosen[n]['inner_correct'] for n in chosen)
        dump(folder/'choices.json', {'seed': seed, 'top2': [r['input'] for r in top2], 'choices': chosen})
        for name, rows in [('A', phase_a), ('B', phase_b), ('C', phase_c)]:
            dump(folder/f'candidates_{name}.json', rows)
            table(folder/f'candidates_{name}.csv', [{k: v for k, v in r.items() if k not in ('predictions', 'neighbors')} for r in rows])
        execution.extend({'seed': seed, 'stage': name, **{k: v for k, v in r.items() if k not in ('predictions', 'neighbors')}} for name, r in chosen.items())
        print(f"seed{seed} frozen: inner {anchor['inner_correct']} -> {chosen['stage_A']['inner_correct']} -> {chosen['stage_B']['inner_correct']} -> {chosen['stage_C']['inner_correct']}", flush=True)
    table(ROOT/'selected_settings.csv', execution)
    files = list((ROOT/'selection').rglob('*'))+list((ROOT/'cv').rglob('*.npz'))+[ROOT/'selected_settings.csv']
    dump(ROOT/'selection_freeze.json', {str(p.relative_to(ROOT)).replace('\\', '/'): file_hash(p) for p in files if p.is_file()})
    dump(ROOT/'selection_execution.json', {'elapsed_seconds': time.perf_counter()-started,
        'CV_model_fits': len(list((ROOT/'cv').rglob('*.npz'))), 'candidate_rows': 900,
        'outer_probe_scored': False, 'all10_selections_frozen': True})


def check_selections():
    check_inputs()
    for n, h in read(ROOT/'selection_freeze.json').items():
        assert file_hash(ROOT/n) == h, n


def independent_bases(x, ranks, init):
    x = np.asarray(x, dtype=float)
    if init == 'hosvd':
        u = []
        for k, r in enumerate(ranks):
            zs = [np.moveaxis(a, k, 0).reshape(x.shape[k+1], -1) for a in x]
            covariance = sum(z@z.T for z in zs)
            _, v = eigh((covariance+covariance.T)/2); u.append(v[:, -r:][:, ::-1])
    else:
        u = [np.eye(m, r) for m, r in zip(x.shape[1:], ranks)]
    for _ in range(3):
        for k, r in enumerate(ranks):
            if k == 0:
                z = np.einsum('nhwc,wb,cd->nhbd', x, u[1], u[2], optimize=True).reshape(len(x), x.shape[1], -1)
            elif k == 1:
                z = np.einsum('nhwc,ha,cd->nwad', x, u[0], u[2], optimize=True).reshape(len(x), x.shape[2], -1)
            else:
                z = np.einsum('nhwc,ha,wb->ncab', x, u[0], u[1], optimize=True).reshape(len(x), x.shape[3], -1)
            covariance = sum(a@a.T for a in z)
            _, v = eigh((covariance+covariance.T)/2); u[k] = v[:, -r:][:, ::-1]
    return u


def check_model(m, source, learn, query, power, init, pca=False, refit=True):
    x, y = tensors(source, learn, power), tensors(source, query, power)
    if pca:
        w = m['pca_basis']; mean = x.astype(float).reshape(len(x), -1).mean(0)
        np.testing.assert_allclose(w.T@w, np.eye(w.shape[1]), atol=1e-9)
        centered = x.astype(float).reshape(len(x), -1)-mean
        ev = eigh(centered@centered.T, eigvals_only=True)[::-1][:w.shape[1]]
        residual = np.linalg.norm(centered.T@(centered@w)-w*ev)/max(1, np.linalg.norm(w*ev))
        assert residual < 1e-8
        g, q = centered@w, (y.astype(float).reshape(len(y), -1)-mean)@w
        update_error = 0.
    else:
        bases = [m[f'axis{k}'] for k in range(3)]; update_error = 0.
        for u in bases:
            np.testing.assert_allclose(u.T@u, np.eye(u.shape[1]), atol=1e-10)
        if refit:
            independent = independent_bases(x, [u.shape[1] for u in bases], init)
            update_error = max(float(np.max(np.abs(a@a.T-b@b.T))) for a, b in zip(bases, independent))
            assert update_error < 2e-7, update_error
        g = np.einsum('nhwc,ha,wb,cd->nabd', x.astype(float), *bases, optimize=True).reshape(len(x), -1)
        q = np.einsum('nhwc,ha,wb,cd->nabd', y.astype(float), *bases, optimize=True).reshape(len(y), -1)
    np.testing.assert_allclose(g, m['gallery_core'], atol=3e-8, rtol=1e-9)
    np.testing.assert_allclose(q, m['query_core'], atol=3e-8, rtol=1e-9)
    residual = 0.
    if 'lda_basis' in m:
        sw, sb, mean = np.zeros_like(m['sw']), np.zeros_like(m['sb']), g.mean(0)
        for c in np.unique(LABELS[learn]):
            a = g[LABELS[learn] == c]; delta = a-a.mean(0); sw += delta.T@delta
            d = a.mean(0)-mean; sb += len(a)*np.outer(d, d)
        np.testing.assert_allclose(sw, m['sw'], atol=1e-7, rtol=1e-8)
        np.testing.assert_allclose(sb, m['sb'], atol=1e-7, rtol=1e-8)
        w = m['lda_basis']; ev = m['eigenvalues']
        eig = eigh(sb, sw, eigvals_only=True)[::-1][:w.shape[1]]
        np.testing.assert_allclose(eig, ev, atol=1e-7, rtol=1e-7)
        residual = float(np.linalg.norm(sb@w-(sw@w)*ev)/max(1, np.linalg.norm(sb@w)))
        assert residual < 1e-8
    return g, q, update_error, residual


def verify_selection():
    check_selections()
    data = {e['id']: np.load(BASE/e['path'], mmap_mode='r') for e in read(ROOT/'feature_manifest.json')}
    prediction_records, checked = 0, set()
    update_error, fisher_error = 0., 0.
    for seed in SEEDS:
        folder = ROOT/'selection'/f'seed{seed}'
        choices = read(folder/'choices.json')['choices']
        for stage in ('A', 'B', 'C'):
            for row in read(folder/f'candidates_{stage}.json'):
                paths = [ROOT/'cv'/f'seed{seed}'/key(row['input'], row['power'], row['init'])/f'fold{i:02d}.npz' for i in range(16)]
                policy = next(p for p in POLICIES if p['id'] == row['policy'])
                count, worst, pred, near = score_cv(paths, policy, independent=True)
                assert count == row['inner_correct'] and worst == row['worst_fold_correct']
                assert pred == row['predictions'] and near == row['neighbors']
                prediction_records += 80
            pool = read(folder/f'candidates_{stage}.json')
            assert max(pool, key=rank) == choices['stage_'+stage]
        train, probe = split(LABELS, 2, seed)
        for row in choices.values():
            for i, (learn, query) in enumerate(folds(train, seed)):
                path = ROOT/'cv'/f'seed{seed}'/key(row['input'], row['power'], row['init'])/f'fold{i:02d}.npz'
                if path in checked:
                    continue
                m = np.load(path)
                np.testing.assert_array_equal(m['learn'], learn); np.testing.assert_array_equal(m['query'], query)
                assert not np.intersect1d(learn, probe).size and not np.intersect1d(query, probe).size
                _, _, error, residual = check_model(m, data[row['input']], learn, query, row['power'], row['init'])
                update_error = max(update_error, error); fisher_error = max(fisher_error, residual); checked.add(path)
        print(f'Independent candidate/selected-CV checks passed seed{seed}', flush=True)
    assert prediction_records == 72000
    dump(ROOT/'selection_verification.json', {'passed': True, 'candidate_rows_checked': 900,
         'validation_prediction_records_checked': prediction_records, 'selected_CV_models_refitted': len(checked),
         'max_CSA_projector_abs_error': update_error, 'max_Fisher_relative_residual': fisher_error,
         'own_gallery_only_checked': True, 'frozen_hashes_unchanged': True})


def evaluate():
    check_selections(); assert read(ROOT/'selection_verification.json')['passed']
    assert not (ROOT/'evaluation/results.csv').exists()
    (ROOT/'evaluation/models').mkdir(); (ROOT/'evaluation/predictions').mkdir()
    data = {e['id']: np.load(BASE/e['path'], mmap_mode='r') for e in read(ROOT/'feature_manifest.json')}
    rows, statistics = [], []; started = time.perf_counter()
    old = csv_rows(BASE/'results/orl_fixed_input_extensions/results.csv')
    for r in old:
        if r['variant'] not in ('baseline', 'crop_context'):
            continue
        rows.append({'seed': int(r['seed']), 'stage': 'legacy_baseline' if r['variant'] == 'baseline' else 'stage0_context',
                     'method': r['method'], 'input': 'legacy_crop' if r['variant'] == 'baseline' else 'context_anchor',
                     'power': 1., 'init': 'identity', 'policy': DEFAULT_POLICY,
                     'correct': int(r['correct']), 'accuracy_percent': float(r['accuracy_percent']),
                     'paper_percent': float(r['paper_percent']) if r['paper_percent'] else None,
                     'model_file': str((Path('results/orl_fixed_input_extensions')/r['model_file'])).replace('\\', '/'),
                     'prediction_file': str((Path('results/orl_fixed_input_extensions')/r['prediction_file'])).replace('\\', '/'),
                     'reused': True})
    for seed in SEEDS:
        choices = read(ROOT/'selection'/f'seed{seed}'/'choices.json')['choices']
        train, query = split(LABELS, 2, seed)
        for stage in ('stage_A', 'stage_B', 'stage_C'):
            row = choices[stage]; source = data[row['input']]
            x, y = tensors(source, train, row['power']), tensors(source, query, row['power'])
            e0 = energy(x, LABELS[train]); policy = next(p for p in POLICIES if p['id'] == row['policy'])
            for name, alg, dims, ld, paper in CASES:
                model = PCA(dims, center=True, solver='gram').fit(x) if alg == 'pca' else fit_csa(x, dims, row['init'])
                g = model.transform(x).reshape(80, -1); q = model.transform(y).reshape(320, -1)
                arrays = {'learn': train, 'query': query, 'gallery_core': g, 'query_core': q, 'mean': model.mean}
                if alg == 'pca':
                    arrays['pca_basis'] = model.basis
                else:
                    arrays.update({f'axis{k}': u for k, u in enumerate(model.bases)})
                lda_basis, condition = None, None
                if ld is not None:
                    lda = LDA().fit(g, LABELS[train])
                    assert ld <= lda.max_components, 'Fixed LDA dimension must be retained'
                    lda_basis = lda.basis[:, :ld]
                    arrays.update(lda_basis=lda_basis, sw=lda.sw, sb=lda.sb, lda_mean=lda.mean,
                                  eigenvalues=lda.eigenvalues[:ld])
                    condition = float(np.linalg.cond(lda.sw))
                a, b = outputs(g, q, lda_basis, policy)
                distances = cdist(b, a, policy['metric']); assert np.isfinite(distances).all()
                nearest = np.argmin(distances, axis=1); pred = LABELS[train[nearest]]
                stem = f'{stage}_seed{seed}_{name}'
                np.savez_compressed(ROOT/'evaluation/models'/f'{stem}.npz', **arrays)
                np.savez_compressed(ROOT/'evaluation/predictions'/f'{stem}.npz', predictions=pred, neighbor_index=train[nearest], query=query, learn=train)
                e1 = energy(g, LABELS[train])
                statistics.append({'seed': seed, 'stage': stage, 'method': name, 'total_variation_retained': e1[0]/e0[0],
                                   'between_variation_retained': e1[1]/e0[1], 'sw_condition': condition})
                rows.append({'seed': seed, 'stage': stage, 'method': name, 'input': row['input'], 'power': row['power'],
                             'init': row['init'], 'policy': row['policy'], 'correct': int(np.sum(pred == LABELS[query])),
                             'accuracy_percent': 100*float(np.mean(pred == LABELS[query])), 'paper_percent': paper,
                             'model_file': str((ROOT/'evaluation/models'/f'{stem}.npz').relative_to(BASE)).replace('\\', '/'),
                             'prediction_file': str((ROOT/'evaluation/predictions'/f'{stem}.npz').relative_to(BASE)).replace('\\', '/'), 'reused': False})
            print(f'Outer score seed{seed} {stage}: all4 shared controls', flush=True)
    table(ROOT/'evaluation/results.csv', rows); table(ROOT/'representation_statistics.csv', statistics)
    dump(ROOT/'execution.json', {'new_outer_fits': 120, 'historical_scores_reused': 80, 'score_rows': len(rows),
                                'elapsed_seconds': time.perf_counter()-started, 'BLAS_threads': 2})


def verify():
    check_selections()
    rows = csv_rows(ROOT/'evaluation/results.csv'); assert len(rows) == 200
    data = {e['id']: np.load(BASE/e['path'], mmap_mode='r') for e in read(ROOT/'feature_manifest.json')}
    count, checked, update_error, fisher_error = 0, 0, 0., 0.
    for row in rows:
        if row['reused'] == 'True':
            # Reused scores were already fully checked; replay saved NN here.
            m = np.load(BASE/row['model_file']); pred = np.load(BASE/row['prediction_file'])
            a, b, learn, query = m['gallery'], m['probe'], m['train'], m['query']
        else:
            m = np.load(BASE/row['model_file']); pred = np.load(BASE/row['prediction_file'])
            learn, query = split(LABELS, 2, int(row['seed']))
            np.testing.assert_array_equal(learn, m['learn']); np.testing.assert_array_equal(query, m['query'])
            g, q, error, residual = check_model(m, data[row['input']], learn, query, float(row['power']), row['init'], 'pca_basis' in m)
            update_error = max(update_error, error); fisher_error = max(fisher_error, residual)
            policy = next(p for p in POLICIES if p['id'] == row['policy'])
            a, b = outputs(g, q, m['lda_basis'] if 'lda_basis' in m else None, policy)
            checked += 1
        metric = next(p['metric'] for p in POLICIES if p['id'] == row['policy'])
        if metric == 'cosine':
            nearest = np.argmax((b/np.linalg.norm(b, axis=1, keepdims=True))@(a/np.linalg.norm(a, axis=1, keepdims=True)).T, axis=1)
        else:
            nearest = np.argmin(np.sum(b*b, axis=1)[:, None]+np.sum(a*a, axis=1)[None, :]-2*b@a.T, axis=1)
        prediction = LABELS[learn[nearest]]
        np.testing.assert_array_equal(pred['predictions'], prediction); np.testing.assert_array_equal(pred['neighbor_index'], learn[nearest])
        assert int(np.sum(prediction == LABELS[query])) == int(row['correct'])
        assert abs(100*np.mean(prediction == LABELS[query])-float(row['accuracy_percent'])) < 1e-10
        count += len(query)
    assert checked == 120 and count == 64000
    dump(ROOT/'verification.json', {'passed': True, 'new_models_checked': checked, 'new_CSA_independent_refits': 60,
         'new_PCA_eigen_checks': 60, 'reused_models_NN_replayed': 80, 'prediction_records_checked': count,
         'max_CSA_projector_abs_error': update_error, 'max_Fisher_relative_residual': fisher_error,
         'all_source_inputs_selection_hashes_unchanged': True})


def analyze():
    check_selections(); assert read(ROOT/'verification.json')['passed']
    rows = csv_rows(ROOT/'evaluation/results.csv'); summary = []
    stages = ['legacy_baseline', 'stage0_context', 'stage_A', 'stage_B', 'stage_C']
    for name, _, _, _, paper in CASES:
        baseline = {int(r['seed']): float(r['accuracy_percent']) for r in rows if r['stage'] == 'legacy_baseline' and r['method'] == name}
        context = {int(r['seed']): float(r['accuracy_percent']) for r in rows if r['stage'] == 'stage0_context' and r['method'] == name}
        for stage in stages:
            group = sorted([r for r in rows if r['stage'] == stage and r['method'] == name], key=lambda r: int(r['seed']))
            values = np.array([float(r['accuracy_percent']) for r in group])
            delta = values-np.array([baseline[s] for s in SEEDS]); dc = values-np.array([context[s] for s in SEEDS])
            summary.append({'method': name, 'stage': stage, 'mean_percent': float(values.mean()), 'std_pp': float(values.std(ddof=1)),
                            'min_percent': float(values.min()), 'max_percent': float(values.max()),
                            'legacy_change_pp': float(delta.mean()), 'context_change_pp': float(dc.mean()),
                            'context_wins': int(np.sum(dc > 0)), 'context_ties': int(np.sum(dc == 0)), 'context_losses': int(np.sum(dc < 0)),
                            'paper_percent': paper, 'paper_difference_pp': None if paper is None else float(values.mean()-paper)})
    dump(ROOT/'summary.json', summary); table(ROOT/'comparison.csv', summary)
    counts = {}
    for seed in SEEDS:
        row = read(ROOT/'selection'/f'seed{seed}'/'choices.json')['choices']['stage_C']
        for name in ('input', 'power', 'init', 'policy'):
            counts.setdefault(name, {}); counts[name][str(row[name])] = counts[name].get(str(row[name]), 0)+1
    dump(ROOT/'final_setting_counts.json', counts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, names in zip(axes, ([c[0] for c in CASES[:2]], [c[0] for c in CASES[2:]])):
        for j, name in enumerate(names):
            group = [next(r for r in summary if r['method'] == name and r['stage'] == stage) for stage in stages]
            ax.errorbar(range(5), [r['mean_percent'] for r in group], yerr=[r['std_pp'] for r in group], marker='o', capsize=3, label=name)
            paper = next(c[4] for c in CASES if c[0] == name)
            if paper is not None:
                ax.axhline(paper, linestyle='--', alpha=.5, label=f'{name} paper best {paper}')
        ax.set(xticks=range(5), xticklabels=['Legacy', 'Context', 'Input A', 'Power B', 'NN/init C'], ylabel='Accuracy (%)')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('ORL G2/P8: gallery-only bounded iterations; original dimensions\nMean ± sample SD, all10 matched splits; historical probes already observed')
    fig.tight_layout(); fig.savefig(ROOT/'iteration_comparison.png', dpi=180); plt.close(fig)
    print(json.dumps({'summary': summary, 'final_settings_counts': counts}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'select', 'verify_selection', 'evaluate', 'verify', 'analyze'], required=True)
    parser.add_argument('--output', type=Path, default=Path('results/orl_unknown_parameters'))
    args = parser.parse_args()
    ROOT = (BASE/args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    assert ROOT.is_relative_to(BASE), 'Output must stay within project'
    with threadpool_limits(limits=2):
        globals()[args.stage]()
