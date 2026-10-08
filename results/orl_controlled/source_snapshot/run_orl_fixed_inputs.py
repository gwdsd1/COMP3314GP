"""Three frozen ORL inputs, 4 fixed methods, 10 paired splits; no search.

TT+histogram and crop-context convolution are explicitly labelled extensions.
Historical results and the shared paper-dimension table are not overwritten.
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
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA, PCA, LDA
from diagnose_orl_gabor import make_kernel, filtered
from gabor import file_hash
from prepare_available_data import cv2, alignment_matrix
from run_available_reproduction import split, transform_batches

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'results/orl_fixed_input_extensions'
LABELS = np.repeat(np.arange(1, 41), 10)
NAMES = [f's{c}/{i}.pgm' for c in range(1, 41) for i in range(1, 11)]
SEEDS = list(range(20, 30))
VARIANTS = ['baseline', 'tt_hist', 'crop_context']
CASES = [('CSA24_LDA23', 'csa', (4, 2, 3), 23, 90.9),
         ('CSA390', 'csa', (15, 1, 26), None, 94.4),
         ('PCA24_LDA23', 'pca', 24, 23, None),
         ('PCA33_LDA25', 'pca', 33, 25, 87.2)]
SPEC = {'base_wavelength': 4., 'sigma': float(2*np.pi), 'radius': 19,
        'boundary': 'constant', 'finite_zero_mean': True,
        'sampling': 'stride00', 'normalization': 'none', 'stable_radius': True}
SOURCES = ['run_orl_fixed_inputs.py', 'csa.py', 'diagnose_orl_gabor.py',
           'prepare_available_data.py', 'run_available_reproduction.py', 'gabor.py']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_metadata():
    entries = read(BASE/'results/orl_protocol_audit/catalogue.json')
    return next(e for e in entries if e['id'] == 'yunet_d20_replicate_fixed39_zero')


def raw_inputs():
    raw = np.stack([np.asarray(Image.open(BASE/'data/raw/orl'/n).convert('L')) for n in NAMES])
    with (BASE/'results/orl_protocol_audit/yunet_eye_coordinates.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert [r['file'] for r in rows] == NAMES
    eyes = np.array([[[float(r['left_x']), float(r['left_y'])],
                      [float(r['right_x']), float(r['right_y'])]] for r in rows])
    return raw, eyes


def aligned(raw, eyes, margin=0):
    target = np.array([[13., 24.], [33., 24.]])
    images = []
    for image, eye in zip(raw, eyes):
        matrix = alignment_matrix(eye, target)
        central_matrix = matrix.copy()
        matrix[:, 2] += margin
        canvas = cv2.warpAffine(image, matrix, (46+2*margin, 56+2*margin),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if margin:
            # OpenCV fixed-point rounding depends slightly on canvas translation.
            # Preserve the exact legacy central warp; change only outer context.
            canvas[margin:margin+56, margin:margin+46] = cv2.warpAffine(
                image, central_matrix, (46, 56), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)
        images.append(canvas)
    return np.stack(images)


def tt(image):
    # Fixed TT defaults; no parameters are estimated from other images.
    x = (np.asarray(image, dtype=np.float64)/255.)**.2
    x = gaussian_filter(x, 1., mode='nearest', truncate=4.) - gaussian_filter(x, 2., mode='nearest', truncate=4.)
    x /= max(float(np.mean(np.abs(x)**.1)**10), 1e-12)
    x /= max(float(np.mean(np.minimum(10., np.abs(x))**.1)**10), 1e-12)
    x = 10.*np.tanh(x/10.)
    # Explicit extra mapping + histogram step; this is not canonical TT.
    mapped = np.rint(np.clip((x+10.)*255./20., 0, 255)).astype(np.uint8)
    return np.asarray(ImageOps.equalize(Image.fromarray(mapped)), dtype=np.float32)


def gray_input(raw, eyes, variant):
    margin = 19 if variant == 'crop_context' else 0
    warped = aligned(raw, eyes, margin)
    if variant == 'tt_hist':
        return np.stack([tt(x) for x in warped])
    mask = None
    if margin:
        mask = np.zeros(warped.shape[1:], dtype=np.uint8)
        mask[19:75, 19:65] = 255
        mask = Image.fromarray(mask)
    return np.stack([np.asarray(ImageOps.equalize(Image.fromarray(x), mask), dtype=np.float32)
                     for x in warped])


def context_features(images):
    # Magnitude is cropped BEFORE downsampling: margin19 has odd pixel parity.
    result = np.empty((len(images), 28, 23, 40), dtype=np.float32)
    radius, margin = 19, 19
    h, w = images.shape[1:]
    shape = (next_fast_len(h+4*radius), next_fast_len(w+4*radius))
    for s in range(5):
        kernels = [make_kernel(4.*np.sqrt(2)**s, o*np.pi/8, SPEC) for o in range(8)]
        spectra = [fftn(k, s=shape) for k in kernels]
        for start in range(0, len(images), 32):
            padded = np.pad(images[start:start+32], ((0, 0), (radius, radius), (radius, radius)))
            transformed = fftn(padded, s=shape, axes=(-2, -1))
            for o, spectrum in enumerate(spectra):
                y0, x0 = 2*radius+margin, 2*radius+margin
                response = np.abs(ifftn(transformed*spectrum, axes=(-2, -1))[:, y0:y0+56, x0:x0+46])
                result[start:start+len(padded), :, :, s*8+o] = response[:, ::2, ::2]
    return result


def independent_tt(image):
    # Explicit Gaussian kernels + direct spatial convolution independently
    # check SciPy's separable Gaussian filtering and both robust rescalings.
    x = (np.asarray(image, dtype=np.float64)/255.)**.2
    smooth = []
    for sigma in (1., 2.):
        radius = int(4*sigma+.5)
        axis = np.arange(-radius, radius+1)
        k = np.exp(-axis*axis/(2*sigma*sigma)); k /= k.sum()
        smooth.append(convolve2d(np.pad(x, radius, mode='edge'), np.outer(k, k), mode='valid'))
    y = smooth[0]-smooth[1]
    for clipped in (False, True):
        a = np.minimum(np.abs(y), 10.) if clipped else np.abs(y)
        scale = float(np.power(np.mean(np.power(a, .1)), 1/.1))
        y = y/max(scale, 1e-12)
    bounded = 10*np.tanh(y/10)
    mapped = np.rint(np.clip((bounded+10)*12.75, 0, 255)).astype(np.uint8)
    return np.asarray(ImageOps.equalize(Image.fromarray(mapped)), dtype=np.float32)


def feature_checks(grays, paths, raw, eyes):
    central = aligned(raw, eyes)
    extended = aligned(raw, eyes, 19)
    np.testing.assert_array_equal(extended[:, 19:75, 19:65], central)
    np.testing.assert_array_equal(grays['crop_context'][:, 19:75, 19:65], grays['baseline'])
    for i in (0, 197, 399):
        np.testing.assert_array_equal(independent_tt(central[i]), grays['tt_hist'][i])
    errors = []
    for variant in VARIANTS:
        data = np.load(paths[variant], mmap_mode='r')
        for i in (0, 197):
            image = grays[variant][i].astype(np.float64)
            for s in range(5):
                for o in range(8):
                    k = make_kernel(4*np.sqrt(2)**s, o*np.pi/8, SPEC).astype(np.complex128)
                    full = np.abs(convolve2d(np.pad(image, 19), k, mode='valid'))
                    if variant == 'crop_context':
                        full = full[19:75, 19:65]
                    error = float(np.max(np.abs(full[::2, ::2]-data[i, :, :, s*8+o])))
                    assert error < 2e-4, (variant, i, s, o, error)
                    errors.append(error)
    return {'passed': True, 'all400_warp_centers_exact': True,
            'all400_equalized_centers_exact': True, 'independent_TT_images': 3,
            'direct_convolution_channel_checks': len(errors), 'max_abs_error': max(errors)}


def prepare():
    ROOT.mkdir(exist_ok=False)
    for name in ('features', 'models', 'predictions', 'source_snapshot'):
        (ROOT/name).mkdir()
    entry = input_metadata()
    for kind in ('gray', 'gabor'):
        assert file_hash(BASE/entry[kind+'_path']) == entry[kind+'_sha256']
    raw, eyes = raw_inputs()
    table(ROOT/'raw_manifest.csv', [{'index': i, 'file': n, 'sha256': file_hash(BASE/'data/raw/orl'/n)}
                                  for i, n in enumerate(NAMES)])
    hashes = {n: file_hash(BASE/n) for n in SOURCES}
    for n in SOURCES:
        shutil.copy2(BASE/n, ROOT/'source_snapshot'/n)
    protocol = {'purpose': 'Three prespecified shared inputs; no candidate or seed selection',
                'seeds': SEEDS, 'gallery_per_class': 2, 'probe_per_class': 8, 'input_anchor': entry,
                'variants': VARIANTS, 'cases': [{'method': n, 'algorithm': a, 'pre_dimensions': d,
                                                'lda_dimension': l, 'paper_percent': p} for n, a, d, l, p in CASES],
                'CSA': {'center': False, 'init': 'identity', 'Tmax': 3, 'epsilon': .001},
                'PCA': {'center': True, 'solver': 'gram'}, 'LDA': {'alpha': 0, 'scale': 'unit_columns'},
                'NN': {'metric': 'cosine', 'origin': 'own gallery output mean for every method'},
                'Gabor': SPEC, 'TT': {'gamma': .2, 'sigma0': 1., 'sigma1': 2., 'alpha': .1, 'tau': 10.,
                    'boundary': 'nearest', 'truncate': 4., 'input': 'warped grayscale /255 before histogram',
                    'extra_steps': 'map bounded[-10,10] to uint8 with nearest rounding, then PIL histogram equalization'},
                'context': {'margin': 19, 'working_shape': [94, 84], 'warp_boundary': 'replicate',
                    'central_warp': 'overwrite expanded central block with exact legacy56x46 warp to avoid canvas-dependent interpolation rounding',
                    'histogram': 'central56x46 only; apply identical LUT to entire canvas',
                    'convolution_boundary': 'zero outside expanded canvas', 'crop_then_stride': True},
                'diagnostic': {'jitter_seed': 20, 'jitter': 'both source eyes x +/-0.5 pixels; own80 gallery only',
                               'no_extra_fit': True, 'not_used_for_selection': True},
                'statistics': 'mean/sample SD, matched change and win/tie/loss; no significance claim',
                'limits': ['historical probes observed; not blind', 'TT+histogram is a combination extension',
                           'context changes effective field of view', 'PCA24 equal-budget control has no matching paper row',
                           'paper best results and repeated mean are different aggregation', 'same400 images reused'],
                'references': {'TT': 'https://inria.hal.science/inria-00548674/document'},
                'eye_sha256': file_hash(BASE/'results/orl_protocol_audit/yunet_eye_coordinates.csv'),
                'source_sha256': hashes}
    dump(ROOT/'protocol.json', protocol)
    splits = [{'seed': s, 'gallery': split(LABELS, 2, s)[0].tolist(),
               'probe': split(LABELS, 2, s)[1].tolist()} for s in SEEDS]
    dump(ROOT/'splits.json', splits)
    grays = {v: gray_input(raw, eyes, v) for v in VARIANTS}
    np.testing.assert_array_equal(grays['baseline'], np.load(BASE/entry['gray_path']))
    paths = {'baseline': BASE/entry['gabor_path']}
    for variant in ('tt_hist', 'crop_context'):
        np.save(ROOT/'features'/f'{variant}_gray.npy', grays[variant])
        features = context_features(grays[variant]) if variant == 'crop_context' else filtered(grays[variant], SPEC)
        assert features.shape == (400, 28, 23, 40) and np.isfinite(features).all()
        paths[variant] = ROOT/'features'/f'{variant}.npy'
        np.save(paths[variant], features)
        print(f'Prepared {variant}: {features.shape}', flush=True)
    dump(ROOT/'feature_verification.json', feature_checks(grays, paths, raw, eyes))
    manifest = {v: {'path': str(p.relative_to(BASE)).replace('\\', '/'), 'sha256': file_hash(p)} for v, p in paths.items()}
    dump(ROOT/'feature_manifest.json', manifest)
    dump(ROOT/'freeze.json', {'protocol_sha256': file_hash(ROOT/'protocol.json'),
                             'splits_sha256': file_hash(ROOT/'splits.json'),
                             'feature_manifest_sha256': file_hash(ROOT/'feature_manifest.json')})
    print('All inputs and settings frozen before current probe scoring.', flush=True)


def check_freeze():
    protocol = read(ROOT/'protocol.json')
    freeze = read(ROOT/'freeze.json')
    for key, value in freeze.items():
        assert file_hash(ROOT/(key.removesuffix('_sha256')+'.json')) == value
    for n, h in protocol['source_sha256'].items():
        current = file_hash(BASE/n)
        if current != h:
            # A recorded verifier-only repair cannot change fitting or inputs.
            amendment = read(ROOT/'verifier_amendment.json')
            assert n == 'run_orl_fixed_inputs.py'
            assert current == amendment['current_source_sha256']
            assert h == amendment['frozen_run_source_sha256']
            assert file_hash(ROOT/'source_snapshot'/n) == h
        else:
            assert current == h, n
    for item in read(ROOT/'feature_manifest.json').values():
        assert file_hash(BASE/item['path']) == item['sha256']
    assert file_hash(BASE/'results/orl_protocol_audit/yunet_eye_coordinates.csv') == protocol['eye_sha256']
    return protocol


def energy(x, labels):
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    mean = x.mean(0)
    total = float(np.sum((x-mean)**2))
    between = sum(np.sum(labels == c)*float(np.sum((x[labels == c].mean(0)-mean)**2)) for c in np.unique(labels))
    return total, between


def evaluate():
    check_freeze()
    assert not (ROOT/'results.csv').exists(), 'Existing experiment is not overwritten'
    records, stats = [], []
    started = time.perf_counter()
    for variant, item in read(ROOT/'feature_manifest.json').items():
        data = np.load(BASE/item['path'], mmap_mode='r')
        for seed in SEEDS:
            train, query = split(LABELS, 2, seed)
            assert np.array_equal(np.bincount(LABELS[train])[1:], np.full(40, 2))
            e0 = energy(data[train], LABELS[train])
            for name, alg, dims, ld, paper in CASES:
                model = PCA(dims, center=True, solver='gram') if alg == 'pca' else CSA(dims, center=False, init='identity', max_iter=3)
                model.fit(data[train])
                ga, qb = transform_batches(model, data, train), transform_batches(model, data, query)
                e1 = energy(ga, LABELS[train])
                arrays = {'train': train, 'query': query, 'mean': model.mean,
                          'core_gallery': ga, 'core_probe': qb}
                if alg == 'pca':
                    arrays['pca_basis'] = model.basis
                else:
                    arrays.update({f'axis{k}': u for k, u in enumerate(model.bases)})
                condition, residual = None, None
                if ld is not None:
                    assert ga.shape[1] <= len(train)-40
                    lda = LDA().fit(ga, LABELS[train])
                    assert ld <= lda.max_components
                    w = lda.basis[:, :ld]; w = w/np.linalg.norm(w, axis=0)
                    origin = ga.mean(0)
                    a, b = (ga-origin)@w, (qb-origin)@w
                    arrays.update(lda_basis=w, lda_mean=origin, sw=lda.sw, sb=lda.sb,
                                  eigenvalues=lda.eigenvalues[:ld])
                    ev = eigh(lda.sw, eigvals_only=True)
                    condition = float(ev[-1]/ev[0])
                    residual = float(np.linalg.norm(lda.sb@w-(lda.sw@w)*lda.eigenvalues[:ld])/max(1, np.linalg.norm(lda.sb@w)))
                else:
                    a, b = ga-ga.mean(0), qb-ga.mean(0)
                # Explicitly use the actual output gallery mean for all methods.
                output_mean = a.mean(0); a, b = a-output_mean, b-output_mean
                distance = cdist(b, a, 'cosine')
                assert np.isfinite(distance).all()
                near = np.argmin(distance, axis=1); predictions = LABELS[train[near]]
                correct = int(np.sum(predictions == LABELS[query]))
                stem = f'{variant}_s{seed}_{name}'
                arrays.update(gallery=a, probe=b, output_mean=output_mean)
                np.savez_compressed(ROOT/'models'/f'{stem}.npz', **arrays)
                if alg == 'csa':
                    dump(ROOT/'models'/f'{stem}_history.json', model.history)
                np.savez_compressed(ROOT/'predictions'/f'{stem}.npz', predictions=predictions,
                                    neighbor_index=train[near], train=train, query=query)
                records.append({'variant': variant, 'seed': seed, 'method': name, 'pre_dimension': ga.shape[1],
                                'output_dimension': a.shape[1], 'correct': correct, 'probe_count': len(query),
                                'accuracy_percent': 100*correct/len(query), 'paper_percent': paper,
                                'model_file': 'models/'+stem+'.npz', 'prediction_file': 'predictions/'+stem+'.npz'})
                stats.append({'variant': variant, 'seed': seed, 'method': name,
                              'total_variation_retained': e1[0]/e0[0], 'between_variation_retained': e1[1]/e0[1],
                              'sw_condition': condition, 'fisher_relative_residual': residual})
            print(f'{variant} seed{seed}: 4 fixed models complete', flush=True)
    assert len(records) == 120
    table(ROOT/'results.csv', records); table(ROOT/'representation_statistics.csv', stats)
    dump(ROOT/'execution.json', {'outer_fits': len(records), 'prediction_records': 38400,
                                'elapsed_seconds': time.perf_counter()-started, 'BLAS_threads': 2})


def independent_csa(x, ranks):
    # Procedure1 reimplemented as explicit einsum + per-object scatter sum.
    x = np.asarray(x, dtype=float)
    u = [np.eye(m, r) for m, r in zip(x.shape[1:], ranks)]
    for iteration in range(3):
        for k, r in enumerate(ranks):
            if k == 0:
                z = np.einsum('nhwc,wb,cd->nhbd', x, u[1], u[2], optimize=True).reshape(len(x), x.shape[1], -1)
            elif k == 1:
                z = np.einsum('nhwc,ha,cd->nwad', x, u[0], u[2], optimize=True).reshape(len(x), x.shape[2], -1)
            else:
                z = np.einsum('nhwc,ha,wb->ncab', x, u[0], u[1], optimize=True).reshape(len(x), x.shape[3], -1)
            scatter = sum(a@a.T for a in z)
            _, basis = eigh((scatter+scatter.T)/2)
            u[k] = basis[:, -r:][:, ::-1]
    return u


def verify():
    check_freeze()
    rows = list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8')))
    assert len(rows) == 120
    expected = {(v, s, n) for v in VARIANTS for s in SEEDS for n, *_ in CASES}
    assert {(r['variant'], int(r['seed']), r['method']) for r in rows} == expected
    data = {v: np.load(BASE/e['path'], mmap_mode='r') for v, e in read(ROOT/'feature_manifest.json').items()}
    max_encoding, max_update, max_fisher = 0., 0., 0.
    baseline_matches, count = 0, 0
    old_paths = [('CSA390', BASE/'results/orl_sample_size/results.csv'),
                 ('CSA24_LDA23', BASE/'results/orl_sample_size/results.csv'),
                 ('PCA24_LDA23', BASE/'results/orl_sample_size/pca_control/results.csv')]
    old = {}
    for name, path in old_paths:
        for r in csv.DictReader(path.open(encoding='utf-8')):
            if r['context'] == 'anchor_P8' and int(r['gallery_per_class']) == 2 and r['method'] == name:
                old[(int(r['seed']), name)] = int(r['correct'])
    for row in rows:
        m = np.load(ROOT/row['model_file']); p = np.load(ROOT/row['prediction_file'])
        train, query = split(LABELS, 2, int(row['seed']))
        for z in (m, p):
            np.testing.assert_array_equal(z['train'], train); np.testing.assert_array_equal(z['query'], query)
        source = data[row['variant']]
        if 'pca_basis' in m:
            basis = m['pca_basis']; np.testing.assert_allclose(basis.T@basis, np.eye(basis.shape[1]), atol=1e-9)
            x = source[train].astype(float).reshape(80, -1); np.testing.assert_allclose(m['mean'], x.mean(0), atol=1e-12)
            centered = x-x.mean(0)
            values = eigh(centered@centered.T, eigvals_only=True)[::-1][:basis.shape[1]]
            np.testing.assert_allclose(centered.T@(centered@basis), basis*values, atol=2e-5, rtol=2e-8)
            encode = lambda x: (x.astype(float).reshape(len(x), -1)-m['mean'])@basis
        else:
            bases = [m[f'axis{k}'] for k in range(3)]
            assert np.count_nonzero(m['mean']) == 0
            independent = independent_csa(source[train], [u.shape[1] for u in bases])
            for a, b in zip(bases, independent):
                np.testing.assert_allclose(a.T@a, np.eye(a.shape[1]), atol=1e-10)
                error = float(np.max(np.abs(a@a.T-b@b.T)))
                max_update = max(max_update, error); assert error < 2e-7
            encode = lambda x: np.einsum('nhwc,ha,wb,cd->nabd', x.astype(float), *bases, optimize=True).reshape(len(x), -1)
        a, b = encode(source[train]), encode(source[query])
        for actual, saved in ((a, m['core_gallery']), (b, m['core_probe'])):
            max_encoding = max(max_encoding, float(np.max(np.abs(actual-saved))))
            np.testing.assert_allclose(actual, saved, atol=2e-8, rtol=1e-10)
        if 'lda_basis' in m:
            mean = a.mean(0); sw, sb = np.zeros_like(m['sw']), np.zeros_like(m['sb'])
            for c in range(1, 41):
                v = a[LABELS[train] == c]; delta = v-v.mean(0)
                sw += delta.T@delta; d = v.mean(0)-mean; sb += len(v)*np.outer(d, d)
            np.testing.assert_allclose(sw, m['sw'], atol=1e-7, rtol=1e-9)
            np.testing.assert_allclose(sb, m['sb'], atol=1e-7, rtol=1e-9)
            w = m['lda_basis']; np.testing.assert_allclose(np.linalg.norm(w, axis=0), 1, atol=1e-10)
            values, _ = eigh(sb, sw)
            np.testing.assert_allclose(values[::-1][:w.shape[1]], m['eigenvalues'], atol=1e-7, rtol=1e-7)
            residual = float(np.linalg.norm(sb@w-(sw@w)*m['eigenvalues'])/max(1, np.linalg.norm(sb@w)))
            max_fisher = max(max_fisher, residual); assert residual < 1e-8
            a, b = (a-mean)@w, (b-mean)@w
        else:
            mean = a.mean(0); a, b = a-mean, b-mean
        output_mean = a.mean(0); a, b = a-output_mean, b-output_mean
        np.testing.assert_allclose(a, m['gallery'], atol=2e-8, rtol=1e-9)
        np.testing.assert_allclose(b, m['probe'], atol=2e-8, rtol=1e-9)
        # Independent normalized dot products, not the scoring cdist path.
        near = np.argmax((b/np.linalg.norm(b, axis=1, keepdims=True))@(a/np.linalg.norm(a, axis=1, keepdims=True)).T, axis=1)
        pred = LABELS[train[near]]
        np.testing.assert_array_equal(p['predictions'], pred); np.testing.assert_array_equal(p['neighbor_index'], train[near])
        correct = int(np.sum(pred == LABELS[query]))
        assert correct == int(row['correct']) and abs(100*correct/320-float(row['accuracy_percent'])) < 1e-12
        if row['variant'] == 'baseline' and (int(row['seed']), row['method']) in old:
            assert correct == old[(int(row['seed']), row['method'])]
            baseline_matches += 1
        count += len(pred)
        if row['method'] == 'PCA33_LDA25':
            print(f"Verified {row['variant']} seed{row['seed']}: all four methods", flush=True)
    assert count == 38400 and baseline_matches == 30
    dump(ROOT/'verification.json', {'passed': True, 'outer_models_checked': 120,
         'independent_CSA_three_sweep_refits': 60, 'PCA_eigen_residual_checks': 60,
         'prediction_records_checked': count, 'historical_baseline_score_matches': baseline_matches,
         'max_encoding_abs_error': max_encoding, 'max_CSA_projector_abs_error': max_update,
         'max_Fisher_relative_residual': max_fisher, 'freeze_hashes_unchanged': True})


def gallery_sensitivity():
    # Small fixed mechanism diagnostic, with no extra fit or probe images.
    raw, eyes = raw_inputs(); train, _ = split(LABELS, 2, 20)
    result = []
    for variant in VARIANTS:
        for shift in (-.5, .5):
            perturbed = eyes[train].copy(); perturbed[:, :, 0] += shift
            gray = gray_input(raw[train], perturbed, variant)
            tensors = context_features(gray) if variant == 'crop_context' else filtered(gray, SPEC)
            for name, alg, _, _, _ in CASES:
                m = np.load(ROOT/'models'/f'{variant}_s20_{name}.npz')
                if alg == 'pca':
                    core = (tensors.astype(float).reshape(80, -1)-m['mean'])@m['pca_basis']
                else:
                    core = np.einsum('nhwc,ha,wb,cd->nabd', tensors.astype(float),
                                     m['axis0'], m['axis1'], m['axis2'], optimize=True).reshape(80, -1)
                original = m['core_gallery']; mean = original.mean(0)
                change = float(np.linalg.norm(core-original)/max(np.linalg.norm(original-mean), 1e-12))
                result.append({'variant': variant, 'seed': 20, 'method': name,
                               'source_eye_x_shift_pixels': shift, 'relative_core_change': change,
                               'gallery_count': 80, 'extra_fits': 0})
    table(ROOT/'gallery_eye_sensitivity.csv', result)


def analyze():
    check_freeze(); assert read(ROOT/'verification.json')['passed']
    rows = list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8')))
    summary, paired = [], []
    for name, _, _, _, paper in CASES:
        base = {int(r['seed']): float(r['accuracy_percent']) for r in rows if r['variant'] == 'baseline' and r['method'] == name}
        for variant in VARIANTS:
            scores = {int(r['seed']): float(r['accuracy_percent']) for r in rows if r['variant'] == variant and r['method'] == name}
            values = np.array([scores[s] for s in SEEDS]); delta = values-np.array([base[s] for s in SEEDS])
            summary.append({'variant': variant, 'method': name, 'n_splits': 10,
                            'mean_percent': float(values.mean()), 'std_pp': float(values.std(ddof=1)),
                            'min_percent': float(values.min()), 'max_percent': float(values.max()),
                            'baseline_change_pp': float(delta.mean()), 'wins': int(np.sum(delta > 0)),
                            'ties': int(np.sum(delta == 0)), 'losses': int(np.sum(delta < 0)),
                            'paper_percent': paper, 'paper_difference_pp': None if paper is None else float(values.mean()-paper)})
            if variant != 'baseline':
                paired.extend({'variant': variant, 'method': name, 'seed': s,
                               'baseline_percent': base[s], 'new_percent': scores[s], 'change_pp': scores[s]-base[s]} for s in SEEDS)
    dump(ROOT/'summary.json', summary); table(ROOT/'comparison.csv', summary); table(ROOT/'paired_changes.csv', paired)
    gallery_sensitivity()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, names in zip(axes, ([c[0] for c in CASES[:2]], [c[0] for c in CASES[2:]])):
        for offset, variant in zip((-.22, 0, .22), VARIANTS):
            group = [next(r for r in summary if r['method'] == n and r['variant'] == variant) for n in names]
            ax.errorbar(np.arange(2)+offset, [r['mean_percent'] for r in group],
                        yerr=[r['std_pp'] for r in group], fmt='o', capsize=4, label=variant)
        for j, name in enumerate(names):
            paper = next(c[4] for c in CASES if c[0] == name)
            if paper is not None:
                ax.scatter(j, paper, marker='*', color='black', s=90)
        ax.set(xticks=range(2), xticklabels=names, ylabel='Accuracy (%)', ylim=(65, 100))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('ORL G2/P8: 10 matched splits, fixed dimensions and shared centered cosine\nMean ± sample SD; black stars = paper best results (different aggregation)')
    fig.tight_layout(); fig.savefig(ROOT/'comparison.png', dpi=180); plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'evaluate', 'verify', 'analyze'], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.stage]()
