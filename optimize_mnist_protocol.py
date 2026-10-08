"""Bounded MNIST protocol study: shared inputs, gallery-only CV, paper dimensions.

Stages freeze the plan, audit sources, prepare Gabor, select all splits, evaluate,
independently verify predictions/selected CV, then summarize. No probe selection.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.linalg import eigh
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA, PCA
from gabor import CONFIG, file_hash, kernel

BASE = Path(__file__).resolve().parent
SOURCE = BASE/'data/processed/available'
OLD = BASE/'results/available_reproduction_v2'
INPUTS = ['adaptive_reflect', 'adaptive_zero', 'fixed39_reflect', 'fixed39_zero']
POLICIES = ['cosine_raw', 'cosine_gallery', 'euclidean']
CASES = {30: {'pca': 60, 'csa': (5, 9, 18), 'paper': {'pca': 88.5, 'csa': 92.8}},
         40: {'pca': 50, 'csa': (15, 5, 13), 'paper': {'pca': 90.5, 'csa': 92.4}}}
SEEDS = list(range(20, 30))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def partition(labels, count, seed):
    rng = np.random.default_rng(seed)
    gallery, probe = [], []
    for label in np.unique(labels):
        order = rng.permutation(np.flatnonzero(labels == label))
        gallery.extend(order[:count])
        probe.extend(order[count:])
    return np.asarray(gallery), np.asarray(probe)


def folds(labels, gallery, seed):
    rng = np.random.default_rng(603314+seed)
    blocks = [np.array_split(rng.permutation(gallery[labels[gallery] == digit]), 5)
              for digit in np.unique(labels)]
    result = []
    for number in range(5):
        query = np.concatenate([block[number] for block in blocks])
        learn = gallery[~np.isin(gallery, query)]
        assert not np.intersect1d(learn, query).size
        result.append((learn, query))
    assert np.array_equal(np.sort(np.concatenate([q for _, q in result])), np.sort(gallery))
    return result


def make_kernel(identifier, scale, orientation):
    wavelength = CONFIG['wavelengths'][scale]
    theta = CONFIG['orientations'][orientation]
    if identifier.startswith('adaptive'):
        # Retain the exact legacy ceil radii in both boundary variants.
        # This isolates padding rather than silently changing radius rounding.
        return kernel(wavelength, theta).astype(np.complex64)
    y, x = np.mgrid[-19:20, -19:20]
    k, sigma = 2*np.pi/wavelength, CONFIG['sigma']
    value = k*k/sigma**2*np.exp(-k*k*(x*x+y*y)/(2*sigma*sigma))* (
        np.exp(1j*k*(x*np.cos(theta)+y*np.sin(theta)))-np.exp(-sigma*sigma/2))
    return (value-value.mean()).astype(np.complex64)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    (root/'source_snapshot').mkdir()
    source_names = ['optimize_mnist_protocol.py', 'csa.py', 'gabor.py']
    for name in source_names:
        shutil.copy2(BASE/name, root/'source_snapshot'/name)
    protocol = {
        'role': 'MNIST bounded optimization; shared paper-dimension comparison, not exact author protocol',
        'seeds': SEEDS, 'pilot_seeds': [20, 21, 22], 'gallery_counts': [30, 40],
        'source_subset': 'first500 per label from official10000 test images, unchanged',
        'shape': [28, 28, 40], 'Tmax': 3, 'cases': CASES,
        'inputs': INPUTS, 'gabor': CONFIG,
        'adaptive_radii': [make_kernel(INPUTS[0], s, 0).shape[0]//2 for s in range(5)],
        'fixed_support': '39x39, radius19',
        'padding': 'reflect or constant-zero; identical kernel within each support pair',
        'response': 'complex magnitude; no sqrt/power/L2/channel normalization',
        'inner': 'stratified5fold inside each own gallery; every gallery image validates once; learning24/32 per class',
        'screen': 'four shared inputs, identity uncentered CSA and centered PCA, raw cosine; pooled mean of two methods',
        'shortlist': 'top2 per outer split and gallery count; tie prefers input order; no cross-split shortlist transfer',
        'selection': 'shortlisted inputs x three shared NN policies x identity/HOSVD CSA starts; maximize pooled mean of PCA and CSA; then worst method-fold, then earlier input, identity, policy order',
        'policies': POLICIES, 'CSA_center': False, 'PCA_center': True,
        'HOSVD': 'object-mode bases from current learning images only; sample axis never compressed',
        'freeze': 'all20 selections and training scores replayed before any current probe scoring',
        'evaluation_profiles': ['baseline', 'input_only', 'shared_policy_identity', 'selected'],
        'controls': 'input_only changes only Gabor; shared_policy_identity adds chosen shared NN; selected additionally uses chosen CSA init; all paired within same seed',
        'limits': ['historical probes observed; continued development, not blind',
                   'overlapping random splits are not independent samples',
                   'finite shortlist may miss interactions',
                   'author sample indices, full Gabor/NN/init/statistical details unknown',
                   '5fold validation gallery24/32 differs from final30/40'],
        'TA_compliance': '500/class, G30/P470 and G40/P460,28x28x40,5scales8orientations,NN,Tmax3; shared inputs/distance; no LDA or extra training samples',
        'original_compliance': 'TableIV dimensions retained; core CSA unchanged; unspecified choices are group conventions, not confirmed author settings',
        'source_sha256': {name: file_hash(BASE/name) for name in source_names},
        'input_sha256': {name: file_hash(SOURCE/name) for name in
                         ['mnist_balanced.npy', 'mnist_labels.npy', 'mnist_source_indices.csv', 'mnist_gabor.npy']},
        'old_result_sha256': {name: file_hash(OLD/name) for name in
                             ['fixed_results.csv', 'predictions.csv', 'protocol.json', 'verification.json']},
    }
    dump(root/'protocol.json', protocol)
    audit(root)


def check_sources(root):
    protocol = read(root/'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        assert file_hash(BASE/name) == digest, f'Source changed: {name}'
    return protocol


def audit(root):
    protocol = check_sources(root)
    for name, digest in protocol['input_sha256'].items():
        assert file_hash(SOURCE/name) == digest
    for name, digest in protocol['old_result_sha256'].items():
        assert file_hash(OLD/name) == digest
    raw_images = np.frombuffer((BASE/'data/raw/mnist/t10k-images-idx3-ubyte').read_bytes(), np.uint8, offset=16).reshape(10000, 28, 28)
    raw_labels = np.frombuffer((BASE/'data/raw/mnist/t10k-labels-idx1-ubyte').read_bytes(), np.uint8, offset=8)
    indices = np.concatenate([np.flatnonzero(raw_labels == d)[:500] for d in range(10)])
    labels = np.load(SOURCE/'mnist_labels.npy')
    assert np.array_equal(np.load(SOURCE/'mnist_balanced.npy'), raw_images[indices])
    assert np.array_equal(labels, raw_labels[indices])
    index_rows = rows(SOURCE/'mnist_source_indices.csv')
    assert np.array_equal(indices, [int(r['test_index_zero_based']) for r in index_rows])
    assert np.array_equal(np.bincount(labels), np.full(10, 500))
    previous = [r for r in rows(OLD/'fixed_results.csv') if r['dataset'].startswith('MNIST') and r['profile'] == 'group_primary']
    predictions = [r for r in rows(OLD/'predictions.csv') if r['dataset'].startswith('MNIST') and r['profile'] == 'group_primary']
    for record in previous:
        group = [p for p in predictions if all(p[k] == record[k] for k in ['dataset', 'seed', 'method'])]
        assert len(group) == int(record['probe_count'])
        assert sum(int(p['correct']) for p in group) == int(record['correct'])
        assert all(int(p['correct']) == (int(p['true']) == int(p['predicted'])) for p in group)
    dump(root/'baseline_audit.json', {'passed': True, 'old_scores_checked': len(previous),
         'old_prediction_records_checked': len(predictions), 'balanced_images': 5000,
         'raw_IDX_source_reconstructed': True, 'source_indices_and_hashes': True,
         'limits': 'historical score readback; actual baseline models refitted during evaluation'})
    print('AUDIT passed:12 historical scores,55800 predictions,5000 source images', flush=True)


def feature_paths(root):
    return {INPUTS[0]: SOURCE/'mnist_gabor.npy',
            **{identifier: root/'features'/f'{identifier}.npy' for identifier in INPUTS[1:]}}


def prepare_features(root):
    check_sources(root)
    assert not (root/'feature_manifest.json').exists(), 'Features already frozen'
    (root/'features').mkdir(exist_ok=True)
    images = np.load(SOURCE/'mnist_balanced.npy')
    manifest, checks = {}, []
    baseline = np.load(SOURCE/'mnist_gabor.npy', mmap_mode='r')
    for identifier, destination in feature_paths(root).items():
        tick = time.perf_counter()
        output = baseline if identifier == INPUTS[0] else np.lib.format.open_memmap(
            destination, mode='w+', dtype=np.float32, shape=(5000, 28, 28, 40))
        for scale in range(5):
            kernels = [make_kernel(identifier, scale, o) for o in range(8)]
            radius = kernels[0].shape[0]//2
            shape = (next_fast_len(28+4*radius),)*2
            spectra = [fftn(k, s=shape) for k in kernels]
            padding = 'reflect' if identifier.endswith('reflect') else 'constant'
            for start in range(0, 5000, 32):
                if identifier == INPUTS[0] and start > 0:
                    break  # Existing frozen cache is reused; first batch independently re-generated.
                padded = np.pad(images[start:start+32].astype(np.float32),
                                ((0, 0), (radius, radius), (radius, radius)), mode=padding)
                spectrum = fftn(padded, s=shape, axes=(-2, -1))
                for orientation, transformed in enumerate(spectra):
                    actual = np.abs(ifftn(spectrum*transformed, axes=(-2, -1))[:,
                                         2*radius:2*radius+28, 2*radius:2*radius+28])
                    channel = scale*8+orientation
                    if identifier == INPUTS[0]:
                        np.testing.assert_array_equal(actual, output[:len(actual), :, :, channel])
                    else:
                        output[start:start+len(actual), :, :, channel] = actual
                    if start == 0:
                        expected = np.abs(convolve2d(padded[0].astype(float), kernels[orientation].astype(np.complex128), mode='valid'))
                        error = float(np.max(np.abs(actual[0]-expected)))
                        assert error < 3e-4, (identifier, scale, orientation, error)
                        checks.append({'input': identifier, 'channel': channel, 'max_abs_error': error})
            print(f'FEATURE {identifier} scale={scale+1}/5 elapsed={time.perf_counter()-tick:.1f}s', flush=True)
        if identifier != INPUTS[0]:
            output.flush()
        assert output.shape == (5000, 28, 28, 40)
        for start in range(0, 5000, 128):
            assert np.isfinite(output[start:start+128]).all()
        manifest[identifier] = {'path': str(destination.relative_to(BASE)), 'sha256': file_hash(destination),
                                'shape': list(output.shape), 'seconds': time.perf_counter()-tick}
        del output
    dump(root/'feature_manifest.json', manifest)
    dump(root/'convolution_verification.json', {'passed': True, 'channels_checked': len(checks),
         'checks': checks, 'max_abs_error': max(c['max_abs_error'] for c in checks),
         'baseline_first32_exact_cache_match': True, 'independent': 'float64 direct spatial convolve2d against complex64 FFT'})


def load_data(root, identifier):
    return np.load(BASE/read(root/'feature_manifest.json')[identifier]['path'], mmap_mode='r')


def hosvd(x, ranks):
    bases = []
    for mode, rank in enumerate(ranks):
        matrix = np.moveaxis(x, mode+1, 0).reshape(x.shape[mode+1], -1)
        _, vectors = eigh(matrix@matrix.T)
        bases.append(vectors[:, -rank:][:, ::-1])
    return bases


def fit_model(data, learn, count, algorithm, init='identity'):
    x = np.asarray(data[learn], dtype=float)
    if algorithm == 'pca':
        model = PCA(CASES[count]['pca'], center=True, solver='gram')
    else:
        ranks = CASES[count]['csa']
        initial = hosvd(x, ranks) if init == 'hosvd' else None
        model = CSA(ranks, center=False, max_iter=3, initial_bases=initial)
    return model.fit(x)


def encode(model, data, indices):
    return np.concatenate([model.transform(data[indices[start:start+128]]).reshape(len(indices[start:start+128]), -1)
                           for start in range(0, len(indices), 128)])


def nn(gallery, query, labels, policy, independent=False):
    mean = gallery.mean(0) if policy == 'cosine_gallery' else np.zeros(gallery.shape[1])
    g, q = gallery-mean, query-mean
    metric = 'euclidean' if policy == 'euclidean' else 'cosine'
    predicted, neighbors, distances = [], [], []
    if independent and metric == 'cosine':
        g = g/np.linalg.norm(g, axis=1, keepdims=True)
        q = q/np.linalg.norm(q, axis=1, keepdims=True)
    for start in range(0, len(q), 256):
        block = q[start:start+256]
        if not independent:
            distance = cdist(block, g, metric)
        elif metric == 'cosine':
            distance = np.clip(1-block@g.T, 0, 2)
        else:
            distance = np.maximum(np.sum(block*block, axis=1)[:, None]+np.sum(g*g, axis=1)[None, :]-2*block@g.T, 0)
        assert np.isfinite(distance).all()
        neighbor = np.argmin(distance, axis=1)
        predicted.extend(labels[neighbor])
        neighbors.extend(neighbor)
        distances.extend(distance[np.arange(len(block)), neighbor])
    return np.asarray(predicted), np.asarray(neighbors), np.asarray(distances)


def cv(root, seed, count, identifier, algorithm, init):
    directory = root/'selection'/f'G{count}_seed{seed}'/'cv'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{identifier}_{algorithm}_{init}.npz'
    labels = np.load(SOURCE/'mnist_labels.npy')
    gallery, _ = partition(labels, count, seed)
    learning_folds = folds(labels, gallery, seed)
    if not path.exists():
        data = load_data(root, identifier)
        payload = {}
        tick = time.perf_counter()
        for number, (learn, query) in enumerate(learning_folds):
            model = fit_model(data, learn, count, algorithm, init)
            payload.update({f'g{number}': encode(model, data, learn), f'q{number}': encode(model, data, query),
                            f'learn{number}': learn, f'query{number}': query})
        np.savez_compressed(path, **payload)
        print(f'CV G{count} seed={seed} {identifier} {algorithm}/{init} {time.perf_counter()-tick:.1f}s', flush=True)
    result = {}
    with np.load(path) as stored:
        for policy in POLICIES:
            correct = []
            for number, (learn, query) in enumerate(learning_folds):
                assert np.array_equal(stored[f'learn{number}'], learn)
                assert np.array_equal(stored[f'query{number}'], query)
                predictions, _, _ = nn(stored[f'g{number}'], stored[f'q{number}'], labels[learn], policy)
                correct.append(int(np.sum(predictions == labels[query])))
            result[policy] = {'correct': sum(correct), 'count': len(gallery), 'fold_correct': correct,
                              'fold_counts': [len(query) for _, query in learning_folds]}
    return result


def select(root, seeds):
    check_sources(root)
    assert not (root/'evaluation').exists(), 'Selection cannot continue after probe scoring'
    for seed in seeds:
        assert seed in SEEDS
        for count in CASES:
            folder = root/'selection'/f'G{count}_seed{seed}'
            selected_path = folder/'selected.json'
            if selected_path.exists():
                print(f'RETAIN frozen G{count} seed={seed}', flush=True)
                continue
            screen, caches = [], {}
            for identifier in INPUTS:
                for algorithm in ['pca', 'csa']:
                    caches[identifier, algorithm, 'identity'] = cv(root, seed, count, identifier, algorithm, 'identity')
                a, b = [caches[identifier, alg, 'identity']['cosine_raw'] for alg in ['pca', 'csa']]
                screen.append({'input': identifier, 'pca_correct': a['correct'], 'csa_correct': b['correct'],
                               'joint_correct': a['correct']+b['correct'], 'total': 2*a['count']})
            shortlisted = sorted(INPUTS, key=lambda identifier: (-next(r['joint_correct'] for r in screen if r['input'] == identifier), INPUTS.index(identifier)))[:2]
            candidates = []
            for identifier in shortlisted:
                caches[identifier, 'csa', 'hosvd'] = cv(root, seed, count, identifier, 'csa', 'hosvd')
                for init in ['identity', 'hosvd']:
                    for policy in POLICIES:
                        a = caches[identifier, 'pca', 'identity'][policy]
                        b = caches[identifier, 'csa', init][policy]
                        worst = min(correct/total for record in [a, b] for correct, total in zip(record['fold_correct'], record['fold_counts']))
                        candidates.append({'input': identifier, 'CSA_init': init, 'policy': policy,
                            'pca_correct': a['correct'], 'csa_correct': b['correct'], 'joint_correct': a['correct']+b['correct'],
                            'total': 2*a['count'], 'worst_method_fold': worst})
            winner = min(candidates, key=lambda r: (-r['joint_correct'], -r['worst_method_fold'],
                         INPUTS.index(r['input']), ['identity', 'hosvd'].index(r['CSA_init']), POLICIES.index(r['policy'])))
            write_csv(folder/'screen.csv', screen)
            write_csv(folder/'candidates.csv', candidates)
            dump(selected_path, {'seed': seed, 'gallery_count': count, 'shortlist': shortlisted, 'winner': winner,
                 'protocol_sha256': file_hash(root/'protocol.json')})
            print(f'SELECT G{count} seed={seed}: {winner}', flush=True)
    all_paths = [root/'selection'/f'G{count}_seed{seed}'/'selected.json' for seed in SEEDS for count in CASES]
    if all(path.exists() for path in all_paths):
        freeze = {'selection_sha256': {str(path.relative_to(root)): file_hash(path) for path in all_paths},
                  'protocol_sha256': file_hash(root/'protocol.json'), 'probe_scored': False}
        dump(root/'selection_freeze.json', freeze)
        verify_cv(root, refit=True)


def verify_cv(root, refit):
    labels = np.load(SOURCE/'mnist_labels.npy')
    checks, refits = 0, 0
    freeze = read(root/'selection_freeze.json')
    for relative, digest in freeze['selection_sha256'].items():
        assert file_hash(root/relative) == digest
        selected = read(root/relative)
        seed, count, winner = selected['seed'], selected['gallery_count'], selected['winner']
        gallery, _ = partition(labels, count, seed)
        folder = (root/relative).parent
        candidates = rows(folder/'candidates.csv')
        expected = min(candidates, key=lambda r: (-int(r['joint_correct']), -float(r['worst_method_fold']),
                      INPUTS.index(r['input']), ['identity', 'hosvd'].index(r['CSA_init']), POLICIES.index(r['policy'])))
        assert all(str(winner[k]) == str(expected[k]) for k in ['input', 'CSA_init', 'policy', 'joint_correct'])
        screen = rows(folder/'screen.csv')
        expected_short = [r['input'] for r in sorted(screen, key=lambda r: (-int(r['joint_correct']), INPUTS.index(r['input'])))[:2]]
        assert expected_short == selected['shortlist']
        # Independently replay EVERY cached fold and every distance, then all election counts.
        independent_counts = {}
        for path in sorted((folder/'cv').glob('*.npz')):
            with np.load(path) as stored:
                totals = {p: [] for p in POLICIES}
                for number, (learn, query) in enumerate(folds(labels, gallery, seed)):
                    assert np.array_equal(stored[f'learn{number}'], learn)
                    assert np.array_equal(stored[f'query{number}'], query)
                    for policy in POLICIES:
                        prediction, _, _ = nn(stored[f'g{number}'], stored[f'q{number}'], labels[learn], policy, True)
                        totals[policy].append(int(np.sum(prediction == labels[query])))
                        checks += len(query)
            independent_counts[path.stem] = totals
        for record in screen:
            a = sum(independent_counts[f"{record['input']}_pca_identity"]['cosine_raw'])
            b = sum(independent_counts[f"{record['input']}_csa_identity"]['cosine_raw'])
            assert (a, b, a+b) == tuple(int(record[k]) for k in ['pca_correct', 'csa_correct', 'joint_correct'])
        for record in candidates:
            a = independent_counts[f"{record['input']}_pca_identity"][record['policy']]
            b = independent_counts[f"{record['input']}_csa_{record['CSA_init']}"][record['policy']]
            assert (sum(a), sum(b), sum(a+b)) == tuple(int(record[k]) for k in ['pca_correct', 'csa_correct', 'joint_correct'])
            denominators = [len(q) for _, q in folds(labels, gallery, seed)]
            assert abs(min(c/n for values in [a, b] for c, n in zip(values, denominators))-float(record['worst_method_fold'])) < 1e-14
        if refit:
            data = load_data(root, winner['input'])
            for algorithm in ['pca', 'csa']:
                init = 'identity' if algorithm == 'pca' else winner['CSA_init']
                count_correct = 0
                for learn, query in folds(labels, gallery, seed):
                    model = fit_model(data, learn, count, algorithm, init)
                    prediction, _, _ = nn(encode(model, data, learn), encode(model, data, query), labels[learn], winner['policy'], True)
                    count_correct += int(np.sum(prediction == labels[query]))
                    refits += 1
                assert count_correct == winner[f'{algorithm}_correct']
        print(f'CV VERIFY G{count} seed={seed} passed', flush=True)
    dump(root/'selection_verification.json', {'passed': True, 'independent_validation_prediction_checks': checks,
         'selected_CV_models_refitted': refits, 'all_screen_candidates_and_ties_recomputed': True,
         'all_validation_inside_own_gallery': True, 'probe_not_scored_before_this_verification': not (root/'evaluation').exists()})


def save_model(model, algorithm):
    payload = {'model_mean': model.mean}
    if algorithm == 'pca':
        payload['basis'] = model.basis
    else:
        payload.update({f'basis{k}': basis for k, basis in enumerate(model.bases)})
    return payload


def evaluate(root):
    check_sources(root)
    assert read(root/'selection_verification.json')['passed']
    freeze = read(root/'selection_freeze.json')
    for relative, digest in freeze['selection_sha256'].items():
        assert file_hash(root/relative) == digest
    out = root/'evaluation'
    out.mkdir(exist_ok=False)
    (out/'models').mkdir()
    labels = np.load(SOURCE/'mnist_labels.npy')
    scores, settings = [], []
    for seed in SEEDS:
        for count in CASES:
            winner = read(root/'selection'/f'G{count}_seed{seed}'/'selected.json')['winner']
            gallery, probe = partition(labels, count, seed)
            profiles = {'baseline': (INPUTS[0], 'cosine_raw', 'identity'),
                        'input_only': (winner['input'], 'cosine_raw', 'identity'),
                        'shared_policy_identity': (winner['input'], winner['policy'], 'identity'),
                        'selected': (winner['input'], winner['policy'], winner['CSA_init'])}
            model_cache = {}
            for profile, (identifier, policy, csa_init) in profiles.items():
                for algorithm in ['pca', 'csa']:
                    init = csa_init if algorithm == 'csa' else 'identity'
                    key = (identifier, algorithm, init)
                    tick = time.perf_counter()
                    if key not in model_cache:
                        data = load_data(root, identifier)
                        model = fit_model(data, gallery, count, algorithm, init)
                        g, q = encode(model, data, gallery), encode(model, data, probe)
                        filename = f'G{count}_seed{seed}_{identifier}_{algorithm}_{init}.npz'
                        np.savez_compressed(out/'models'/filename, gallery=g, probe=q, gallery_indices=gallery,
                                            probe_indices=probe, **save_model(model, algorithm))
                        model_cache[key] = (g, q, filename)
                    g, q, filename = model_cache[key]
                    predicted, neighbor, distance = nn(g, q, labels[gallery], policy)
                    prediction_file = f'G{count}_seed{seed}_{profile}_{algorithm}_predictions.npz'
                    np.savez_compressed(out/prediction_file, true=labels[probe], predicted=predicted,
                                        probe_indices=probe, neighbor_indices=gallery[neighbor], distance=distance)
                    correct = int(np.sum(predicted == labels[probe]))
                    scores.append({'gallery_count': count, 'seed': seed, 'profile': profile, 'algorithm': algorithm,
                        'input': identifier, 'policy': policy, 'init': init, 'dimension': int(np.prod(CASES[count][algorithm])),
                        'correct': correct, 'probe_count': len(probe), 'accuracy_percent': 100*correct/len(probe),
                        'paper_percent': CASES[count]['paper'][algorithm], 'model_file': f'models/{filename}',
                        'prediction_file': prediction_file, 'wall_seconds': time.perf_counter()-tick})
            settings.append({'gallery_count': count, 'seed': seed, **winner})
            write_csv(out/'results.csv', scores)
            print(f'EVALUATE G{count} seed={seed}: '+str([(r['profile'], r['algorithm'], round(r['accuracy_percent'], 4)) for r in scores[-8:]]), flush=True)
    write_csv(root/'selected_settings.csv', settings)
    dump(out/'completion.json', {'scores': len(scores), 'all_seeds': SEEDS, 'protocol_sha256': file_hash(root/'protocol.json')})


def direct_core(x, bases):
    return np.einsum('nhwc,ha,wb,cd->nabd', x, *bases, optimize=True)


def independent_updates(x, count, init, final_bases):
    ranks = CASES[count]['csa']
    if init == 'hosvd':
        bases = []
        for mode, rank in enumerate(ranks):
            covariance = sum(np.moveaxis(image, mode, 0).reshape(image.shape[mode], -1) @
                             np.moveaxis(image, mode, 0).reshape(image.shape[mode], -1).T for image in x)
            _, vectors = eigh(covariance)
            bases.append(vectors[:, -rank:][:, ::-1])
    else:
        bases = [np.eye(size, rank) for size, rank in zip(x.shape[1:], ranks)]
    for _ in range(3):
        for mode, rank in enumerate(ranks):
            a, b, c = bases
            if mode == 0:
                projected = np.einsum('nhwc,wb,cd->nhbd', x, b, c, optimize=True)
            elif mode == 1:
                projected = np.einsum('nhwc,ha,cd->nwad', x, a, c, optimize=True)
            else:
                projected = np.einsum('nhwc,ha,wb->ncab', x, a, b, optimize=True)
            matrix = projected.reshape(len(x), x.shape[mode+1], -1)
            covariance = sum(sample@sample.T for sample in matrix)
            _, vectors = eigh((covariance+covariance.T)/2)
            bases[mode] = vectors[:, -rank:][:, ::-1]
    return max(float(np.linalg.norm(a@a.T-b@b.T)) for a, b in zip(bases, final_bases))


def verify(root):
    protocol = check_sources(root)
    assert read(root/'evaluation/completion.json')['scores'] == 160
    for identifier, entry in read(root/'feature_manifest.json').items():
        assert file_hash(BASE/entry['path']) == entry['sha256']
    for relative, digest in read(root/'selection_freeze.json')['selection_sha256'].items():
        assert file_hash(root/relative) == digest
    labels = np.load(SOURCE/'mnist_labels.npy')
    records = rows(root/'evaluation/results.csv')
    old_records = rows(OLD/'fixed_results.csv')
    checked_models, checked_updates, max_update_error = set(), set(), 0.
    prediction_count, max_encoding_error = 0, 0.
    for record in records:
        count, seed = int(record['gallery_count']), int(record['seed'])
        gallery, probe = partition(labels, count, seed)
        assert np.array_equal(np.bincount(labels[gallery]), np.full(10, count))
        assert np.array_equal(np.bincount(labels[probe]), np.full(10, 500-count))
        assert not np.intersect1d(gallery, probe).size
        with np.load(root/'evaluation'/record['model_file']) as model:
            assert np.array_equal(model['gallery_indices'], gallery)
            assert np.array_equal(model['probe_indices'], probe)
            g, q = model['gallery'], model['probe']
            if record['model_file'] not in checked_models:
                data = load_data(root, record['input'])
                sample_indices = np.r_[gallery[:3], probe[:3]]
                x = np.asarray(data[sample_indices], dtype=float)
                if record['algorithm'] == 'pca':
                    basis = model['basis']
                    np.testing.assert_allclose(basis.T@basis, np.eye(basis.shape[1]), atol=1e-9)
                    expected = np.stack([(image.reshape(-1)-model['model_mean'])@basis for image in x])
                else:
                    bases = [model[f'basis{k}'] for k in range(3)]
                    for basis in bases:
                        np.testing.assert_allclose(basis.T@basis, np.eye(basis.shape[1]), atol=1e-10)
                    assert not model['model_mean'].any()
                    expected = direct_core(x, bases).reshape(6, -1)
                    update_key = (count, record['init'])
                    if update_key not in checked_updates:
                        error = independent_updates(np.asarray(data[gallery], dtype=float), count, record['init'], bases)
                        assert error < 1e-6, error
                        max_update_error = max(max_update_error, error)
                        checked_updates.add(update_key)
                actual = np.r_[g[:3], q[:3]]
                error = float(np.max(np.abs(expected-actual)))
                np.testing.assert_allclose(expected, actual, atol=1e-7, rtol=1e-10)
                max_encoding_error = max(max_encoding_error, error)
                checked_models.add(record['model_file'])
            predicted, neighbors, _ = nn(g, q, labels[gallery], record['policy'], True)
        with np.load(root/'evaluation'/record['prediction_file']) as stored:
            assert np.array_equal(stored['probe_indices'], probe)
            assert np.array_equal(stored['true'], labels[probe])
            assert np.array_equal(stored['predicted'], predicted)
            assert np.array_equal(stored['neighbor_indices'], gallery[neighbors])
        correct = int(np.sum(predicted == labels[probe]))
        assert correct == int(record['correct'])
        assert abs(100*correct/len(probe)-float(record['accuracy_percent'])) < 1e-12
        prediction_count += len(probe)
        if record['profile'] == 'baseline' and seed in [20, 21, 22]:
            method = 'PCA (Gabor)' if record['algorithm'] == 'pca' else 'CSA-3'
            previous = next(r for r in old_records if r['dataset'] == f'MNIST G{count}/P{500-count}' and
                            r['profile'] == 'group_primary' and r['method'] == method and int(r['seed']) == seed)
            assert correct == int(previous['correct']), 'Historical baseline changed'
    assert len(records) == 160
    assert read(root/'selection_verification.json')['selected_CV_models_refitted'] == 200
    dump(root/'verification.json', {'passed': True, 'scores_checked': len(records), 'prediction_records_checked': prediction_count,
        'unique_outer_models_checked': len(checked_models), 'independent_CSA_update_cases': len(checked_updates),
        'max_update_projector_error': max_update_error, 'max_encoding_absolute_error': max_encoding_error,
        'historical12_baseline_scores_exact_match': True, 'all_inputs_and_selection_hashes_unchanged': True,
        'limits': 'all NN predictions independently replayed; encodings sampled per unique model; updates independently refitted per G/init, not every model; not proof of unknown author protocol'})
    print(f'VERIFY passed:{len(records)} scores,{prediction_count} prediction records,{len(checked_models)} models', flush=True)


def analyze(root):
    assert read(root/'verification.json')['passed']
    records = rows(root/'evaluation/results.csv')
    summaries, paired = [], []
    for count in CASES:
        for algorithm in ['pca', 'csa']:
            for profile in ['baseline', 'input_only', 'shared_policy_identity', 'selected']:
                group = sorted([r for r in records if int(r['gallery_count']) == count and r['algorithm'] == algorithm and r['profile'] == profile], key=lambda r: int(r['seed']))
                values = np.array([float(r['accuracy_percent']) for r in group])
                baseline = np.array([float(next(b['accuracy_percent'] for b in records if int(b['gallery_count']) == count and b['algorithm'] == algorithm and b['profile'] == 'baseline' and b['seed'] == r['seed'])) for r in group])
                delta = values-baseline
                summaries.append({'gallery_count': count, 'algorithm': algorithm, 'profile': profile, 'n_splits': len(values),
                    'mean_percent': float(values.mean()), 'std_pp': float(values.std(ddof=1)), 'min_percent': float(values.min()),
                    'max_percent': float(values.max()), 'paper_percent': CASES[count]['paper'][algorithm],
                    'paper_difference_pp': float(values.mean()-CASES[count]['paper'][algorithm]),
                    'baseline_change_pp': float(delta.mean()), 'improved': int(np.sum(delta > 1e-10)),
                    'tied': int(np.sum(np.abs(delta) <= 1e-10)), 'declined': int(np.sum(delta < -1e-10))})
                for r, change in zip(group, delta):
                    paired.append({'gallery_count': count, 'algorithm': algorithm, 'seed': int(r['seed']), 'profile': profile,
                                   'accuracy_percent': float(r['accuracy_percent']), 'baseline_change_pp': float(change)})
    dump(root/'summary.json', summaries)
    write_csv(root/'comparison.csv', summaries)
    write_csv(root/'paired_changes.csv', paired)
    pilot = [{**s, 'n_splits': 3,
              'mean_percent': float(np.mean([float(r['accuracy_percent']) for r in records if int(r['gallery_count']) == s['gallery_count'] and r['algorithm'] == s['algorithm'] and r['profile'] == s['profile'] and int(r['seed']) in [20, 21, 22]])),
              'std_pp': float(np.std([float(r['accuracy_percent']) for r in records if int(r['gallery_count']) == s['gallery_count'] and r['algorithm'] == s['algorithm'] and r['profile'] == s['profile'] and int(r['seed']) in [20, 21, 22]], ddof=1))}
             for s in summaries if s['profile'] in ['baseline', 'selected']]
    # Recompute differences for the common original-three-seed comparison.
    for row in pilot:
        base = next(s for s in pilot if s['gallery_count'] == row['gallery_count'] and s['algorithm'] == row['algorithm'] and s['profile'] == 'baseline')
        row['baseline_change_pp'] = row['mean_percent']-base['mean_percent']
        row['paper_difference_pp'] = row['mean_percent']-row['paper_percent']
        for key in ['min_percent', 'max_percent', 'improved', 'tied', 'declined']:
            del row[key]  # Do not carry ten-split counts into a three-split table.
    write_csv(root/'original_three_seed_comparison.csv', pilot)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, count in zip(axes, CASES):
        x = np.arange(2)
        for offset, profile, label in [(-.19, 'baseline', 'Before (10 splits)'), (.19, 'selected', 'Gallery-selected (10 splits)')]:
            items = [next(s for s in summaries if s['gallery_count'] == count and s['algorithm'] == a and s['profile'] == profile) for a in ['pca', 'csa']]
            ax.bar(x+offset, [s['mean_percent'] for s in items], .38, yerr=[s['std_pp'] for s in items], capsize=3, label=label)
        ax.scatter(x, [CASES[count]['paper'][a] for a in ['pca', 'csa']], marker='_', s=450, color='black', label='Paper best result')
        ax.set_xticks(x, ['PCA', 'CSA'])
        ax.set_title(f'G{count}/P{500-count}; paper dimensions')
        ax.set_ylim(80, 96)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Top-one accuracy (%)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Shared input/NN; ordinary CSA, Tmax=3; mean +/- split SD')
    fig.tight_layout()
    fig.savefig(root/'comparison.png', dpi=180)
    plt.close(fig)
    # Descriptive confusion counts reuse the same5000 images across overlapping splits.
    confusion = {}
    for count in CASES:
        for algorithm in ['pca', 'csa']:
            for profile in ['baseline', 'selected']:
                matrix = np.zeros((10, 10), dtype=int)
                for r in records:
                    if int(r['gallery_count']) == count and r['algorithm'] == algorithm and r['profile'] == profile:
                        with np.load(root/'evaluation'/r['prediction_file']) as stored:
                            np.add.at(matrix, (stored['true'], stored['predicted']), 1)
                confusion[f'G{count}_{algorithm}_{profile}'] = matrix.tolist()
    dump(root/'confusion_counts.json', {'role': 'post-hoc descriptive repeated prediction counts, not independent images', 'matrices': confusion})
    print(json.dumps([s for s in summaries if s['profile'] in ['baseline', 'selected']], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'features', 'select', 'evaluate', 'verify', 'analyze'], required=True)
    parser.add_argument('--output', type=Path, default=BASE/'results/mnist_protocol_optimized')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.is_relative_to(BASE/'results'), 'Output must remain inside project results'
    tick = time.perf_counter()
    with threadpool_limits(2):
        if args.stage == 'select':
            select(root, args.seeds)
        else:
            {'prepare': prepare, 'features': prepare_features, 'evaluate': evaluate,
             'verify': verify, 'analyze': analyze}[args.stage](root)
    print(f'DONE stage={args.stage} elapsed={time.perf_counter()-tick:.1f}s', flush=True)


if __name__ == '__main__':
    main()
