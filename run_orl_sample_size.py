"""Fixed-probe ORL learning curves; an explicitly labelled protocol extension.

No search or test-based selection: seeds, features, ranks, distance and alpha
are frozen before scoring. Existing G2/P8 experiments are not overwritten.
"""
import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA, PCA
from gabor import file_hash
from optimize_csa_lda import fisher, coordinates
from run_available_reproduction import transform_batches
from run_orl_gray import write_csv

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'results/orl_sample_size'
LABELS = np.repeat(np.arange(1, 41), 10)
SEEDS = list(range(20, 30))
ALPHA = .15
POLICY = {'scaling': 'unit_columns', 'metric': 'cosine', 'lda_center': True}
INPUT_ID = 'yunet_d20_replicate_fixed39_zero'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def splits(seed):
    rng = np.random.default_rng(seed)
    pool, probe = [], []
    for c in range(1, 41):
        order = rng.permutation(np.flatnonzero(LABELS == c))
        pool.append(order[:5])
        probe.extend(order[5:])
    pool, probe = np.array(pool), np.array(probe)
    return pool, probe


def scatter(z, labels):
    mean = z.mean(0)
    sw, sb = np.zeros((z.shape[1], z.shape[1])), np.zeros((z.shape[1], z.shape[1]))
    for c in np.unique(labels):
        a = z[labels == c]
        delta = a - a.mean(0)
        sw += delta.T @ delta
        d = a.mean(0) - mean
        sb += len(a) * np.outer(d, d)
    return sw, sb


def energy(z, labels):
    z = np.asarray(z, dtype=float).reshape(len(z), -1)
    total = float(np.sum((z-z.mean(0))**2))
    between = sum(len(z[labels == c])*float(np.sum((z[labels == c].mean(0)-z.mean(0))**2))
                  for c in np.unique(labels))
    return total, between


def score(folder, records, statistics, seed, gcount, context, name, a, b, train, query, sw_condition):
    distances = cdist(b, a, 'cosine')
    assert np.isfinite(distances).all()
    nearest = np.argmin(distances, axis=1)
    pred = LABELS[train[nearest]]
    stem = f's{seed}_g{gcount}_{context}_{name}'
    np.savez_compressed(folder/'predictions'/f'{stem}.npz', train=train, query=query,
                        gallery=a, probe=b, nearest=nearest, predictions=pred)
    correct = int(np.sum(pred == LABELS[query]))
    records.append({'seed': seed, 'gallery_per_class': gcount, 'context': context,
                    'method': name, 'correct': correct, 'probe_count': len(query),
                    'accuracy_percent': 100*correct/len(query), 'prediction_file': stem+'.npz'})
    # Post-score diagnosis only: probe labels never fit a model or select a setting.
    train_sw, _ = scatter(a, LABELS[train])
    probe_sw, _ = scatter(b, LABELS[query])
    tr = np.trace(train_sw)/(len(train)-40)
    pr = np.trace(probe_sw)/(len(query)-40)
    statistics.append({'seed': seed, 'gallery_per_class': gcount, 'context': context,
                       'method': name, 'pre_sw_condition': sw_condition,
                       'probe_to_train_within_variance': float(pr/tr)})


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['context'], row['method'], row['gallery_per_class'])].append(row['accuracy_percent'])
    return [{'context': context, 'method': method, 'gallery_per_class': g,
             'n_splits': len(v), 'mean_percent': float(np.mean(v)), 'std_pp': float(np.std(v, ddof=1))}
            for (context, method, g), v in groups.items()]


def plot(folder, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name in dict.fromkeys(r['method'] for r in summary):
        rows = sorted([r for r in summary if r['method'] == name and r['context'] == 'fixed_P5'],
                      key=lambda r: r['gallery_per_class'])
        ax.errorbar([r['gallery_per_class'] for r in rows], [r['mean_percent'] for r in rows],
                    yerr=[r['std_pp'] for r in rows], marker='o', capsize=3, label=name)
    ax.set(xlabel='Training/gallery images per identity', ylabel='Accuracy (%)', xticks=[2, 3, 5],
           title='ORL: identical P5 within each seed; fixed settings (mean ± sample SD)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder/'learning_curve.png', dpi=180)
    plt.close(fig)


def run(diagnostic=False):
    if diagnostic:
        protocol = read(ROOT/'protocol.json')
        assert read(ROOT/'verification.json')['passed']
        folder = ROOT/'pca_control'
        folder.mkdir(exist_ok=False)
        dump(folder/'protocol.json', {'role': 'post-main fixed equal-budget PCA control; no search',
                                     'PCA_components': 24, 'LDA_components': 23,
                                     'alpha': ALPHA, 'centering': True, 'parent_protocol': 'protocol.json'})
    else:
        ROOT.mkdir(exist_ok=False)
        folder = ROOT
        entry = next(e for e in read(BASE/'results/orl_protocol_audit/catalogue.json') if e['id'] == INPUT_ID)
        protocol = {'role': 'sample-size extension, not replacement for paper G2/P8', 'seeds': SEEDS,
                    'input': entry, 'feature_normalization': 'none', 'gallery_sizes': [2, 3, 5],
                    'probe': 'last5 of per-class seeded permutation; identical for G2/G3/G5',
                    'gallery': 'nested first2/3/5 from the other5; unused candidates do not train',
                    'anchor': 'G2/P8 with same first2; includes fixedP5 plus unused3',
                    'csa_ranks': [[15, 1, 26], [4, 2, 3]], 'CSA_center': False,
                    'Tmax': 3, 'init': 'identity', 'LDA_dimension': 23, 'policy': POLICY,
                    'alpha': ALPHA, 'alpha_selection': 'fixed illustrative .15; not selected in this study',
                    'feature_choice': 'one existing shared input fixed for every seed/method/G; no new search',
                    'pure_CSA_origin': 'subtract gallery core mean before cosine, also for390; differs from old raw cosine',
                    'limitations': 'historical probes have been seen; continued development, not a blind evaluation',
                    'source_sha256': {n: file_hash(BASE/n) for n in
                                       ['run_orl_sample_size.py', 'csa.py', 'optimize_csa_lda.py']}}
        dump(ROOT/'protocol.json', protocol)
        (ROOT/'source_snapshot').mkdir()
        for n in ['run_orl_sample_size.py', 'csa.py', 'optimize_csa_lda.py']:
            shutil.copy2(BASE/n, ROOT/'source_snapshot'/n)
    (folder/'predictions').mkdir()
    (folder/'models').mkdir()
    feature_path = BASE/protocol['input']['gabor_path']
    assert file_hash(feature_path) == protocol['input']['gabor_sha256']
    data = np.load(feature_path, mmap_mode='r')
    assert data.shape == (400, 28, 23, 40) and np.isfinite(data).all()
    records, statistics, model_stats, split_records = [], [], [], []
    start = time.perf_counter()
    for seed in SEEDS:
        pool, probe = splits(seed)
        split_records.append({'seed': seed, 'gallery_pool5': pool.tolist(), 'fixed_probe5': probe.tolist()})
        for gcount in [2, 3, 5]:
            train = pool[:, :gcount].ravel()
            assert not np.intersect1d(train, probe).size
            unused = pool[:, gcount:].ravel()
            query = np.r_[probe, unused] if gcount == 2 else probe
            original_energy = energy(data[train], LABELS[train])
            cases = [('PCA24', None)] if diagnostic else [('CSA390', (15, 1, 26)), ('CSA24', (4, 2, 3))]
            for name, ranks in cases:
                model = PCA(24, center=True, solver='gram') if diagnostic else CSA(ranks, center=False, init='identity', max_iter=3)
                model.fit(data[train])
                a, b = transform_batches(model, data, train), transform_batches(model, data, query)
                core_energy = energy(a, LABELS[train])
                model_stats.append({'seed': seed, 'gallery_per_class': gcount, 'model': name,
                                    'total_variance_retained': core_energy[0]/original_energy[0],
                                    'between_variance_retained': core_energy[1]/original_energy[1]})
                arrays = {'train': train, 'query': query, 'core_gallery': a, 'core_probe': b, 'mean': model.mean}
                if diagnostic:
                    arrays['pca_basis'] = model.basis
                else:
                    arrays.update({f'axis{k}': u for k, u in enumerate(model.bases)})
                    dump(folder/'models'/f's{seed}_g{gcount}_{name}_history.json', model.history)
                mean = a.mean(0)
                outputs = [(name, a-mean, b-mean, '')]
                if name != 'CSA390':
                    ordinary = fisher(a, LABELS[train], 0., check=True)
                    eigen = eigh(ordinary.sw, eigvals_only=True)
                    cond = float(eigen[-1]/eigen[0])
                    arrays['sw'], arrays['sb'] = ordinary.sw, ordinary.sb
                    for alpha, suffix in [(0., '_LDA23'), (ALPHA, '_shrink15_LDA23')]:
                        lda = fisher(a, LABELS[train], alpha, check=True, base=ordinary)
                        ga, qb = coordinates(a, b, lda, POLICY)
                        basis = lda.basis/np.linalg.norm(lda.basis, axis=0)
                        arrays[suffix+'_basis'] = basis[:, :23]
                        arrays[suffix+'_eigenvalues'] = lda.eigenvalues[:23]
                        outputs.append((name+suffix, ga[:, :23], qb[:, :23], cond))
                np.savez_compressed(folder/'models'/f's{seed}_g{gcount}_{name}.npz', **arrays)
                for out_name, ga, qb, cond in outputs:
                    score(folder, records, statistics, seed, gcount, 'fixed_P5', out_name, ga, qb[:200], train, probe, cond)
                    if gcount == 2:
                        score(folder, records, statistics, seed, gcount, 'anchor_P8', out_name, ga, qb, train, query, cond)
            print(f'{"PCA control" if diagnostic else "Main"} seed={seed} G{gcount}/P5 complete ({time.perf_counter()-start:.1f}s)', flush=True)
            write_csv(folder/'results.csv', records)
        dump(folder/'splits.json', split_records)
    summary = aggregate(records)
    dump(folder/'summary.json', summary)
    write_csv(folder/'statistics.csv', statistics)
    write_csv(folder/'representation_statistics.csv', model_stats)
    plot(folder, summary)
    dump(folder/'execution.json', {'elapsed_seconds': time.perf_counter()-start, 'scores': len(records),
                                 'predictions': sum(r['probe_count'] for r in records)})
    verify(folder)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def verify(folder):
    """Independent tensor contraction, scatter, Fisher equation and cosine NN."""
    protocol = read(ROOT/'protocol.json')
    data = np.load(BASE/protocol['input']['gabor_path'], mmap_mode='r')
    model_count, prediction_count = 0, 0
    max_residual, max_update_error = 0., 0.
    for path in sorted((folder/'models').glob('*.npz')):
        m = np.load(path)
        train, query = m['train'], m['query']
        if 'pca_basis' in m:
            encode = lambda x: (np.asarray(x).reshape(len(x), -1)-m['mean'])@m['pca_basis']
        else:
            encode = lambda x: np.einsum('nhwc,ha,wb,cd->nabd', np.asarray(x, dtype=float)-m['mean'],
                                        m['axis0'], m['axis1'], m['axis2'], optimize=True).reshape(len(x), -1)
            for k in range(3):
                u = m[f'axis{k}']
                np.testing.assert_allclose(u.T@u, np.eye(u.shape[1]), atol=1e-10)
            # Independently replay Procedure 1: direct tensor contractions and
            # per-object scatter sums, without csa.mode_product/unfold/fit.
            learning = np.asarray(data[train], dtype=float)-m['mean']
            ranks = [m[f'axis{k}'].shape[1] for k in range(3)]
            bases = [np.eye(size, rank) for size, rank in zip(learning.shape[1:], ranks)]
            for _ in range(3):
                for k, expression in enumerate(['nhwc,wb,cd->nhbd', 'nhwc,ha,cd->nwad', 'nhwc,ha,wb->ncab']):
                    others = [bases[j] for j in range(3) if j != k]
                    unfolded = np.einsum(expression, learning, *others, optimize=True)
                    covariance = sum(obj.reshape(obj.shape[0], -1)@obj.reshape(obj.shape[0], -1).T
                                     for obj in unfolded)
                    _, vectors = eigh(covariance)
                    bases[k] = vectors[:, -ranks[k]:][:, ::-1]
            for k, reference in enumerate(bases):
                actual = m[f'axis{k}']
                error = float(np.max(np.abs(reference@reference.T-actual@actual.T)))
                assert error < 1e-8
                max_update_error = max(max_update_error, error)
        for indices, key in [(train, 'core_gallery'), (query, 'core_probe')]:
            np.testing.assert_allclose(encode(data[indices]), m[key], atol=1e-9, rtol=1e-9)
        if 'sw' in m:
            sw, sb = scatter(m['core_gallery'], LABELS[train])
            np.testing.assert_allclose(sw, m['sw'], atol=1e-7, rtol=1e-10)
            np.testing.assert_allclose(sb, m['sb'], atol=1e-7, rtol=1e-10)
            assert m['core_gallery'].shape[1] <= len(train)-40
            for alpha, suffix in [(0., '_LDA23'), (ALPHA, '_shrink15_LDA23')]:
                matrix = (1-alpha)*sw+alpha*np.trace(sw)/len(sw)*np.eye(len(sw))
                w = m[suffix+'_basis']
                res = np.linalg.norm(sb@w-(matrix@w)*m[suffix+'_eigenvalues'])/max(1, np.linalg.norm(sb@w))
                assert res < 1e-8
                max_residual = max(max_residual, float(res))
        model_count += 1
    with (folder/'results.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    expected_methods = 3 if folder.name == 'pca_control' else 4
    assert len(rows) == 10*4*expected_methods
    for row in rows:
        d = np.load(folder/'predictions'/row['prediction_file'])
        pool, probe = splits(int(row['seed']))
        train = pool[:, :int(row['gallery_per_class'])].ravel()
        np.testing.assert_array_equal(train, d['train'])
        expected_query = np.r_[probe, pool[:, 2:].ravel()] if row['context'] == 'anchor_P8' else probe
        np.testing.assert_array_equal(expected_query, d['query'])
        assert not np.intersect1d(train, d['query']).size
        a, b = d['gallery'], d['probe']
        distance = 1-(b/np.linalg.norm(b, axis=1)[:, None])@(a/np.linalg.norm(a, axis=1)[:, None]).T
        near = np.argmin(distance, axis=1)
        np.testing.assert_array_equal(near, d['nearest'])
        np.testing.assert_array_equal(LABELS[train[near]], d['predictions'])
        assert int(np.sum(d['predictions'] == LABELS[d['query']])) == int(row['correct'])
        prediction_count += len(near)
    dump(folder/'verification.json', {'passed': True, 'models_checked': model_count,
                                     'scores_checked': len(rows), 'predictions_checked': prediction_count,
                                     'maximum_relative_Fisher_residual': max_residual,
                                     'maximum_CSA_update_projector_error': max_update_error,
                                     'checks': ['independent tensor/PCA encoding', 'orthogonality', 'scatter sums',
                                                'independent three-sweep CSA Procedure 1', 'Fisher equation', 'nested splits and no train/probe overlap',
                                                'independent cosine NN and CSV scores']})


def analyze():
    """Summarize paired changes after both fixed experiments are complete."""
    folders = [ROOT, ROOT/'pca_control']
    rows, statistics, representation = [], [], []
    for folder in folders:
        assert read(folder/'verification.json')['passed']
        for filename, target in [('results.csv', rows), ('statistics.csv', statistics),
                                 ('representation_statistics.csv', representation)]:
            with (folder/filename).open(encoding='utf-8-sig', newline='') as f:
                target.extend(csv.DictReader(f))
    scores = {(int(r['seed']), int(r['gallery_per_class']), r['method']): float(r['accuracy_percent'])
              for r in rows if r['context'] == 'fixed_P5'}
    paired = []
    for method in dict.fromkeys(r['method'] for r in rows):
        for small, large in [(2, 3), (2, 5), (3, 5)]:
            delta = [scores[s, large, method]-scores[s, small, method] for s in SEEDS]
            paired.append({'method': method, 'from_G': small, 'to_G': large,
                           'mean_change_pp': float(np.mean(delta)), 'improved_splits': sum(d > 0 for d in delta),
                           'tied_splits': sum(d == 0 for d in delta), 'std_change_pp': float(np.std(delta, ddof=1))})
    shrinkage = []
    for stem in ['CSA24', 'PCA24']:
        for g in [2, 3, 5]:
            delta = [scores[s, g, stem+'_shrink15_LDA23']-scores[s, g, stem+'_LDA23'] for s in SEEDS]
            shrinkage.append({'model': stem, 'gallery_per_class': g, 'mean_change_pp': float(np.mean(delta)),
                              'improved_splits': sum(d > 0 for d in delta), 'tied_splits': sum(d == 0 for d in delta)})
    rep_summary, stat_summary = [], []
    for g in [2, 3, 5]:
        for name in ['CSA390', 'CSA24', 'PCA24']:
            group = [r for r in representation if int(r['gallery_per_class']) == g and r['model'] == name]
            rep_summary.append({'model': name, 'gallery_per_class': g,
                                'mean_total_variance_retained': float(np.mean([float(r['total_variance_retained']) for r in group])),
                                'mean_between_variance_retained': float(np.mean([float(r['between_variance_retained']) for r in group]))})
        for name in ['CSA24_LDA23', 'CSA24_shrink15_LDA23', 'PCA24_LDA23']:
            group = [r for r in statistics if int(r['gallery_per_class']) == g
                     and r['method'] == name and r['context'] == 'fixed_P5']
            stat_summary.append({'method': name, 'gallery_per_class': g,
                                 'median_pre_sw_condition': float(np.median([float(r['pre_sw_condition']) for r in group])),
                                 'median_probe_to_train_within_variance': float(np.median([float(r['probe_to_train_within_variance']) for r in group]))})
    value = {'paired_sample_size_changes': paired, 'paired_shrinkage_changes': shrinkage,
             'training_representation': rep_summary, 'scatter_diagnosis': stat_summary,
             'interpretation_limits': ['retained variance is not a percentage of identity information',
                                      'more samples change representation, scatter estimates and gallery coverage together',
                                      'probe statistics are post-score diagnostics, never settings selection'],
             'analysis_source_sha256': file_hash(Path(__file__))}
    dump(ROOT/'analysis.json', value)
    write_csv(ROOT/'paired_changes.csv', paired)
    print(json.dumps(value, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', default='results/orl_sample_size')
    args = parser.parse_args()
    ROOT = (BASE/args.output).resolve()
    if not ROOT.is_relative_to(BASE):
        raise ValueError('Output must remain inside this project')
    with threadpool_limits(limits=2):
        if args.verify:
            verify(ROOT)
            verify(ROOT/'pca_control')
        elif args.analyze:
            analyze()
        else:
            run(args.diagnostic)
