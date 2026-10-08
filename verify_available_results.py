"""Read back reproduction outputs: splits, all predictions, summaries, hashes, and reconstruction metrics."""

import ast
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from gabor import file_hash

BASE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row['dataset'], row['seed'], row['profile'], row['method']


def verify(output):
    protocol = json.loads((output/'protocol.json').read_text())
    source = BASE/'data/processed/available'
    checks = []
    for name, digest in protocol['code_sha256'].items():
        saved = output/'source_snapshot'/name
        assert file_hash(saved if saved.exists() else BASE/name) == digest, name
    for name, digest in protocol['prepared_data_sha256'].items():
        assert file_hash(source/name) == digest, name
    assert file_hash(BASE/'data/annotations/orl_eye_coordinates.csv') == protocol['preprocessing']['eye_annotations_sha256']
    manifests = json.loads((output/'feature_manifests.json').read_text())
    for name, manifest in manifests.items():
        array = source/Path(name).with_suffix('.npy')
        assert file_hash(array) == manifest['file_sha256']
        assert list(np.load(array, mmap_mode='r').shape) == manifest['specification']['input_shape'] + [40]
        assert file_hash(BASE/'gabor.py') == manifest['specification']['implementation_sha256']
    checks.append('run source snapshots, prepared arrays, annotations and feature hashes/shapes')

    splits = read_csv(output/'splits.csv')
    expected = {'ORL': (40,2,8), 'MNIST G30/P470': (10,30,470), 'MNIST G40/P460': (10,40,460)}
    partitions = defaultdict(dict)
    for row in splits:
        part = partitions[row['dataset'],row['seed'],row['partition']]
        assert row['file'] not in part
        part[row['file']] = int(row['label'])
    mnist_labels = np.load(source/'mnist_labels.npy')
    index_rows = read_csv(source/'mnist_source_indices.csv')
    index_labels = {f"test/{r['test_index_zero_based']}": int(r['digit']) for r in index_rows}
    assert len(index_labels)==5000 and Counter(mnist_labels.tolist())==Counter({i:500 for i in range(10)})
    assert all(int(r['digit'])==int(label) for r,label in zip(index_rows,mnist_labels))
    for (dataset,seed,part), members in partitions.items():
        nc,ng,npb = expected[dataset]
        assert len(members)==nc*(ng if part=='train_gallery' else npb)
        assert all(v==(ng if part=='train_gallery' else npb) for v in Counter(members.values()).values())
        for name,label in members.items():
            assert label == (int(name.split('/')[0][1:]) if dataset=='ORL' else index_labels[name])
        if part=='train_gallery':
            probes=partitions[dataset,seed,'probe']
            assert not members.keys() & probes.keys()
            assert len(members)+len(probes)==nc*(ng+npb)
    checks.append('balanced class counts, source labels, no train/probe overlap and complete coverage')

    predictions = read_csv(output/'predictions.csv')
    groups = defaultdict(list)
    for row in predictions:
        gallery=partitions[row['dataset'],row['seed'],'train_gallery']
        probe=partitions[row['dataset'],row['seed'],'probe']
        assert int(row['true'])==probe[row['probe']]
        assert int(row['predicted'])==gallery[row['nearest_gallery']]
        assert int(row['correct'])==int(row['true']==row['predicted'])
        assert np.isfinite(float(row['distance']))
        groups[key(row)].append(row)
    fixed = read_csv(output/'fixed_results.csv')
    for row in fixed:
        group=groups[key(row)]
        assert len(group)==len({r['probe'] for r in group})==int(row['probe_count'])
        assert sum(int(r['correct']) for r in group)==int(row['correct'])
        assert np.isclose(100*int(row['correct'])/len(group),float(row['accuracy_percent']))
        if row['lda_dim']:
            dims=ast.literal_eval(row['pre_dims'])
            assert np.prod(dims)<=40 and int(row['lda_dim'])<=min(np.prod(dims),39)
    summary=json.loads((output/'summary.json').read_text())['summary']
    for item in summary:
        values=[float(r['accuracy_percent']) for r in fixed if all(r[k]==item[k] for k in ['dataset','profile','method'])]
        assert len(values)==item['runs']==3
        assert np.isclose(np.mean(values),item['mean_percent'])
        assert np.isclose(np.std(values,ddof=1),item['std_pp'])
    checks.append('all predictions match gallery/probe labels, score counts and recomputed mean/sample SD')

    history=json.loads((output/'recognition_csa_history.json').read_text())
    assert len(history)==30
    for item in history:
        assert 1<=len(item['history'])<=3
        errors=[h['sse'] for h in item['history']]
        assert np.all(np.diff(errors)<=1e-5*max(errors))
    convergence=json.loads((output/'orl_convergence.json').read_text())
    assert len(convergence['history'])==10
    reconstruction=read_csv(output/'orl_reconstruction.csv')
    n,D=40,56*46*40
    for row in reconstruction:
        r=int(row['dimensions'].split('x')[0])
        storage=r*(n+D) if row['method']=='PCA' else n*r**3+r*(56+46+40)
        assert storage==int(row['stored_scalars'])
        assert np.isclose(n*D/storage,float(row['compression_ratio']))
        assert np.isclose(np.sqrt(float(row['sse'])/n),float(row['paper_rmse']))
    checks.append('recognition <=3 iterations; diagnostic 10; reconstruction CR includes bases and RMSE uses N')
    counts={'fixed_results':len(fixed),'predictions':len(predictions),'splits':len(splits),
            'dimension_curves':len(read_csv(output/'dimension_curves.csv')),
            'reconstruction':len(reconstruction),'summary_groups':len(summary)}
    assert counts==dict(fixed_results=60,predictions=123120,splits=31200,dimension_curves=1164,reconstruction=80,summary_groups=20),counts
    checks.append('expected output row counts')
    for filename in ['fixed_config_results.csv','lda_dimension_curves.csv']:
        assert read_csv(BASE/'results/orl_gray_regression'/filename)==read_csv(BASE/'results/orl_gray_verified'/filename)
    checks.append('original gray entrypoint scores and all dimension curves unchanged')
    for name,digest in json.loads((BASE/'results/available_reproduction/protocol.json').read_text())['code_sha256'].items():
        assert file_hash(BASE/'results/available_reproduction/source_snapshot'/name)==digest
    checks.append('first-run code snapshots retained and match first-run hashes')
    result={'status':'passed','counts':counts,'checks':checks,
            'limits':'checks establish internal consistency, not identity with original undisclosed protocol'}
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'results/available_reproduction_v2')
    verify(parser.parse_args().output)
