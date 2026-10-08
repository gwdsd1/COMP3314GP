"""Read back every Gabor diagnostic prediction, training scale, cache, and baseline rather than checking means alone."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from gabor import file_hash
from run_available_reproduction import split
from run_feret import read_csv

BASE=Path(__file__).resolve().parent


def verify(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());y=np.repeat(np.arange(1,41),10)
    names=[r['file'] for r in read_csv(BASE/'data/annotations/orl_eye_coordinates_v2.csv')]
    for source,digest in protocol['sources'].items():assert file_hash(root/'source_snapshot'/source)==digest
    assert file_hash(BASE/'results/orl_branches_v3/protocol.json')==protocol['stage1_protocol_sha256']
    assert len(protocol['features'])==12 and max(r['max_abs_error'] for r in protocol['convolution_checks'])<2e-6
    manifests=json.loads((root/'feature_manifests.json').read_text());data={}
    for m in manifests:
        assert file_hash(root/m['path'])==m['sha256']
        gray=BASE/'data/processed/orl_branches_v2'/f'{m["branch"]}_seed20.npy'
        assert file_hash(gray)==m['gray_sha256']
        x=np.load(root/m['path'],mmap_mode='r');assert x.shape==(400,28,23,40) and np.isfinite(x).all()
        data[m['branch'],m['variant']]=x
    assert len(manifests)==24
    for branch in protocol['branches']:
        x=data[branch,'baseline'];gray=np.load(BASE/'data/processed/orl_branches_v2'/f'{branch}_seed20.npy')
        import hashlib
        digest=hashlib.sha256(gray.tobytes()).hexdigest()
        assert np.array_equal(x,np.load(BASE/'data/processed/orl_branches_v2'/f'gabor_{digest[:16]}.npy'))
        jet=data[branch,'jet_l2'];assert np.max(np.abs(np.linalg.norm(jet,axis=-1)-1))<1e-6
        tensor=data[branch,'tensor_l2'];assert np.max(np.abs(np.linalg.norm(tensor.reshape(400,-1),axis=1)-1))<1e-6
        assert np.array_equal(data[branch,'sqrt_response'],np.sqrt(x))
        assert np.array_equal(data[branch,'train_channel_rms'],x)
        from diagnose_orl_gabor import filtered, specification
        # The parameterized FFT must also reproduce the original filtering at baseline settings, beyond small-image convolution checks.
        assert np.max(np.abs(filtered(gray[:4],specification('baseline'))-x[:4]))<2e-5
    scales=json.loads((root/'training_scales.json').read_text());assert len(scales)==6
    for row in scales:
        train,_=split(y,2,row['seed']);assert train.tolist()==row['train_indices']
        scale=np.sqrt(np.mean(data[row['branch'],'baseline'][train].astype(np.float64)**2,axis=(0,1,2)))
        assert np.allclose(scale,row['scale'],rtol=0,atol=1e-12)
    key=lambda r:(r['branch'],r['feature_variant'],int(r['seed']),r['method'],r['model_variant'])
    rows=read_csv(root/'fixed_results.csv');preds=read_csv(root/'predictions.csv');groups=defaultdict(list)
    for r in preds:groups[key(r)].append(r)
    assert len(rows)==252 and len(preds)==80640 and len(groups)==252
    for row in rows:
        k=key(row);train,probe=split(y,2,k[2]);ps=groups[k]
        assert len(ps)==320 and {int(r['probe_index']) for r in ps}==set(probe)
        correct=0
        for r in ps:
            i=int(r['probe_index']);j=int(r['neighbor_index']);assert j in train
            assert r['probe_file']==names[i] and r['neighbor_file']==names[j]
            assert int(r['true_label'])==y[i] and int(r['predicted_label'])==y[j]
            assert int(r['correct'])==int(y[i]==y[j]);correct+=int(r['correct'])
            assert np.isfinite(float(r['distance'])) and float(r['distance'])>=-1e-12
        assert correct==int(row['correct']) and int(row['total'])==320
        assert abs(float(row['accuracy_percent'])-100*correct/320)<1e-10
        assert int(row['Tmax'])==(10 if k[4]=='iterations10' else 3)
    summary=json.loads((root/'summary.json').read_text())['summary'];assert len(summary)==84
    for s in summary:
        rs=[r for r in rows if (r['branch'],r['feature_variant'],r['method'],r['model_variant'])==
            (s['branch'],s['feature_variant'],s['method'],s['model_variant'])]
        assert {int(r['seed']) for r in rs}=={20,21,22}
        values=[float(r['accuracy_percent']) for r in rs]
        assert abs(np.mean(values)-s['mean_percent'])<1e-10 and abs(np.std(values,ddof=1)-s['std_pp'])<1e-10
    previous=read_csv(BASE/'results/orl_branches_v3/predictions.csv')
    old={(r['branch'],int(r['seed']),r['method'],r['probe']):r for r in previous}
    matched=0
    for r in preds:
        if r['feature_variant']=='baseline' and r['model_variant']=='default':
            reference=old[r['branch'],int(r['seed']),r['method'],r['probe_file']]
            assert int(r['predicted_label'])==int(reference['predicted'])
            assert r['neighbor_file']==reference['nearest_gallery']
            matched+=1
    assert matched==5760
    histories=json.loads((root/'csa_history.json').read_text());assert len(histories)==180
    for record in histories:
        errors=[record['initial_sse']]+[v for h in record['history'] for v in h['mode_sse']]
        assert np.all(np.diff(errors)<=max(1,errors[0])*1e-9)
        assert 1<=len(record['history'])<=(10 if record['model_variant']=='iterations10' else 3)
    result={'passed':True,'counts':{'configurations':252,'predictions':80640,'summary_groups':84,'feature_arrays':24,
                                  'CSA_histories':180,'baseline_predictions_matched':matched},
            'checks':['source and feature hashes, fixed dimensions and 3/10 iteration limit',
                      'six independent spatial-convolution checks and parameterized baseline replay; finite arrays and per-image normalization',
                      'channel RMS computed only from 80 training images',
                      'all 80640 predictions, gallery membership, mean and sample SD',
                      'stage1 baseline predictions exactly retained; CSA SSE monotone'],
            'limits':'development diagnostics; unreported protocols remain uncertain; no blind winner selection'}
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'results/orl_gabor_diagnosis');verify(p.parse_args().output)
