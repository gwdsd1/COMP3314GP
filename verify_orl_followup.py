"""Read back ORL follow-up outputs: branch provenance, selection objectives, all predictions, and Gabor single-factor results."""
import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from gabor import file_hash
from run_feret import read_csv
from run_available_reproduction import split

BASE=Path(__file__).resolve().parent


def verify_branches(output):
    root=Path(output);protocol=json.loads((root/'protocol.json').read_text());meta=protocol['preprocessing']
    assert protocol['selected_by_probe'] is False
    if 'teacher_quality_rule' in protocol:
        rule=protocol['teacher_quality_rule'];candidate=[r for r in meta['arrays'] if r['branch']==rule['candidate']]
        accepted=all(r['support_mean']>=rule['require_mean_support'] and r['support_min']>=rule['require_min_support'] for r in candidate)
        assert accepted==rule['accepted'] and protocol['main_branches'][0]==('teacher_revised' if accepted else 'eyes_only')
    for name,digest in protocol['code_sha256'].items():assert file_hash(root/'source_snapshot'/name)==digest
    for name,key in [('orl_eye_coordinates.csv','old_eyes_sha256'),('orl_eye_coordinates_v2.csv','v2_eyes_sha256')]:
        assert file_hash(BASE/'data/annotations'/name)==meta[key]
    eye_rows=read_csv(BASE/'data/annotations/orl_eye_coordinates_v2.csv')
    eyes=np.array([[[float(r[s+'_x']),float(r[s+'_y'])] for s in ['left','right']] for r in eye_rows])
    y=np.repeat(np.arange(1,41),10);names=[r['file'] for r in eye_rows];name_index={n:i for i,n in enumerate(names)}
    splits=read_csv(root/'splits.csv');sets=defaultdict(set)
    for r in splits:
        i=int(r['index']);assert r['file']==names[i] and int(r['label'])==y[i]
        sets[r['branch'],int(r['seed']),r['partition']].add(i)
    for r in meta['arrays']:
        array=np.load(BASE/r['path'],mmap_mode='r');assert array.shape==(400,56,46) and np.isfinite(array).all()
        assert file_hash(BASE/r['path'])==r['sha256']
        train,probe=split(y,2,r['seed'])
        assert sets[r['branch'],r['seed'],'train_gallery']==set(train) and sets[r['branch'],r['seed'],'probe']==set(probe)
        if r['branch'] in ['teacher_revised','geometry_only']:
            midpoint=np.median(eyes[train].mean(axis=1),axis=0)/2
            spacing=np.median(np.sqrt(np.sum((eyes[train,1]-eyes[train,0])**2,axis=1)))/2
            expected=np.array([midpoint+[-spacing/2,0],midpoint+[spacing/2,0]])
            np.testing.assert_allclose(r['target_xy'],expected,atol=1e-12)
    quality=read_csv(BASE/'data/processed/orl_branches_v2/alignment_quality.csv')
    assert len(quality)==4800 and max(float(r['eye_mapping_error']) for r in quality)<1e-10
    manifests=json.loads((root/'feature_manifests.json').read_text())
    for name,m in manifests.items():
        assert file_hash(BASE/'data/processed/orl_branches_v2'/name)==m['file_sha256']
        assert m['specification']['implementation_sha256']==protocol['code_sha256']['gabor.py']
    results=read_csv(root/'fixed_results.csv');predictions=read_csv(root/'predictions.csv');grouped=defaultdict(list)
    for r in predictions:
        i,j=name_index[r['probe']],name_index[r['nearest_gallery']];key=r['branch'],int(r['seed'])
        assert i in sets[*key,'probe'] and j in sets[*key,'train_gallery']
        assert int(r['true'])==y[i] and int(r['predicted'])==y[j] and int(r['correct'])==int(y[i]==y[j])
        assert np.isfinite(float(r['distance']))
        grouped[*key,r['method']].append(r)
    expected={c[0]:c for c in protocol['fixed_cases']}
    for r in results:
        rows=grouped[r['branch'],int(r['seed']),r['method']]
        assert len(rows)==len({p['probe'] for p in rows})==320 and sum(int(p['correct']) for p in rows)==int(r['correct'])
        assert np.isclose(float(r['accuracy_percent']),100*int(r['correct'])/320)
        case=expected[r['method']]
        assert ast.literal_eval(r['pre_dims'])==tuple(case[3]) if isinstance(case[3],list) else ast.literal_eval(r['pre_dims'])==case[3]
        assert r['lda_dim']==('' if case[4] is None else str(case[4])) and float(r['paper_percent'])==case[5]
    summary=json.loads((root/'summary.json').read_text())['summary'];old=json.loads((BASE/'results/available_reproduction_v2/summary.json').read_text())['summary']
    for r in summary:
        values=[float(s['accuracy_percent']) for s in results if (s['branch'],s['method'])==(r['branch'],r['method'])]
        assert len(values)==3 and np.isclose(r['mean_percent'],np.mean(values)) and np.isclose(r['std_pp'],np.std(values,ddof=1))
        if r['branch']=='teacher_old':
            prior=next(o for o in old if o['dataset']=='ORL' and o['profile']=='group_primary' and o['method']==r['method'])
            assert prior['mean_percent']==r['mean_percent']
    histories=json.loads((root/'csa_history.json').read_text())
    for r in histories:
        assert len(r['history'])<=3
        error=[v for h in r['history'] for v in h['mode_sse']];assert np.max(np.diff(error))<=max(error)*1e-10+1e-6
    curves=read_csv(root/'dimension_curves.csv')
    assert (len(results),len(predictions),len(splits),len(summary),len(histories),len(curves))==(90,28800,6000,30,45,2829)
    verification={'passed':True,'counts':{'fixed_results':90,'predictions':28800,'split_rows':6000,'summary_groups':30,'histories':45,'curves':2829},
                  'checks':['snapshots, revised eye annotations, grayscale arrays and Gabor cache hashes','train-only median target replay and exact two-eye mapping',
                            'same 2/8 split, all predictions, fixed paper dimensions, mean and sample SD','all six old teacher baseline means exactly preserved','three-iteration monotone CSA'],
                  'limits':'not a fresh blind test; revised median geometry may perform worse; no test-selected main protocol'}
    (root/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8');print(json.dumps(verification,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--branches',type=Path,default=BASE/'results/orl_branches_v3')
    verify_branches(parser.parse_args().branches)
