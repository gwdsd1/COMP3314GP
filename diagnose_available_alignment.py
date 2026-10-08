"""Resize-only control with matching seeds and a fixed protocol to diagnose the new eye-alignment preprocessing combination.

This control does not meet the instructor's two-eye alignment requirement; it explains discrepancies and does not replace the main experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from csa import PCA
from gabor import cached_features
from run_orl_gray import load_orl,write_csv
from run_available_reproduction import SEEDS,ORL_CASES,split,record_recognition


def main(out):
    base=Path(__file__).resolve().parent;out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    gray,labels,names,digest=load_orl(base/'data/raw/orl')
    (out/'protocol.json').write_text(json.dumps({'scope':'resize-only sensitivity control; no new parameter selection',
                                                'seeds':SEEDS,'raw_sha256':digest,
                                                'preprocessing':'Pillow bilinear + equalize; no eye alignment',
                                                'fixed_profiles':'same as available reproduction, no selection from probe'},indent=2),encoding='utf-8')
    gabor=cached_features(gray,base/'data/processed/available/orl_resize_gabor.npy',2)
    scores=[];predictions=[];curves=[];histories=[]
    for seed in SEEDS:
        train,probe=split(labels,2,seed)
        if seed==SEEDS[0]:
            aligned=np.load(base/'data/processed/available/orl_aligned.npy')
            for center in [False,True]:
                a=PCA(30,center=center).fit(aligned[train]);b=PCA(30,center=center,solver='gram').fit(aligned[train])
                residual=np.linalg.norm(a.basis@a.basis.T-b.basis@b.basis.T)
                assert residual<1e-6
                print('Actual ORL SVD/Gram projector error',center,residual,flush=True)
        for profile in ['euclidean_control','group_primary']:
            for case in ORL_CASES:
                record_recognition('ORL resize-only',seed,profile,case,gray if case[1]=='gray' else gabor,
                                   labels,names,train,probe,scores,predictions,curves,histories)
    write_csv(out/'fixed_results.csv',scores);write_csv(out/'predictions.csv',predictions)
    summary=[]
    for profile,method in dict.fromkeys((r['profile'],r['method']) for r in scores):
        values=[r['accuracy_percent'] for r in scores if r['profile']==profile and r['method']==method]
        summary.append({'profile':profile,'method':method,'mean_percent':float(np.mean(values)),
                        'std_pp':float(np.std(values,ddof=1))})
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results/alignment_diagnosis')
    with threadpool_limits(limits=4):main(parser.parse_args().output)
