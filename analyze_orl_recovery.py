"""Supplementary ORL recovery analysis; no new fits or candidate changes."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gabor import file_hash

BASE = Path(__file__).resolve().parent
ROOT = BASE/'results/orl_recovery_20261027'


def rows(path):
    with path.open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    models = rows(ROOT/'models.csv')
    scores = rows(ROOT/'results.csv')
    summary = rows(ROOT/'summary.csv')
    protocol = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    for name, sha in protocol['source_sha256'].items():
        assert file_hash(BASE/name)==sha
    for name, sha in json.loads((ROOT/'input_freeze.json').read_text(encoding='utf-8')).items():
        assert file_hash(ROOT/name)==sha
    selected=[]
    for geometry in ['d20_y24','d18_y25','d22_y22']:
        for seed in range(20,30):
            # This decision depends only on fitting reconstruction SSE.
            chosen=min([r for r in models if r['input']==geometry and int(r['seed'])==seed
                        and r['start']!='PCA24'],key=lambda r:float(r['gallery_sse']))
            result=next(r for r in scores if r['input']==geometry and int(r['seed'])==seed
                        and r['start']==chosen['start'] and r['policy']==protocol['primary_policy'])
            selected.append(dict(input=geometry,seed=seed,start=chosen['start'],
                                 gallery_sse=float(chosen['gallery_sse']),
                                 accuracy_percent=float(result['accuracy_percent'])))
    with (ROOT/'gallery_sse_selected.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    sse_summary=[]
    for geometry in ['d20_y24','d18_y25','d22_y22']:
        values=np.array([r['accuracy_percent'] for r in selected if r['input']==geometry])
        sse_summary.append(dict(input=geometry,mean_percent=float(values.mean()),
             std_pp=float(values.std(ddof=1)),min_percent=float(values.min()),max_percent=float(values.max())))
    target=lambda r:r['input']=='d20_y24' and r['policy']==protocol['primary_policy']
    rect=sorted([r for r in scores if target(r) and r['start']=='rectangular_identity'],key=lambda r:int(r['seed']))
    full=sorted([r for r in scores if target(r) and r['start']=='full_identity'],key=lambda r:int(r['seed']))
    change=np.array([float(a['accuracy_percent'])-float(b['accuracy_percent']) for a,b in zip(full,rect)])
    csa=[r for r in scores if r['start']!='PCA24']
    primary=[r for r in summary if r['policy']==protocol['primary_policy']]
    report=dict(
        source_sha256=file_hash(BASE/'analyze_orl_recovery.py'),
        gallery_SSE_selected_initialization=sse_summary,
        full_minus_rectangular_same_input=dict(mean_change_pp=float(change.mean()),
            wins=int(np.sum(change>0)),ties=int(np.sum(change==0)),losses=int(np.sum(change<0))),
        single_maximum_candidates=[r for r in csa if float(r['accuracy_percent'])==max(float(x['accuracy_percent']) for x in csa)],
        best_rank_conforming_start=max([r for r in summary if r['start'] in ['rectangular_identity','hosvd']],key=lambda r:float(r['mean_percent'])),
        best_weighted_metric=max([r for r in summary if r['start']!='PCA24' and not r['policy'].endswith('_g0')],key=lambda r:float(r['mean_percent'])),
        limits=['Input and metric are not selected by gallery SSE.',
                'Choosing the highest mean/maximum across candidates uses already-seen probes and is exploratory.',
                'Full-square initialization has a larger initial mode dimension than the final ranks; not an author-confirmed setting.',
                'SSE selection is an additional training-only analysis, not a preregistered outer-score selection.'])
    (ROOT/'supplementary_analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11,5))
    selected_primary=[r for r in primary if r['start']!='PCA24']
    labels=[r['input']+'\n'+r['start'].replace('_',' ') for r in selected_primary]
    ax.bar(np.arange(9),[float(r['mean_percent']) for r in selected_primary],
           yerr=[float(r['std_pp']) for r in selected_primary],capsize=3,color=['#4776a3','#709e74','#ba9360']*3)
    ax.axhline(90.9,color='#b64040',linestyle='--',label='Paper best-configuration reference: 90.9%')
    ax.set_xticks(np.arange(9),labels,rotation=35,ha='right',fontsize=8)
    ax.set_ylim(80,97);ax.set_ylabel('Accuracy (%)');ax.legend(loc='upper left',fontsize=8)
    ax.set_title('ORL G2/P8, CSA 4x2x3 + ordinary LDA23, Tmax=3\nTen observed splits 20-29; fixed shared centered-cosine NN; context extension')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(gallery_sse=sse_summary,full_gain=report['full_minus_rectangular_same_input'],
         best_rank_conforming_start=report['best_rank_conforming_start'],
         maximum=float(report['single_maximum_candidates'][0]['accuracy_percent'])),indent=2))


if __name__=='__main__':
    main()
