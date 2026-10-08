"""补查迭代上限：停机规则会在第三轮停止，因此另强制执行十轮。"""
import json
import shutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from csa import CSA
from gabor import file_hash
from run_available_reproduction import split,transform_batches,nn
from run_orl_gray import write_csv
from run_feret import read_csv

BASE=Path(__file__).resolve().parent


def run():
    root=BASE/'results/orl_gabor_forced10';root.mkdir(parents=True,exist_ok=False)
    snap=root/'source_snapshot';snap.mkdir()
    for name in ['diagnose_orl_iterations.py','csa.py']:shutil.copy2(BASE/name,snap/name)
    protocol={'reason':'Tmax10 with original stopping rule actually stopped at iteration3; this follow-up forces 10 sweeps',
              'branches':['eyes_only','paper_description'],'seeds':[20,21,22],'ranks':[15,1,26],'center':False,
              'metric':'cosine','max_iter':10,'stop_on_convergence':False,'feature':'baseline',
              'status':'development single-factor diagnostic, not paper Tmax3',
              'sources':{n:file_hash(snap/n) for n in ['diagnose_orl_iterations.py','csa.py']}}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    y=np.repeat(np.arange(1,41),10);names=[r['file'] for r in read_csv(BASE/'data/annotations/orl_eye_coordinates_v2.csv')]
    old=read_csv(BASE/'results/orl_gabor_diagnosis/predictions.csv');reference=json.loads((BASE/'results/orl_gabor_diagnosis/csa_history.json').read_text())
    rows=[];predictions=[];histories=[]
    for branch in protocol['branches']:
        path=BASE/'results/orl_gabor_diagnosis/features'/f'{branch}_baseline.npy';x=np.load(path,mmap_mode='r')
        for seed in protocol['seeds']:
            train,probe=split(y,2,seed);m=CSA((15,1,26),max_iter=10,stop_on_convergence=False).fit(x[train])
            assert len(m.history)==10
            pred,neighbor,distance=nn(transform_batches(m,x,train),transform_batches(m,x,probe),y[train],'cosine')
            ref=next(r for r in reference if r['branch']==branch and r['seed']==seed and r['feature_variant']=='baseline' and r['method']=='CSA-3' and r['model_variant']=='default')
            first=m.history[2]['sse'];assert np.isclose(first,ref['history'][-1]['sse'],rtol=1e-12)
            error=[m.initial_sse]+[s for h in m.history for s in h['mode_sse']];assert np.all(np.diff(error)<=m.initial_sse*1e-9)
            histories.append({'branch':branch,'seed':seed,'history':m.history,'feature_sha256':file_hash(path)})
            previous={r['probe_file']:int(r['predicted_label']) for r in old if r['branch']==branch and r['seed']==str(seed) and r['feature_variant']=='baseline' and r['method']=='CSA-3' and r['model_variant']=='default'}
            rows.append({'branch':branch,'seed':seed,'accuracy_percent':100*float(np.mean(pred==y[probe])),
                         'changed_predictions':sum(int(p)!=previous[names[i]] for i,p in zip(probe,pred)),
                         'relative_sse_reduction_after3':(first-m.history[-1]['sse'])/first,'actual_iterations':10})
            for i,p,j,d in zip(probe,pred,neighbor,distance):
                predictions.append({'branch':branch,'seed':seed,'probe_index':int(i),'probe_file':names[i],
                                    'true_label':int(y[i]),'predicted_label':int(p),'neighbor_index':int(train[j]),
                                    'neighbor_file':names[train[j]],'correct':int(p==y[i]),'distance':float(d)})
            print(rows[-1],flush=True)
    write_csv(root/'fixed_results.csv',rows);write_csv(root/'predictions.csv',predictions)
    (root/'csa_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    saved=read_csv(root/'predictions.csv');assert len(saved)==1920
    for row in read_csv(root/'fixed_results.csv'):
        rs=[r for r in saved if r['branch']==row['branch'] and r['seed']==row['seed']];train,probe=split(y,2,int(row['seed']))
        assert {int(r['probe_index']) for r in rs}==set(probe)
        for r in rs:
            i,j=int(r['probe_index']),int(r['neighbor_index']);assert j in train and int(r['predicted_label'])==y[j] and int(r['correct'])==int(y[i]==y[j])
        assert abs(100*sum(int(r['correct']) for r in rs)/320-float(row['accuracy_percent']))<1e-10
    (root/'verification.json').write_text(json.dumps({'passed':True,'configurations':6,'predictions':1920,
        'checks':['ten actual sweeps and monotone SSE','third-sweep SSE matches prior run','all predictions and summary accuracy reread'],
        'limits':'development diagnostic; not independent blind evaluation'},indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
