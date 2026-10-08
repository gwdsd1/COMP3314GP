"""仅修正 ceil 的浮点边界误差；不改变旧 gabor.py 和其他数据集。"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from csa import CSA,PCA,LDA
from gabor import file_hash
from diagnose_orl_gabor import filtered,specification,make_kernel
from run_available_reproduction import split,transform_batches,nn,lda_coordinates
from run_orl_gray import write_csv
from run_feret import read_csv

BASE=Path(__file__).resolve().parent


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);snap=root/'source_snapshot';snap.mkdir()
    for name in ['diagnose_orl_radius_rounding.py','diagnose_orl_gabor.py','csa.py']:shutil.copy2(BASE/name,snap/name)
    s=specification('baseline');s['stable_radius']=True
    sizes=[make_kernel(4*np.sqrt(2)**i,0,s).shape[0] for i in range(5)];assert sizes==[25,35,49,69,97]
    (root/'protocol.json').write_text(json.dumps({'specification':s,'actual_kernel_sizes':sizes,
        'status':'follow-up discovered at final parameter verification; development diagnostics only',
        'sources':{n.name:file_hash(n) for n in snap.iterdir()}},indent=2),encoding='utf-8')
    y=np.repeat(np.arange(1,41),10);names=[r['file'] for r in read_csv(BASE/'data/annotations/orl_eye_coordinates_v2.csv')]
    rows=[];predictions=[];histories=[]
    for branch in ['eyes_only','paper_description']:
        gray=np.load(BASE/'data/processed/orl_branches_v2'/f'{branch}_seed20.npy');x=filtered(gray,s)
        path=root/f'{branch}_rounded_radius.npy';np.save(path,x)
        for seed in [20,21,22]:
            train,probe=split(y,2,seed)
            for method in ['CSA-3','PCA (Gabor)','CSA-3+LDA']:
                model=PCA(76,solver='gram') if method=='PCA (Gabor)' else CSA((4,2,3) if method.endswith('+LDA') else (15,1,26),max_iter=3)
                model.fit(x[train]);g=transform_batches(model,x,train);p=transform_batches(model,x,probe)
                if method.endswith('+LDA'):
                    lda=LDA().fit(g,y[train]);g=lda_coordinates(g,lda,23,'unit_columns');p=lda_coordinates(p,lda,23,'unit_columns')
                predicted,j,d=nn(g,p,y[train],'cosine');key={'branch':branch,'seed':seed,'method':method}
                rows.append({**key,'accuracy_percent':100*float(np.mean(predicted==y[probe])),'feature_sha256':file_hash(path)})
                for i,label,neighbor,distance in zip(probe,predicted,j,d):
                    predictions.append({**key,'probe_index':int(i),'probe_file':names[i],'true_label':int(y[i]),
                        'predicted_label':int(label),'neighbor_index':int(train[neighbor]),'correct':int(label==y[i]),'distance':float(distance)})
                if isinstance(model,CSA):histories.append({**key,'initial_sse':model.initial_sse,'history':model.history})
                print(rows[-1],flush=True)
    write_csv(root/'fixed_results.csv',rows);write_csv(root/'predictions.csv',predictions)
    (root/'csa_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    saved=read_csv(root/'predictions.csv');assert len(saved)==5760
    for row in read_csv(root/'fixed_results.csv'):
        rs=[r for r in saved if all(r[k]==row[k] for k in ['branch','seed','method'])];train,probe=split(y,2,int(row['seed']))
        assert len(rs)==320 and {int(r['probe_index']) for r in rs}==set(probe)
        for r in rs:
            i,j=int(r['probe_index']),int(r['neighbor_index']);assert j in train and int(r['predicted_label'])==y[j]
            assert int(r['true_label'])==y[i] and int(r['correct'])==int(y[i]==y[j])
        assert abs(100*sum(int(r['correct']) for r in rs)/320-float(row['accuracy_percent']))<1e-10
        assert file_hash(root/f'{row["branch"]}_rounded_radius.npy')==row['feature_sha256']
    for h in json.loads((root/'csa_history.json').read_text()):
        errors=[h['initial_sse']]+[v for r in h['history'] for v in r['mode_sse']];assert np.all(np.diff(errors)<=h['initial_sse']*1e-9)
    (root/'verification.json').write_text(json.dumps({'passed':True,'configurations':18,'predictions':5760,
        'checks':['expected radius sizes','all saved predictions and feature hashes','CSA monotone SSE'],
        'limits':'development diagnostic; not author-verified original filter'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'results/orl_gabor_radius_rounding')
    with threadpool_limits(limits=4):run(p.parse_args().output)
