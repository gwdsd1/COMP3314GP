"""ORL 固定论文配置：教师旧分支、修订教师分支、论文描述对照与两个单因素对照。"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from threadpoolctl import threadpool_limits

from gabor import cached_features,file_hash
from prepare_available_data import alignment_matrix,TARGET_EYES
from run_available_reproduction import ORL_CASES,SEEDS,split,record_recognition,pure_csa_curve
from run_orl_gray import write_csv
from run_feret import read_csv

BASE=Path(__file__).resolve().parent
PREPARED=BASE/'data/processed/orl_branches_v2'
BRANCHES=['teacher_old','eyes_only','geometry_only','teacher_revised','paper_description']
sys.path.insert(0,str(BASE/'.deps'))
import cv2


def points(rows):
    return np.array([[[float(r['left_x']),float(r['left_y'])],[float(r['right_x']),float(r['right_y'])]] for r in rows])


def training_target(eyes,train):
    """half-size raw-eye median center/distance; horizontal target, no label/score fitting."""
    center=np.median(eyes[train].mean(axis=1),axis=0)/2
    distance=np.median(np.linalg.norm(eyes[train,1]-eyes[train,0],axis=1))/2
    target=np.array([center+[-distance/2,0],center+[distance/2,0]])
    assert 0<target[0,0]<target[1,0]<46 and 0<target[0,1]<56
    return target


def warp_batch(raw,eyes,target):
    images=[];quality=[]
    for i,(image,source) in enumerate(zip(raw,eyes)):
        matrix=alignment_matrix(source,target)
        error=float(np.max(np.abs(np.column_stack([source,np.ones(2)])@matrix.T-target)))
        assert error<1e-10
        warped=cv2.warpAffine(image,matrix,(46,56),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        mask=cv2.warpAffine(np.ones_like(image,dtype=np.float32),matrix,(46,56),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        images.append(np.asarray(ImageOps.equalize(Image.fromarray(warped)),dtype=np.float32))
        quality.append({'index':i,'eye_mapping_error':error,'support_fraction':float(mask.mean()),
                        'scale':float(np.linalg.norm(matrix[:,0]))})
    return np.stack(images),quality


def prepare():
    PREPARED.mkdir(parents=True,exist_ok=True)
    old=read_csv(BASE/'data/annotations/orl_eye_coordinates.csv');new=read_csv(BASE/'data/annotations/orl_eye_coordinates_v2.csv')
    assert [r['file'] for r in old]==[r['file'] for r in new]==[f's{p}/{i}.pgm' for p in range(1,41) for i in range(1,11)]
    raw=np.stack([np.asarray(Image.open(BASE/'data/raw/orl'/r['file']).convert('L')) for r in old])
    raw_digest=hashlib.sha256()
    for r in old:raw_digest.update(r['file'].encode());raw_digest.update((BASE/'data/raw/orl'/r['file']).read_bytes())
    labels=np.repeat(np.arange(1,41),10);np.testing.assert_array_equal(labels,np.load(BASE/'data/processed/available/orl_labels.npy'))
    e_old,e_new=points(old),points(new)
    legacy,q_old=warp_batch(raw,e_old,TARGET_EYES)
    np.testing.assert_array_equal(legacy,np.load(BASE/'data/processed/available/orl_aligned.npy'))
    revised_eyes,q_eyes=warp_batch(raw,e_new,TARGET_EYES)
    paper=np.stack([np.asarray(ImageOps.equalize(Image.fromarray(x).resize((46,56),Image.Resampling.BILINEAR)),dtype=np.float32) for x in raw])
    records=[];qualities=[];arrays={}
    for seed in SEEDS:
        train,probe=split(labels,2,seed);target=training_target(e_new,train)
        geometry,q_geo=warp_batch(raw,e_old,target);revised,q_new=warp_batch(raw,e_new,target)
        for branch,data,eyes_used,eye_target,qs in [('teacher_old',legacy,'old',TARGET_EYES,q_old),('eyes_only',revised_eyes,'v2',TARGET_EYES,q_eyes),
             ('geometry_only',geometry,'old',target,q_geo),('teacher_revised',revised,'v2',target,q_new),('paper_description',paper,'none',None,[])]:
            destination=PREPARED/f'{branch}_seed{seed}.npy';np.save(destination,data)
            arrays[branch,seed]=data
            records.append({'branch':branch,'seed':seed,'path':str(destination.relative_to(BASE)).replace('\\','/'),'eyes':eyes_used,
                            'target_xy':eye_target.tolist() if eye_target is not None else None,'sha256':file_hash(destination),
                            'support_mean':float(np.mean([q['support_fraction'] for q in qs])) if qs else 1.,
                            'support_min':min(q['support_fraction'] for q in qs) if qs else 1.})
            qualities.extend({'branch':branch,'seed':seed,'file':old[q['index']]['file'],**q} for q in qs)
    metadata={'raw_sha256':raw_digest.hexdigest(),'old_eyes_sha256':file_hash(BASE/'data/annotations/orl_eye_coordinates.csv'),
              'v2_eyes_sha256':file_hash(BASE/'data/annotations/orl_eye_coordinates_v2.csv'),
              'target_rule':'median training eye midpoint /2; median training inter-eye distance /2; horizontal target; no probe statistics',
              'shape':[400,56,46],'interpolation':'OpenCV bilinear, constant-zero boundary, Pillow equalize AFTER warp',
              'paper_description':'Pillow bilinear resize 92x112 to46x56 then equalize; no extra eye alignment',
              'annotations':'400 overlays rechecked; six image-based approximate corrections; not official ground truth','arrays':records}
    (PREPARED/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');write_csv(PREPARED/'alignment_quality.csv',qualities)
    canvas=Image.new('RGB',(5*180,7*145+30),'white');pen=ImageDraw.Draw(canvas)
    for col,branch in enumerate(BRANCHES):
        pen.text((col*180,3),branch,fill='black')
        for row,i in enumerate([0,3,10,125,127,359,210]):
            canvas.paste(Image.fromarray(arrays[branch,20][i].astype(np.uint8)).resize((92,112)),(col*180,row*145+25))
            pen.text((col*180,row*145+139),old[i]['file'],fill='black')
    canvas.save(PREPARED/'branches.png')
    return arrays,labels,[r['file'] for r in old],metadata


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source=['run_orl_branches.py','csa.py','gabor.py','prepare_available_data.py','run_available_reproduction.py','run_orl_gray.py']
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in source:shutil.copy2(BASE/name,snapshot/name)
    arrays,labels,names,metadata=prepare()
    # Reject extrapolating geometry by image quality before computing recognition scores.
    candidate=[r for r in metadata['arrays'] if r['branch']=='teacher_revised']
    accept_median=all(r['support_mean']>=.95 and r['support_min']>=.60 for r in candidate)
    teacher_branch='teacher_revised' if accept_median else 'eyes_only'
    protocol={'branches':BRANCHES,'main_branches':[teacher_branch,'paper_description'],'seeds':SEEDS,'fixed_cases':ORL_CASES,
              'teacher_quality_rule':{'candidate':'teacher_revised','require_mean_support':.95,'require_min_support':.60,
                  'accepted':accept_median,'fallback':'eyes_only: reviewed eyes with previously fixed target; no score used'},
              'preprocessing':metadata,'profile':'group_primary, unchanged model/metric/LDA scaling; no internal winner selection',
              'selected_by_probe':False,'chronology':'development evaluation; previous probes already seen',
              'code_sha256':{name:file_hash(BASE/name) for name in source}}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];predictions=[];curves=[];histories=[];splits=[];feature_metadata={}
    feature_cache={}
    for branch in BRANCHES:
        for seed in SEEDS:
            gray=arrays[branch,seed];digest=hashlib.sha256(gray.tobytes()).hexdigest()
            if digest not in feature_cache:
                path=PREPARED/f'gabor_{digest[:16]}.npy'
                feature_cache[digest]=cached_features(gray,path,step=2)
                feature_metadata[path.name]=json.loads(path.with_suffix('.json').read_text())
            gabor=feature_cache[digest];train,probe=split(labels,2,seed)
            splits.extend({'branch':branch,'seed':seed,'partition':part,'index':int(i),'file':names[i],'label':int(labels[i])}
                          for part,indices in [('train_gallery',train),('probe',probe)] for i in indices)
            for case in ORL_CASES:
                before=[len(x) for x in [rows,predictions,curves,histories]]
                record_recognition('ORL',seed,'group_primary',case,gray if case[1]=='gray' else gabor,labels,names,train,probe,rows,predictions,curves,histories)
                for container,n in zip([rows,predictions,curves,histories],before):
                    for r in container[n:]:r['branch']=branch
            if branch in protocol['main_branches']:
                n=len(curves);pure_csa_curve(gabor,labels,train,probe,seed,'group_primary',curves)
                for r in curves[n:]:r['branch']=branch
    for name,data in [('fixed_results',rows),('predictions',predictions),('dimension_curves',curves),('splits',splits)]:write_csv(output/f'{name}.csv',data)
    (output/'csa_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    (output/'feature_manifests.json').write_text(json.dumps(feature_metadata,indent=2),encoding='utf-8')
    summary=[]
    for branch in BRANCHES:
        for case in ORL_CASES:
            values=[r['accuracy_percent'] for r in rows if r['branch']==branch and r['method']==case[0]]
            summary.append({'branch':branch,'method':case[0],'mean_percent':float(np.mean(values)),'std_pp':float(np.std(values,ddof=1)),
                            'paper_percent':case[5],'difference_pp':float(np.mean(values))-case[5]})
    (output/'summary.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-started,'summary':summary},indent=2),encoding='utf-8')
    x=np.arange(6);fig,axis=plt.subplots(figsize=(12,5))
    for offset,(branch,color) in enumerate([('teacher_old','#999999'),(teacher_branch,'#d9822b'),('paper_description','#278d68')]):
        rs=[r for r in summary if r['branch']==branch]
        axis.bar(x+(offset-1)*.2,[r['mean_percent'] for r in rs],.2,yerr=[r['std_pp'] for r in rs],capsize=2,label=branch,color=color)
    axis.scatter(x,[c[5] for c in ORL_CASES],color='#2878b5',marker='D',label='Paper reference')
    axis.set(xticks=x,xticklabels=[c[0].replace('+LDA','\n+LDA') for c in ORL_CASES],ylabel='Accuracy (%)',ylim=(0,100),title='ORL: fixed paper dimensions and same 2/8 splits; mean +/- sample SD')
    axis.legend(loc='lower right');fig.tight_layout();fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE/'results/orl_branches_v3')
    with threadpool_limits(limits=4):run(parser.parse_args().output)
