"""按四个检查点执行最终共享协议；保留全部对照，不从 probe 挑主方案。"""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits

from csa import CSA, PCA, LDA
from diagnose_orl_gabor import filtered, specification
from gabor import file_hash
from prepare_available_data import alignment_matrix, TARGET_EYES, cv2
from run_available_reproduction import ORL_CASES, split, nn, transform_batches, lda_coordinates
from run_orl_branches import points
from run_orl_gray import write_csv
from run_feret import read_csv

BASE=Path(__file__).resolve().parent
ROOT=BASE/'results/shared_protocol_final'
LABELS=np.repeat(np.arange(1,41),10)
NAMES=[f's{p}/{i}.pgm' for p in range(1,41) for i in range(1,11)]
EXTRA=[('PCA-64 (Gabor)','gabor','pca',64,None,None),
       ('CSA-64 (Gabor)','gabor','csa',(4,2,8),None,None)]
PREPROCESS=['whole_hist','valid_hist','valid_zscore','valid_zscore_lanczos']
FEATURES=['adaptive_reflect','adaptive_zero','fixed39_reflect','fixed39_zero']

def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()

def feature_spec(name):
    s=specification('baseline')
    if name!='legacy_reflect':
        s['stable_radius']=True
        s['radius']=19 if name.startswith('fixed39') else None
        s['boundary']='constant' if name.endswith('zero') else 'reflect'
    return s

def summarize(rows):
    result=[]
    for variant,method in dict.fromkeys((r['variant'],r['method']) for r in rows):
        group=[r for r in rows if (r['variant'],r['method'])==(variant,method)]
        values=[r['accuracy_percent'] for r in group]
        paper=group[0]['paper_percent']
        result.append({'variant':variant,'method':method,'seeds':[r['seed'] for r in group],
                       'n_splits':len(values),'mean_percent':float(np.mean(values)),
                       'std_pp':float(np.std(values,ddof=1)) if len(values)>1 else 0.,
                       'min_percent':min(values),'max_percent':max(values),'paper_percent':paper,
                       'difference_pp':float(np.mean(values)-paper) if paper is not None else None})
    return result

def evaluate(out,variants,seeds,cases=ORL_CASES):
    rows=[];predictions=[];histories=[];splits=[];manifests=[]
    (out/'embeddings').mkdir()
    for seed in seeds:
        train,probe=split(LABELS,2,seed)
        splits.extend({'seed':seed,'partition':part,'order':j,'index':int(i),'file':NAMES[i],
                       'label':int(LABELS[i])} for part,indices in [('gallery',train),('probe',probe)] for j,i in enumerate(indices))
    for variant,gray_path,gabor_path in variants:
        gray=np.load(gray_path,mmap_mode='r');gabor=np.load(gabor_path,mmap_mode='r')
        assert gray.shape==(400,56,46) and gabor.shape==(400,28,23,40)
        manifests.append({'variant':variant,'gray_path':str(gray_path.relative_to(BASE)),
                          'gray_sha256':file_hash(gray_path),'gabor_path':str(gabor_path.relative_to(BASE)),
                          'gabor_sha256':file_hash(gabor_path)})
        for seed in seeds:
            train,probe=split(LABELS,2,seed)
            for number,(method,kind,algorithm,dimensions,lda_dim,paper) in enumerate(cases):
                tick=time.perf_counter();data=gray if kind=='gray' else gabor
                model=PCA(dimensions,center=True,solver='gram') if algorithm=='pca' else CSA(dimensions,center=False,max_iter=3)
                model.fit(data[train]);gallery=transform_batches(model,data,train);queries=transform_batches(model,data,probe)
                if lda_dim is not None:
                    assert gallery.shape[1]<=80-40
                    lda=LDA().fit(gallery,LABELS[train])
                    gallery=lda_coordinates(gallery,lda,lda_dim,'unit_columns')
                    queries=lda_coordinates(queries,lda,lda_dim,'unit_columns')
                predicted,neighbor,distance=nn(gallery,queries,LABELS[train],'cosine')
                key={'variant':variant,'seed':seed,'method':method}
                identifier=f'{variant}_{seed}_{number}'
                np.savez_compressed(out/'embeddings'/f'{identifier}.npz',gallery=gallery,probe=queries,
                                    train_indices=train,probe_indices=probe)
                rows.append({**key,'dataset':'ORL','pre_dims':str(dimensions),'lda_dim':lda_dim,
                             'center':algorithm=='pca','metric':'cosine','lda_scaling':'unit_columns',
                             'correct':int(np.sum(predicted==LABELS[probe])),'probe_count':320,
                             'accuracy_percent':100*float(np.mean(predicted==LABELS[probe])),
                             'paper_percent':paper,'embedding':f'embeddings/{identifier}.npz',
                             'input_sha256':manifests[-1][f'{kind}_sha256'],
                             'wall_seconds':time.perf_counter()-tick})
                predictions.extend({**key,'probe_index':int(i),'probe':NAMES[i],'true':int(LABELS[i]),
                                    'predicted':int(p),'neighbor_index':int(train[j]),
                                    'nearest_gallery':NAMES[train[j]],'distance':float(d),'correct':int(p==LABELS[i])}
                                   for i,p,j,d in zip(probe,predicted,neighbor,distance))
                if algorithm=='csa':histories.append({**key,'initial_sse':model.initial_sse,'history':model.history})
            print(f'{out.name}: {variant} seed={seed} '+', '.join(f'{r["method"]}={r["accuracy_percent"]:.2f}' for r in rows[-len(cases):]),flush=True)
    write_csv(out/'fixed_results.csv',rows);write_csv(out/'predictions.csv',predictions);write_csv(out/'splits.csv',splits)
    dump(out/'csa_history.json',histories);dump(out/'feature_manifests.json',manifests)
    summary=summarize(rows);dump(out/'summary.json',summary)
    fig,ax=plt.subplots(figsize=(12,6));methods=list(dict.fromkeys(r['method'] for r in rows))
    names=list(dict.fromkeys(r['variant'] for r in rows));x=np.arange(len(methods));width=.8/len(names)
    for j,variant in enumerate(names):
        selected=[next(r for r in summary if (r['variant'],r['method'])==(variant,m)) for m in methods]
        ax.bar(x+(j-(len(names)-1)/2)*width,[r['mean_percent'] for r in selected],width,
               yerr=[r['std_pp'] for r in selected],capsize=2,label=variant)
    papers=[next(r['paper_percent'] for r in summary if r['method']==m) for m in methods]
    for i,paper in enumerate(papers):
        if paper is not None:ax.plot([i-.4,i+.4],[paper,paper],color='black',linestyle='--',alpha=.65)
    ax.set(xticks=x,xticklabels=[m.replace(' (Gabor)','\nGabor').replace(' (Gray)','\nGray') for m in methods],
           ylabel='Accuracy (%)',ylim=(0,100),title=f'{out.name}: shared inputs, cosine NN; mean +/- split SD')
    ax.tick_params(axis='x',labelsize=9);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    return rows

def histogram(image,mask):
    """只在有效像素统计 CDF；CDF-min 映射到 0..255，无效像素保持 0。"""
    values=image[mask];counts=np.bincount(values,minlength=256);cdf=np.cumsum(counts)
    cmin=cdf[np.flatnonzero(counts)[0]]
    lut=np.round(255*np.maximum(cdf-cmin,0)/max(1,len(values)-cmin)).astype(np.uint8)
    output=np.zeros_like(image,dtype=np.float32);output[mask]=lut[values]
    return output

def prepare(out):
    annotations=read_csv(BASE/'data/annotations/orl_eye_coordinates_v2.csv');eyes=points(annotations)
    assert [r['file'] for r in annotations]==NAMES
    arrays={v:[] for v in PREPROCESS};masks={v:[] for v in PREPROCESS};coverage=[];qualities=[]
    for i,(name,source) in enumerate(zip(NAMES,eyes)):
        raw=np.asarray(Image.open(BASE/'data/raw/orl'/name).convert('L'))
        for high in [False,True]:
            # PIL resize pixel centers: low = (high + .5)/2 - .5.
            target=2*TARGET_EYES+.5 if high else TARGET_EYES
            matrix=alignment_matrix(source,target);shape=(92,112) if high else (46,56)
            image=cv2.warpAffine(raw,matrix,shape,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            weight=cv2.warpAffine(np.ones_like(raw,dtype=np.float32),matrix,shape,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            mapped=np.column_stack([source,np.ones(2)])@matrix.T
            if high:
                mapped=(mapped+.5)/2-.5
                image=np.asarray(Image.fromarray(image).resize((46,56),Image.Resampling.LANCZOS))
                weight=np.clip(np.asarray(Image.fromarray(weight).resize((46,56),Image.Resampling.LANCZOS)),0,1)
            error=float(np.max(np.abs(mapped-TARGET_EYES)));assert error<1e-10
            mask=weight>=.999
            qualities.append({'index':i,'file':name,'resampling':'lanczos' if high else 'direct',
                              'coverage_fraction':float(weight.mean()),'valid_fraction':float(mask.mean()),'eye_error':error})
            if high:
                coverage.append(weight)
            variants=['valid_zscore_lanczos'] if high else PREPROCESS[:3]
            for variant in variants:
                data=np.asarray(ImageOps.equalize(Image.fromarray(image)),dtype=np.float32) if variant=='whole_hist' else histogram(image,mask)
                if variant.startswith('valid_zscore'):
                    values=data[mask].astype(np.float64);data[mask]=(values-values.mean())/max(values.std(),1e-12)
                    data[~mask]=0
                arrays[variant].append(data);masks[variant].append(mask)
    for variant in PREPROCESS:
        np.save(out/f'{variant}.npy',np.stack(arrays[variant]));np.save(out/f'{variant}_mask.npy',np.stack(masks[variant]))
    np.save(out/'lanczos_coverage.npy',np.stack(coverage));write_csv(out/'image_quality.csv',qualities)
    baseline=np.load(BASE/'data/processed/orl_branches_v2/eyes_only_seed20.npy')
    np.testing.assert_array_equal(np.stack(arrays['whole_hist']),baseline)
    quality={v:{'mean_coverage':float(np.mean([r['coverage_fraction'] for r in qualities if r['resampling']==v])),
                'min_coverage':min(r['coverage_fraction'] for r in qualities if r['resampling']==v)} for v in ['direct','lanczos']}
    selected='valid_zscore_lanczos' if quality['lanczos']['mean_coverage']>=.95 and quality['lanczos']['min_coverage']>=.60 else 'valid_zscore'
    assert quality['direct']['mean_coverage']>=.95 and quality['direct']['min_coverage']>=.60
    dump(out/'preprocessing_selection.json',{'selected':selected,'quality':quality,'selected_before_scores':True,
         'rule':'valid-region histogram + population z-score, Lanczos if geometric coverage mean>=.95 and min>=.60; otherwise direct warp',
         'not_author_parameters':True,'valid_mask':'coverage >= .999; no ellipse mask',
         'histogram':'CDF-min uint8 rounded 0..255; invalid pixels zero; then z-score on valid pixels',
         'resampling':'warp to 112x92 at target 2*eyes+.5; Lanczos resize to 56x46',
         'annotation_sha256':file_hash(BASE/'data/annotations/orl_eye_coordinates_v2.csv')})
    selected_indices=[0,3,120,125,126,127,359,np.argmin(np.stack(coverage).mean(axis=(1,2)))]
    fig,axes=plt.subplots(len(selected_indices),len(PREPROCESS),figsize=(9,14))
    for row,i in enumerate(selected_indices):
        for col,v in enumerate(PREPROCESS):
            axes[row,col].imshow(arrays[v][i],cmap='gray');axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(v,fontsize=9)
            if col==0:axes[row,col].text(-.1,.5,NAMES[i],transform=axes[row,col].transAxes,rotation=90,va='center')
            axes[row,col].plot(TARGET_EYES[:,0],TARGET_EYES[:,1],'r+',markersize=3)
    fig.tight_layout();fig.savefig(out/'preprocessing_examples.png',dpi=150);plt.close(fig)
    return selected

def collect_external(out):
    rows=[];sources=[]
    for directory,profile,dataset in [('available_reproduction_v2','group_primary','MNIST'),('feret_validation_selected_v3','cosine_control','FERET')]:
        source=BASE/'results'/directory
        assert json.loads((source/'verification.json').read_text())['status']=='passed'
        records=[r for r in read_csv(source/'fixed_results.csv') if r['profile']==profile and (dataset=='FERET' or r['dataset'].startswith('MNIST'))]
        for r in records:
            assert r['metric']=='cosine' and r['lda_scaling']=='unit_columns'
            rows.append({'dataset':r.get('dataset',f'FERET q={r.get("q")}'),'method':r['method'],
                         'seed':r.get('seed','fixed'),'correct':int(r['correct']),'probe_count':int(r['probe_count']),
                         'accuracy_percent':float(r['accuracy_percent']),'paper_percent':float(r['paper_percent']),
                         'source':str(source.relative_to(BASE)),'source_profile':profile})
        sources.append({'directory':str(source.relative_to(BASE)),'profile':profile,
                        'sha256':{n:file_hash(source/n) for n in ['fixed_results.csv','predictions.csv','protocol.json','verification.json']}})
    write_csv(out/'external_shared_results.csv',rows);dump(out/'external_sources.json',sources)

def main(stage):
    ROOT.mkdir(exist_ok=True)
    if stage>1:
        assert json.loads((ROOT/f'step{stage-1}'/'verification.json').read_text())['passed']
    out=ROOT/f'step{stage}';out.mkdir(exist_ok=False)
    snap=out/'source_snapshot';snap.mkdir()
    for name in ['run_shared_protocol.py','verify_shared_protocol.py','csa.py','diagnose_orl_gabor.py','run_available_reproduction.py','prepare_available_data.py']:
        shutil.copy2(BASE/name,snap/name)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,
              'not_blind':'ORL probes and historical variants previously observed; all alternatives retained',
              'group_conventions':{'metric':'cosine','lda_scaling':'unit_columns','PCA_center':True,'CSA_center':False,
                                   'CSA_init':'identity','CSA_Tmax':3,'epsilon':.001},
              'fixed_paper_cases':ORL_CASES,'code_sha256':{p.name:file_hash(p) for p in snap.iterdir()}}
    dump(out/'protocol.json',protocol)
    if stage==1:
        gray=BASE/'data/processed/orl_branches_v2/eyes_only_seed20.npy'
        array=np.load(gray);path=gray.parent/f'gabor_{digest(array)[:16]}.npy'
        assert path.exists()
        protocol.update({'seeds':[20,21,22],'feature_spec':feature_spec('legacy_reflect')})
        dump(out/'protocol.json',protocol)
        evaluate(out,[('shared_baseline',gray,path)],[20,21,22]);collect_external(out)
    elif stage==2:
        selected=prepare(out)
        protocol.update({'preprocessing_candidates':PREPROCESS,'selected_preprocessing':selected,
                         'seeds':[20,21,22],'feature_spec':feature_spec('legacy_reflect')})
        dump(out/'protocol.json',protocol) # preprocessing selection precedes model fitting
        variants=[]
        for v in PREPROCESS:
            path=out/f'{v}_gabor.npy';np.save(path,filtered(np.load(out/f'{v}.npy'),feature_spec('legacy_reflect')))
            variants.append((v,out/f'{v}.npy',path))
        evaluate(out,variants,[20,21,22])
    elif stage==3:
        prep=ROOT/'step2';selected=json.loads((prep/'preprocessing_selection.json').read_text())['selected']
        # Standard 3-sigma truncation and zero extension are fixed here before scores.
        protocol.update({'selected_preprocessing':selected,'selected_gabor':'adaptive_zero',
                         'selection_reason':'canonical adaptive 3-sigma DC-free kernel; zero exterior avoids reflected synthetic facial structure',
                         'gabor_candidates':{v:feature_spec(v) for v in FEATURES},'seeds':[20,21,22],
                         'reference':'https://cenl.ucsd.edu/pubs/ieee-1997.pdf',
                         'reference_limit':'boundary/support and reuse of wavelength 4..16 on ORL are group conventions; not original CSA settings'})
        dump(out/'protocol.json',protocol)
        variants=[]
        for v in FEATURES:
            path=out/f'{v}.npy';np.save(path,filtered(np.load(prep/f'{selected}.npy'),feature_spec(v)))
            variants.append((v,prep/f'{selected}.npy',path))
        evaluate(out,variants,[20,21,22],ORL_CASES[2:])
    else:
        prior=json.loads((ROOT/'step3/protocol.json').read_text());prep=prior['selected_preprocessing'];feature=prior['selected_gabor']
        protocol.update({'frozen_preprocessing':prep,'frozen_gabor':feature,'seeds':list(range(20,30)),
                         'prior_protocol_sha256':file_hash(ROOT/'step3/protocol.json'),
                         'extra_equal_coefficients':EXTRA,'no_selection_from_step4_scores':True})
        dump(out/'protocol.json',protocol)
        rows=evaluate(out,[('frozen_main',ROOT/'step2'/f'{prep}.npy',ROOT/'step3'/f'{feature}.npy')],list(range(20,30)),ORL_CASES+EXTRA)
        pairs=[]
        for pca,csa in [('PCA+LDA (Gray)','CSA-2+LDA'),('PCA (Gabor)','CSA-3'),('PCA+LDA (Gabor)','CSA-3+LDA'),('PCA-64 (Gabor)','CSA-64 (Gabor)')]:
            for seed in range(20,30):
                p=next(r for r in rows if r['method']==pca and r['seed']==seed)
                c=next(r for r in rows if r['method']==csa and r['seed']==seed)
                pairs.append({'pca':pca,'csa':csa,'seed':seed,'pca_percent':p['accuracy_percent'],
                              'csa_percent':c['accuracy_percent'],'csa_minus_pca_pp':c['accuracy_percent']-p['accuracy_percent']})
        write_csv(out/'paired_differences.csv',pairs)
        external=read_csv(ROOT/'step1/external_shared_results.csv');common=[]
        for r in summarize(rows):common.append({'dataset':'ORL','method':r['method'],'n_splits':r['n_splits'],
                    'mean_percent':r['mean_percent'],'std_pp':r['std_pp'],'paper_percent':r['paper_percent'],
                    'difference_pp':r['difference_pp'],'source':'step4/fixed_results.csv'})
        for dataset,method in dict.fromkeys((r['dataset'],r['method']) for r in external):
            selected=[r for r in external if (r['dataset'],r['method'])==(dataset,method)]
            values=[float(r['accuracy_percent']) for r in selected];paper=float(selected[0]['paper_percent'])
            common.append({'dataset':dataset,'method':method,'n_splits':len(values),'mean_percent':float(np.mean(values)),
                           'std_pp':float(np.std(values,ddof=1)) if len(values)>1 else 0.,'paper_percent':paper,
                           'difference_pp':float(np.mean(values)-paper),'source':selected[0]['source']})
        write_csv(ROOT/'shared_comparison.csv',common)
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for ax,subset in zip(axes,[pairs[:30],pairs[30:]]):
            for label in dict.fromkeys(r['csa'] for r in subset):
                group=[r for r in subset if r['csa']==label]
                ax.plot([r['seed'] for r in group],[r['csa_minus_pca_pp'] for r in group],marker='o',label=label)
            ax.axhline(0,color='black',linewidth=1);ax.set(xlabel='Split seed',ylabel='CSA - PCA (percentage points)');ax.legend(fontsize=8);ax.grid(alpha=.2)
        axes[0].set_title('Paper dimensions (unequal coefficient counts)');axes[1].set_title('Equal 64 coefficients (not equal basis storage)')
        fig.tight_layout();fig.savefig(out/'paired_differences.png',dpi=160);plt.close(fig)
    print(f'Stage {stage} complete; independent verification required before next stage.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,choices=[1,2,3,4],required=True)
    parser.add_argument('--output',type=Path,default=ROOT,help='四个阶段共用一个新的输出根目录')
    args=parser.parse_args()
    ROOT=args.output.resolve()
    with threadpool_limits(limits=4):main(args.stage)
