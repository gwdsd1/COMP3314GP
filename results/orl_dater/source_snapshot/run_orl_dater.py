"""One frozen ORL DATER extension:20models,24-D direct NN and23-D LDA.

Gallery-only fitting and independent mode/eigen checks precede probe scoring.
CSA/PCA control fits are reused; no input,rank,iteration or probe selection.
"""
import argparse
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from dater import DATER,canonical_columns
from gabor import file_hash
import run_orl_controlled as C

BASE=Path(__file__).resolve().parent
ROOT=BASE/'results/orl_dater'
SEEDS=list(range(4200,4220));RANKS=(4,2,3);TMAX=20;EPSILON=.001
SOURCES=['run_orl_dater.py','dater.py']+C.SOURCES
SOURCES=list(dict.fromkeys(SOURCES))
read,dump,table,csv_rows=C.read,C.dump,C.table,C.csv_rows


def independent_fit(x,labels,ranks,max_iter=20,epsilon=.001):
    # Explicit tensor contractions and per-object scatter accumulation;
    # generalized problem solved through Sw whitening, not eigh(Sb,Sw).
    x=np.asarray(x,dtype=float);u=[np.eye(m) for m in x.shape[1:]];states=[];hist=[]
    classes=np.unique(labels)
    for iteration in range(1,max_iter+1):
        previous=[v.copy() for v in u];changes=[]
        for k,rank in enumerate(ranks):
            if k==0:z=np.einsum('nhwc,wb,cd->nhbd',x,u[1],u[2],optimize=True)
            elif k==1:z=np.einsum('nhwc,ha,cd->nwad',x,u[0],u[2],optimize=True)
            else:z=np.einsum('nhwc,ha,wb->ncab',x,u[0],u[1],optimize=True)
            z=z.reshape(len(x),x.shape[k+1],-1);overall=z.mean(0)
            means={c:z[labels==c].mean(0) for c in classes}
            sw=sum((v-means[c])@(v-means[c]).T for v,c in zip(z,labels))
            sb=sum(np.sum(labels==c)*(means[c]-overall)@(means[c]-overall).T for c in classes)
            sw=(sw+sw.T)/2;sb=(sb+sb.T)/2
            d,v=eigh(sw);assert d[0]>len(d)*np.finfo(float).eps*d[-1]
            whitening=v/np.sqrt(d)
            values,w=eigh(whitening.T@sb@whitening)
            values=values[-rank:][::-1];w=whitening@w[:,-rank:][:,::-1]
            w=w/np.sqrt(np.sum(w*w,axis=0))
            for j in range(rank):
                if w[np.argmax(np.abs(w[:,j])),j]<0:w[:,j]*=-1
            u[k]=w;states.append(dict(iteration=iteration,mode=k,sw=sw,sb=sb,basis=w.copy(),eigenvalues=values))
            changes.append(None if previous[k].shape!=w.shape else float(np.sqrt(np.sum((w-previous[k])**2))/(x.shape[k+1]*rank)))
        hist.append(changes)
        if iteration>2 and all(c<epsilon for c in changes):break
    return u,states,hist


def preflight():
    # The n=1 special case must agree with ordinary LDA directions.
    rng=np.random.default_rng(814);labels=np.repeat(np.arange(5),5)
    vectors=rng.normal(size=(25,6))+rng.normal(size=(5,6))[labels]
    d=DATER((3,),max_iter=TMAX,epsilon=EPSILON).fit(vectors,labels)
    lda=C.E.LDA().fit(vectors,labels)
    expected=canonical_columns(lda.basis[:,:3]);np.testing.assert_allclose(d.bases[0],expected,atol=1e-10)
    # A genuine tensor checks mode scatters,full initialization and whitening.
    x=rng.normal(size=(25,6,5,4))+rng.normal(size=(5,6,5,4))[labels]
    model=DATER((2,2,2),max_iter=TMAX,epsilon=EPSILON).fit(x,labels)
    axes,states,history=independent_fit(x,labels,(2,2,2),TMAX,EPSILON)
    assert len(states)==len(model.updates)
    errors=[]
    for a,b in zip(model.updates,states):
        np.testing.assert_allclose(a['sw'],b['sw'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(a['sb'],b['sb'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(a['basis'],b['basis'],atol=1e-9,rtol=1e-9)
        errors.append(float(np.max(np.abs(a['basis']-b['basis']))))
    return dict(passed=True,synthetic_only=True,no_ORL_probe_scoring=True,
        one_mode_LDA_max_basis_error=float(np.max(np.abs(d.bases[0]-expected))),
        tensor_modes_checked=len(states),max_reference_basis_error=max(errors))


def prepare():
    assert not ROOT.exists(),'Use a new output directory'
    checks=preflight();C.check();C.check_models()
    ROOT.mkdir(parents=True)
    for f in ['models','predictions','source_snapshot']:(ROOT/f).mkdir()
    for n in SOURCES:shutil.copy2(BASE/n,ROOT/'source_snapshot'/n)
    prior=BASE/'results/orl_controlled'
    assert read(prior/'verification.json')['passed']
    entry=next(e for e in read(prior/'feature_manifest.json') if e['id']=='2pi_crop')
    controls=[r for r in csv_rows(prior/'results.csv') if r['input']=='2pi_crop' and r['method'] in ['CSA24_LDA23','PCA24_LDA23']]
    assert len(controls)==40
    old_files=[prior/n for n in ['protocol.json','results.csv','verification.json','splits.json']]
    old_files.extend(BASE/r[k] for r in controls for k in ['model_file','prediction_file'])
    source_pdf=BASE/'docs/sources/DATER_CVPR2005.pdf'
    dump(ROOT/'protocol.json',dict(date='2026-10-08',seeds=SEEDS,input=entry,ranks=RANKS,
        gallery_per_class=2,probe_per_class=8,n_classes=40,learning_count=80,query_count=320,
        algorithm='DATER Figure3,modewise label scatter and generalized eigenvectors',
        initialization='full square I28,I23,I40; sequentially replaced by 28x4,23x2,40x3',
        mode_order=[0,1,2],projection_center=False,
        solver='scipy.linalg.eigh(Sb,Sw); descending eigenvalues; no ridge,shrinkage,QR or nullspace fallback',
        column_scale='Euclidean unit length; largest-absolute-coordinate positive sign',
        max_iter=TMAX,epsilon=EPSILON,stop='t>2 and every Frobenius(U_t-U_prev)/(m*r)<epsilon; otherwise cap20',
        iteration_choice='first convergence or cap; no probe or best-training-DTC iteration selection',
        classifications=['24D direct gallery-centered cosine','24->23 ordinary LDA unit columns and gallery-centered cosine'],
        LDA=dict(alpha=0,pre_dimension=24,output_dimension=23,limit='24<=80-40'),
        controls='reuse fixed2pi_crop CSA24 and PCA24 cores,ordinaryLDA and full saved predictions; no control refits',
        freeze='all20gallery models and update states frozen+independently checked before any new probe scores',
        reference=dict(title='Discriminant Analysis with Tensor Representation',year=2005,
            url='https://mmlab.ie.cuhk.edu.hk/archive/2005/01467312.pdf',path=source_pdf.relative_to(BASE).as_posix(),sha256=file_hash(source_pdf),
            source_settings='Figure3fullidentity,GaussSeidel modes,generalizedeigenproblem,basis-change rule; paper numerical epsilon,Tmax,normalization not specified'),
        source_sha256={n:file_hash(BASE/n) for n in SOURCES},
        historical_sha256={p.relative_to(BASE).as_posix():file_hash(p) for p in old_files},
        limits=['supervised method extension,not originalCSA reproduction or TA-endorsed DATER settings',
            'DATER paper has PIE/FERET,not matched ORL24D table; no direct DATER paper numeric comparison',
            'CSA90.9 is a different-method reference for LDA branch only',
            'same400images and observed splits,not blind or independent subject repetitions',
            'unit normalization,max20,epsilon.001 and24D ranks are disclosed group choices',
            'DATER also changes initialization,iterations andnonorthogonalmetric; not supervision-only causal isolation',
            'Figure3 generalized-eigen rule followed; no assertion of monotonic joint DTC or global trace-ratio optimum']))
    dump(ROOT/'controls.json',controls);dump(ROOT/'splits.json',read(prior/'splits.json'))
    dump(ROOT/'preflight.json',checks)
    dump(ROOT/'freeze.json',{n:file_hash(ROOT/n) for n in ['protocol.json','controls.json','splits.json']})
    print('Frozen one input,20splits,ranks,solver,normalization and stopping; no probe scores.',flush=True)


def check():
    for n,h in read(ROOT/'freeze.json').items():assert file_hash(ROOT/n)==h,n
    p=read(ROOT/'protocol.json')
    for n,h in p['source_sha256'].items():assert file_hash(BASE/n)==h,n
    for n,h in p['historical_sha256'].items():assert file_hash(BASE/n)==h,n
    assert file_hash(BASE/p['input']['path'])==p['input']['sha256']
    assert file_hash(BASE/p['reference']['path'])==p['reference']['sha256']
    return p


def fit():
    p=check();assert not (ROOT/'model_freeze.json').exists()
    data=np.load(BASE/p['input']['path'],mmap_mode='r');rows=[];started=time.perf_counter()
    for seed in SEEDS:
        learn,query=C.E.split(C.E.LABELS,2,seed)
        path=ROOT/'models'/f'seed{seed}.npz';assert not path.exists()
        tic=time.perf_counter();model=DATER(RANKS,max_iter=TMAX,epsilon=EPSILON).fit(data[learn],C.E.LABELS[learn])
        g=model.transform(data[learn]).reshape(80,24);lda=C.E.LDA().fit(g,C.E.LABELS[learn]);assert lda.max_components>=23
        arrays=dict(learn=learn,query=query,gallery_core=g,lda_basis=lda.basis[:,:23],lda_mean=lda.mean,
            sw=lda.sw,sb=lda.sb,eigenvalues=lda.eigenvalues[:23])
        arrays.update({f'axis{k}':u for k,u in enumerate(model.bases)})
        for state in model.updates:
            stem=f't{state["iteration"]:02d}_k{state["mode"]}'
            for key in ['sw','sb','basis','eigenvalues']:arrays[f'{stem}_{key}']=state[key]
        np.savez_compressed(path,**arrays)
        history=ROOT/'models'/f'seed{seed}_history.json'
        dump(history,dict(stop_reason=model.stop_reason,history=model.history,
            updates=[{k:v for k,v in u.items() if k not in ['sw','sb','basis','eigenvalues']} for u in model.updates]))
        rows.append(dict(seed=seed,iterations=len(model.history),stop_reason=model.stop_reason,
            mode_updates=len(model.updates),final_DTC_ratio=model.history[-1]['DTC_ratio'],
            largest_Sw_condition=max(u['sw_condition'] for u in model.updates),
            fit_gallery_encode_LDA_seconds=time.perf_counter()-tic,
            model_file=path.relative_to(BASE).as_posix(),history_file=history.relative_to(BASE).as_posix()))
        print(f'Gallery-only seed{seed}:iterations{len(model.history)},stop{model.stop_reason}',flush=True)
    table(ROOT/'training.csv',rows)
    files=[ROOT/'training.csv']+[BASE/r[k] for r in rows for k in ['model_file','history_file']]
    dump(ROOT/'model_freeze.json',{p.relative_to(BASE).as_posix():file_hash(p) for p in files})
    dump(ROOT/'fit_execution.json',dict(new_DATER_models=20,control_refits=0,no_probe_scoring=True,
        elapsed_seconds=time.perf_counter()-started,BLAS_threads=2,python=platform.python_version()))


def check_models():
    for n,h in read(ROOT/'model_freeze.json').items():assert file_hash(BASE/n)==h,n


def verify_models():
    p=check();check_models();data=np.load(BASE/p['input']['path'],mmap_mode='r')
    rows=csv_rows(ROOT/'training.csv');assert len(rows)==20
    max_basis=max_residual=max_scatter=max_core=0.;updates=0
    for row in rows:
        seed=int(row['seed']);learn,query=C.E.split(C.E.LABELS,2,seed)
        with np.load(BASE/row['model_file']) as m:
            np.testing.assert_array_equal(m['learn'],learn);np.testing.assert_array_equal(m['query'],query)
            x=np.asarray(data[learn],dtype=float);axes,states,hist=independent_fit(x,C.E.LABELS[learn],RANKS,TMAX,EPSILON)
            saved=read(BASE/row['history_file']);assert len(hist)==int(row['iterations'])
            assert len(states)==int(row['mode_updates'])
            for state in states:
                stem=f't{state["iteration"]:02d}_k{state["mode"]}'
                for key in ['sw','sb']:
                    a,b=m[f'{stem}_{key}'],state[key]
                    err=float(np.linalg.norm(a-b)/max(1,np.linalg.norm(a)));max_scatter=max(max_scatter,err)
                    assert err<1e-9
                a,b=m[f'{stem}_basis'],state['basis'];err=float(np.max(np.abs(a-b)));max_basis=max(max_basis,err)
                np.testing.assert_allclose(a,b,atol=3e-7,rtol=2e-7)
                np.testing.assert_allclose(np.linalg.norm(a,axis=0),1,atol=1e-12)
                ev=m[f'{stem}_eigenvalues'];np.testing.assert_allclose(ev,state['eigenvalues'],atol=2e-6,rtol=2e-7)
                residual=float(np.linalg.norm(m[f'{stem}_sb']@a-(m[f'{stem}_sw']@a)*ev)/max(1,np.linalg.norm(m[f'{stem}_sb']@a)))
                assert residual<1e-8;max_residual=max(max_residual,residual);updates+=1
            for k,u in enumerate(axes):np.testing.assert_allclose(u,m[f'axis{k}'],atol=3e-7,rtol=2e-7)
            g=np.einsum('nhwc,ha,wb,cd->nabd',x,m['axis0'],m['axis1'],m['axis2'],optimize=True).reshape(80,24)
            max_core=max(max_core,float(np.max(np.abs(g-m['gallery_core']))))
            np.testing.assert_allclose(g,m['gallery_core'],atol=3e-8,rtol=1e-10)
            mean=g.mean(0);sw=np.zeros((24,24));sb=np.zeros((24,24))
            for c in range(1,41):
                z=g[C.E.LABELS[learn]==c];cm=z.mean(0);delta=z-cm;sw+=delta.T@delta
                sb+=len(z)*np.outer(cm-mean,cm-mean)
            np.testing.assert_allclose(sw,m['sw'],atol=2e-7,rtol=2e-8)
            np.testing.assert_allclose(sb,m['sb'],atol=2e-7,rtol=2e-8)
            ev=eigh(sb,sw,eigvals_only=True)[::-1][:23]
            np.testing.assert_allclose(ev,m['eigenvalues'],atol=1e-6,rtol=2e-7)
            w=m['lda_basis'];f=float(np.linalg.norm(sb@w-(sw@w)*ev)/max(1,np.linalg.norm(sb@w)))
            assert f<1e-8;max_residual=max(max_residual,f)
            if saved['stop_reason']=='paper_basis_change':assert all(v<EPSILON for v in hist[-1])
            else:assert len(hist)==TMAX and saved['stop_reason']=='max_iter'
        print(f'Independently checked gallery-only DATER seed{seed}',flush=True)
    dump(ROOT/'model_verification.json',dict(passed=True,models_checked=20,independent_DATER_refits=20,
        mode_scatter_and_eigen_updates_checked=updates,ordinary_LDA_checks=20,
        max_unit_basis_abs_error=max_basis,max_scatter_relative_error=max_scatter,
        max_generalized_eigen_residual=max_residual,max_core_encode_abs_error=max_core,
        source_input_models_and_historical_hashes_unchanged=True,no_probe_scoring=True))


def evaluate():
    p=check();check_models();assert read(ROOT/'model_verification.json')['passed'];assert not (ROOT/'results.csv').exists()
    data=np.load(BASE/p['input']['path'],mmap_mode='r');rows=[];started=time.perf_counter()
    controls=csv_rows(ROOT/'controls.json')
    for seed in SEEDS:
        learn,query=C.E.split(C.E.LABELS,2,seed)
        candidates=[('DATER24',ROOT/'models'/f'seed{seed}.npz',None)]
        candidates.extend((name,BASE/r['model_file'],r) for name in ['CSA24','PCA24']
            for r in controls if int(r['seed'])==seed and r['method']==name+'_LDA23')
        for name,path,prior in candidates:
            with np.load(path) as m:
                g=m['gallery_core'];q=C.encode(m,data[query])
                if prior is not None:np.testing.assert_allclose(q,m['query_core'],atol=3e-8,rtol=1e-10)
                for branch,ld in [('direct24',None),('LDA23',23)]:
                    w=m['lda_basis'] if ld else None;a,b=C.E.outputs(g,q,w,C.E.POLICIES[0])
                    pred,near=C.nn(a,b,learn)
                    if prior is not None and ld:
                        with np.load(BASE/prior['prediction_file']) as old:
                            np.testing.assert_array_equal(pred,old['predictions']);np.testing.assert_array_equal(near,old['neighbor_index'])
                    pf=ROOT/'predictions'/f'seed{seed}_{name}_{branch}.npz'
                    np.savez_compressed(pf,learn=learn,query=query,gallery=a,probe=b,query_core=q,predictions=pred,neighbor_index=near)
                    rows.append(dict(seed=seed,method=name,branch=branch,output_dimension=ld or 24,
                        correct=int(np.sum(pred==C.E.LABELS[query])),accuracy_percent=100*float(np.mean(pred==C.E.LABELS[query])),
                        model_file=path.relative_to(BASE).as_posix(),prediction_file=pf.relative_to(BASE).as_posix(),
                        control_fit_reused=prior is not None))
        print(f'Probe scored seed{seed}:all6matched branches',flush=True)
    assert len(rows)==120
    table(ROOT/'results.csv',rows)
    dump(ROOT/'score_freeze.json',{p.relative_to(BASE).as_posix():file_hash(p)
        for p in [ROOT/'results.csv']+[BASE/r['prediction_file'] for r in rows]})
    dump(ROOT/'evaluation_execution.json',dict(score_rows=120,prediction_records=38400,new_DATER_fits=0,
        control_fits_reused=40,LDA_control_prediction_matches=40,elapsed_seconds=time.perf_counter()-started,BLAS_threads=2))


def verify():
    check();check_models()
    for n,h in read(ROOT/'score_freeze.json').items():assert file_hash(BASE/n)==h,n
    rows=csv_rows(ROOT/'results.csv');assert len(rows)==120
    expected={(s,m,b) for s in SEEDS for m in ['DATER24','CSA24','PCA24'] for b in ['direct24','LDA23']}
    assert {(int(r['seed']),r['method'],r['branch']) for r in rows}==expected
    data=np.load(BASE/read(ROOT/'protocol.json')['input']['path'],mmap_mode='r');count=0
    for row in rows:
        learn,query=C.E.split(C.E.LABELS,2,int(row['seed']))
        with np.load(BASE/row['model_file']) as m,np.load(BASE/row['prediction_file']) as p:
            np.testing.assert_array_equal(m['learn'],learn);np.testing.assert_array_equal(m['query'],query)
            g=C.encode(m,data[learn]);q=C.encode(m,data[query]);np.testing.assert_allclose(g,m['gallery_core'],atol=3e-8,rtol=1e-10)
            w=m['lda_basis'] if row['branch']=='LDA23' else None
            mean=g.mean(0);a,b=g-mean,q-mean
            if w is not None:
                w=w/np.sqrt(np.sum(w*w,axis=0));a,b=a@w,b@w
            np.testing.assert_allclose(a,p['gallery'],atol=3e-8,rtol=1e-10)
            np.testing.assert_allclose(b,p['probe'],atol=3e-8,rtol=1e-10)
            pred,near=C.nn(a,b,learn,True)
            np.testing.assert_array_equal(p['learn'],learn);np.testing.assert_array_equal(p['query'],query)
            np.testing.assert_array_equal(pred,p['predictions']);np.testing.assert_array_equal(near,p['neighbor_index'])
            assert int(np.sum(pred==C.E.LABELS[query]))==int(row['correct'])
            assert abs(100*np.mean(pred==C.E.LABELS[query])-float(row['accuracy_percent']))<1e-10
            count+=len(query)
        if row['method']=='PCA24' and row['branch']=='LDA23':print(f'Independent NN verified seed{row["seed"]}',flush=True)
    dump(ROOT/'verification.json',dict(passed=True,model_count=60,new_DATER_models=20,reused_control_models=40,
        independent_DATER_model_verification=read(ROOT/'model_verification.json'),score_rows_checked=120,
        prediction_records_checked=count,unchanged_LDA_control_predictions=40,
        all_source_input_model_score_and_historical_hashes_unchanged=True))


def analyze():
    check();check_models();assert read(ROOT/'verification.json')['passed']
    rows=csv_rows(ROOT/'results.csv');summary=[];paired=[]
    for branch in ['direct24','LDA23']:
        scores={m:np.array([float(next(r['accuracy_percent'] for r in rows if int(r['seed'])==s and r['method']==m and r['branch']==branch)) for s in SEEDS]) for m in ['DATER24','CSA24','PCA24']}
        for m,v in scores.items():
            delta=v-scores['CSA24']
            summary.append(dict(method=m,branch=branch,n_splits=20,mean_percent=float(v.mean()),std_pp=float(v.std(ddof=1)),
                min_percent=float(v.min()),max_percent=float(v.max()),CSA_matched_change_pp=float(delta.mean()),
                wins_vs_CSA=int(np.sum(delta>0)),ties_vs_CSA=int(np.sum(delta==0)),losses_vs_CSA=int(np.sum(delta<0)),
                DATER_paper_ORL_reference=None,CSA_paper_LDA_reference=90.9 if branch=='LDA23' else None))
        for control in ['CSA24','PCA24']:
            d=scores['DATER24']-scores[control]
            paired.append(dict(branch=branch,contrast=f'DATER24_minus_{control}',mean_change_pp=float(d.mean()),std_pp=float(d.std(ddof=1)),
                wins=int(np.sum(d>0)),ties=int(np.sum(d==0)),losses=int(np.sum(d<0))))
    table(ROOT/'summary.csv',summary);dump(ROOT/'summary.json',summary);table(ROOT/'paired_changes.csv',paired)
    training=csv_rows(ROOT/'training.csv')
    dump(ROOT/'training_summary.json',dict(models=20,iterations=[int(r['iterations']) for r in training],
        converged=sum(r['stop_reason']=='paper_basis_change' for r in training),
        reached_cap=sum(r['stop_reason']=='max_iter' for r in training),
        maximum_Sw_condition=max(float(r['largest_Sw_condition']) for r in training),
        no_regularization=True,settings_selected_by_probe=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for i,m in enumerate(['CSA24','PCA24','DATER24']):
        group=[next(r for r in summary if r['method']==m and r['branch']==b) for b in ['direct24','LDA23']]
        ax.errorbar(np.arange(2)+(i-1)*.15,[r['mean_percent'] for r in group],yerr=[r['std_pp'] for r in group],fmt='o',capsize=4,label=m)
    ax.scatter(1,90.9,marker='*',c='black',s=95,label='CSA paper LDA (different method/aggregation)')
    ax.set(xticks=range(2),xticklabels=['24D direct NN','24D + ordinary LDA23'],ylabel='Accuracy (%)',ylim=(50,100),
        title='ORL supervised tensor extension:20 matched fixed splits\n2pi crop input; no probe selection; mean +/- sample SD')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180);plt.close(fig)
    print(__import__('json').dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['prepare','fit','verify_models','evaluate','verify','analyze'])
    parser.add_argument('--output',default='results/orl_dater');args=parser.parse_args();ROOT=BASE/args.output
    with threadpool_limits(limits=2):globals()[args.stage]()
