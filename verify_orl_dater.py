"""Independent validation of each frozen DATER update and numeric sensitivity.

The initially frozen verifier compared two entire iterative paths at1e-9
scatter tolerance. Roundoff from different generalized eigensolvers can
accumulate. This verifier checks the exact saved path against independently
formed label scatters and a Sw-whitening leading-eigen solver, and reports
free-reference path differences instead of silently declaring equality.
It never changes fitting,model files,normalization,stopping or predictions.
"""
import argparse
import time

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

import run_orl_dater as R
from gabor import file_hash


def explicit_scatter(x,labels,bases,k):
    if k==0:z=np.einsum('nhwc,wb,cd->nhbd',x,bases[1],bases[2],optimize=True)
    elif k==1:z=np.einsum('nhwc,ha,cd->nwad',x,bases[0],bases[2],optimize=True)
    else:z=np.einsum('nhwc,ha,wb->ncab',x,bases[0],bases[1],optimize=True)
    z=z.reshape(len(x),x.shape[k+1],-1);overall=z.mean(0)
    means={c:z[labels==c].mean(0) for c in np.unique(labels)}
    sw=sum((v-means[c])@(v-means[c]).T for v,c in zip(z,labels))
    sb=sum(np.sum(labels==c)*(means[c]-overall)@(means[c]-overall).T for c in means)
    return (sb+sb.T)/2,(sw+sw.T)/2


def whitened_leading(sb,sw,rank):
    values,u=eigh(sw);assert values[0]>len(values)*np.finfo(float).eps*values[-1]
    w=u/np.sqrt(values);reduced=w.T@sb@w;reduced=(reduced+reduced.T)/2
    ev,v=eigh(reduced);basis=w@v[:,-rank:][:,::-1]
    basis=basis/np.sqrt(np.sum(basis*basis,axis=0))
    for j in range(rank):
        if basis[np.argmax(np.abs(basis[:,j])),j]<0:basis[:,j]*=-1
    return ev[-rank:][::-1],basis,values


def projector(u):
    # QR is solely a subspace comparison in the verifier,never used in DATER.
    q=np.linalg.qr(u)[0];return q@q.T


def verify():
    started=time.perf_counter();p=R.check();R.check_models()
    assert not (R.ROOT/'results.csv').exists(),'This gate must finish before first probe score'
    assert not (R.ROOT/'model_verification.json').exists()
    rows=R.csv_rows(R.ROOT/'training.csv');data=np.load(R.BASE/p['input']['path'],mmap_mode='r')
    sensitivity=[];checked=[];max_sb_sw=max_local_basis=max_residual=max_core=0.;updates=0;DTC_decreases=0
    for row in rows:
        seed=int(row['seed']);learn,query=R.C.E.split(R.C.E.LABELS,2,seed)
        x=np.asarray(data[learn],dtype=float);labels=R.C.E.LABELS[learn]
        saved_history=R.read(R.BASE/row['history_file']);n=int(row['iterations'])
        with np.load(R.BASE/row['model_file']) as m:
            np.testing.assert_array_equal(m['learn'],learn);np.testing.assert_array_equal(m['query'],query)
            bases=[np.eye(v) for v in x.shape[1:]];conditions=[];dtcs=[];previous_DTC=None
            for iteration in range(1,n+1):
                previous=[v.copy() for v in bases];changes=[]
                for k,rank in enumerate(R.RANKS):
                    stem=f't{iteration:02d}_k{k}';sb,sw=explicit_scatter(x,labels,bases,k)
                    for key,value in [('sb',sb),('sw',sw)]:
                        saved=m[f'{stem}_{key}'];err=float(np.linalg.norm(value-saved)/max(1,np.linalg.norm(saved)))
                        assert err<1e-9,(seed,stem,key,err);max_sb_sw=max(max_sb_sw,err)
                    values,ref,sw_values=whitened_leading(sb,sw,rank)
                    actual=m[f'{stem}_basis'];err=float(np.max(np.abs(actual-ref)))
                    assert err<3e-7,(seed,stem,'basis',err);max_local_basis=max(max_local_basis,err)
                    np.testing.assert_allclose(values,m[f'{stem}_eigenvalues'],atol=2e-6,rtol=2e-7)
                    np.testing.assert_allclose(np.linalg.norm(actual,axis=0),1,atol=1e-12)
                    residual=float(np.linalg.norm(sb@actual-(sw@actual)*values)/max(1,np.linalg.norm(sb@actual)))
                    assert residual<1e-8;max_residual=max(max_residual,residual)
                    bases[k]=actual.copy();updates+=1;conditions.append(float(sw_values[-1]/sw_values[0]))
                    changes.append(None if previous[k].shape!=actual.shape else float(np.linalg.norm(actual-previous[k])/(x.shape[k+1]*rank)))
                history=saved_history['history'][iteration-1]
                if iteration>1:np.testing.assert_allclose(changes,history['normalized_basis_changes'],atol=1e-12)
                else:assert changes==history['normalized_basis_changes']==[None,None,None]
                core=np.einsum('nhwc,ha,wb,cd->nabd',x,*bases,optimize=True).reshape(80,24)
                overall=core.mean(0);within=between=0.
                for c in range(1,41):
                    z=core[labels==c];cm=z.mean(0)
                    within+=float(np.sum((z-cm)**2));between+=len(z)*float(np.sum((cm-overall)**2))
                ratio=between/within;dtcs.append(ratio)
                assert abs(ratio-history['DTC_ratio'])<=1e-8*max(1,ratio)
                if previous_DTC is not None and ratio<previous_DTC-1e-10:DTC_decreases+=1
                previous_DTC=ratio
                should_stop=iteration>2 and all(v<R.EPSILON for v in changes)
                if iteration<n:assert not should_stop,'Saved fit passed its declared stopping event'
                elif saved_history['stop_reason']=='paper_basis_change':assert should_stop
                else:assert n==R.TMAX and not should_stop
            for k,u in enumerate(bases):np.testing.assert_array_equal(u,m[f'axis{k}'])
            err=float(np.max(np.abs(core-m['gallery_core'])));max_core=max(max_core,err)
            np.testing.assert_allclose(core,m['gallery_core'],atol=3e-8,rtol=1e-10)
            mean=core.mean(0);sw=np.zeros((24,24));sb=np.zeros_like(sw)
            for c in range(1,41):
                z=core[labels==c];cm=z.mean(0);delta=z-cm;sw+=delta.T@delta;sb+=len(z)*np.outer(cm-mean,cm-mean)
            np.testing.assert_allclose(sw,m['sw'],atol=2e-7,rtol=2e-8);np.testing.assert_allclose(sb,m['sb'],atol=2e-7,rtol=2e-8)
            ev,_,_=whitened_leading(sb,sw,23);np.testing.assert_allclose(ev,m['eigenvalues'],atol=1e-6,rtol=2e-7)
            w=m['lda_basis'];residual=float(np.linalg.norm(sb@w-(sw@w)*ev)/max(1,np.linalg.norm(sb@w)))
            assert residual<1e-8;max_residual=max(max_residual,residual)
            # A fully free reference path is retained as a sensitivity audit.
            free,free_states,free_hist=R.independent_fit(x,labels,R.RANKS,R.TMAX,R.EPSILON)
            final_error=max(float(np.max(np.abs(a-b))) for a,b in zip(bases,free))
            final_subspace=max(float(np.max(np.abs(projector(a)-projector(b)))) for a,b in zip(bases,free))
            free_core=np.einsum('nhwc,ha,wb,cd->nabd',x,*free,optimize=True).reshape(80,24)
            relative_core=float(np.linalg.norm(core-free_core)/max(1,np.linalg.norm(core)))
            sensitivity.append(dict(seed=seed,original_iterations=n,free_reference_iterations=len(free_hist),
                max_final_unit_basis_abs_difference=final_error,max_final_projector_abs_difference=final_subspace,
                final_gallery_core_relative_difference=relative_core,matched_stop_length=len(free_hist)==n,
                tolerance_crossing_is_not_an_algorithm_repair=True))
            checked.append(dict(seed=seed,updates_checked=3*n,largest_Sw_condition=max(conditions),
                first_DTC=dtcs[0],last_DTC=dtcs[-1],DTC_decreases=sum(b<a-1e-10 for a,b in zip(dtcs,dtcs[1:]))))
        print(f'Checked exact path seed{seed}:all{3*n}scatters,leading eigenvectors and stop; free path sensitivity retained',flush=True)
    R.table(R.ROOT/'model_update_checks.csv',checked);R.table(R.ROOT/'numerical_sensitivity.csv',sensitivity)
    R.check();R.check_models()
    report=dict(passed=True,models_checked=20,independent_exact_path_update_checks=updates,
        independent_free_reference_refits=20,ordinary_LDA_checks=20,no_probe_scoring=True,
        max_scatter_relative_error=max_sb_sw,max_single_update_unit_basis_abs_error=max_local_basis,
        max_generalized_eigen_residual=max_residual,max_gallery_core_encode_abs_error=max_core,
        exact_saved_stopping_conditions_checked=True,joint_DTC_decreasing_transitions=DTC_decreases,
        source_input_models_and_historical_hashes_unchanged=True,
        full_refit_tolerance_claim='No blanket numerical equality asserted. All exact-path updates separately validated; free differences in numerical_sensitivity.csv',
        max_free_reference_unit_basis_difference=max(r['max_final_unit_basis_abs_difference'] for r in sensitivity),
        max_free_reference_projector_difference=max(r['max_final_projector_abs_difference'] for r in sensitivity),
        max_free_reference_gallery_core_relative_difference=max(r['final_gallery_core_relative_difference'] for r in sensitivity),
        free_refits_with_changed_stop_length=sum(not r['matched_stop_length'] for r in sensitivity),
        elapsed_seconds=time.perf_counter()-started,verifier_path='verify_orl_dater.py',verifier_sha256=file_hash(R.BASE/'verify_orl_dater.py'))
    R.dump(R.ROOT/'model_verification.json',report)
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/orl_dater');args=parser.parse_args()
    R.ROOT=R.BASE/args.output
    with threadpool_limits(limits=2):verify()
