"""One finite, probe-informed crop neighborhood after the gray recovery study.

Six new geometries; two uncentered rank-conforming starts; two PCA controls.
The parent batch is immutable. No further automatic expansion follows.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from PIL import Image

import run_orl_gray_recovery as R
import verify_orl_gray_recovery as V


def run(root):
    root.mkdir(exist_ok=False)
    for name in ['features','models','predictions','source_snapshot']:
        (root/name).mkdir()
    parent=root.parent
    assert R.read(parent/'scoring_checks.json')['passed']
    parent_hashes={name:R.file_hash(parent/name) for name in
                   ['protocol.json','splits.json','summary.csv','results.csv','analysis.json']}
    specs=[dict(id=f'd{d}_y{y}',distance=float(d),eye_y=float(y))
           for d,y in [(14,24),(15,24),(17,24),(16,22),(16,26),(12,24)]]
    starts=['rectangular_identity','hosvd']
    methods=starts+R.CONTROLS
    files=R.SOURCES+['run_orl_gray_refinement.py']
    for name in files:shutil.copy2(R.BASE/name,root/'source_snapshot'/name)
    protocol=dict(date='2026-10-30',inputs=specs,starts=starts,methods=methods,
        policies=R.POLICIES,CSA_ranks=[11,2],CSA_LDA=16,Tmax=3,center=False,
        planned_CSA_fits=120,planned_PCA_fits=120,planned_scores=1440,
        eye_annotations='same frozen YuNet eyes as parent',histogram='PIL',
        interpolation='linear',warp_border='replicate',input_size=[56,46],
        gallery_per_class=2,probe_per_class=8,seeds=R.SEEDS,
        parent_sha256=parent_hashes,source_sha256={f:R.file_hash(R.BASE/f) for f in files},
        selection='Neighborhood chosen after parent probe results; best global mean is exploratory, not blind.',
        stopping='This six-geometry neighborhood is the final crop refinement; report remaining gaps without automatic expansion.',
        limits=R.read(parent/'protocol.json')['limits'])
    R.dump(root/'protocol.json',protocol)
    shutil.copy2(parent/'splits.json',root/'splits.json')
    raw=np.stack([np.asarray(Image.open(R.BASE/'data/raw/orl'/n).convert('L')) for n in R.NAMES])
    eye_rows=R.rows(R.BASE/'results/orl_protocol_audit/yunet_eye_coordinates.csv')
    assert [r['file'] for r in eye_rows]==R.NAMES
    eyes=np.array([[[float(r['left_x']),float(r['left_y'])],
                    [float(r['right_x']),float(r['right_y'])]] for r in eye_rows])
    manifests=[]
    for spec in specs:
        target=np.array([[23-spec['distance']/2,spec['eye_y']],
                         [23+spec['distance']/2,spec['eye_y']]])
        array=[]
        for image,eye in zip(raw,eyes):
            matrix=R.alignment_matrix(eye,target)
            np.testing.assert_allclose(np.c_[eye,np.ones(2)]@matrix.T,target,atol=1e-10)
            warped=R.cv2.warpAffine(image,matrix,(46,56),flags=R.cv2.INTER_LINEAR,
                                    borderMode=R.cv2.BORDER_REPLICATE)
            array.append(R.equalize(warped,'PIL'))
        path=root/'features'/f"{spec['id']}.npy"
        np.save(path,np.stack(array))
        manifests.append(dict(input=spec['id'],path=path.relative_to(R.BASE).as_posix(),sha256=R.file_hash(path)))
    R.dump(root/'feature_manifest.json',manifests)
    R.dump(root/'input_freeze.json',{f:R.file_hash(root/f) for f in
            ['protocol.json','splits.json','feature_manifest.json']})
    fitted=[]
    splits=R.read(root/'splits.json')
    for spec in specs:
        data=np.asarray(R.load_data(root,spec['id'],20),dtype=float)
        for s in splits:
            g=np.array(s['gallery']); x=data[g]
            for method in methods:
                if method in R.CONTROLS:
                    rank,dimension=(30,18) if method==R.CONTROLS[0] else (22,16)
                    model=R.PCA(rank,solver='gram').fit(x)
                    saved=dict(basis=model.basis,mean=model.mean)
                    history=[]
                else:
                    dimension=16
                    model,history=R.fit_csa(x,method)
                    saved=dict(u0=model.bases[0],u1=model.bases[1],mean=model.mean)
                core=model.transform(x).reshape(80,-1)
                lda=R.LDA().fit(core,R.LABELS[g])
                values,_=eigh(lda.sb,lda.sw)
                np.testing.assert_allclose(values[::-1][:dimension],lda.eigenvalues[:dimension],rtol=1e-7,atol=1e-7)
                saved.update(lda_basis=lda.basis[:,:dimension],lda_mean=lda.mean,
                             eigenvalues=lda.eigenvalues[:dimension],gallery=g,gallery_core=core)
                filename=f"{spec['id']}_s{s['seed']}_{method}.npz"
                np.savez(root/'models'/filename,**saved)
                fitted.append(dict(input=spec['id'],seed=s['seed'],start=method,
                    model='models/'+filename,sha256=R.file_hash(root/'models'/filename),trajectory=json.dumps(history)))
        print('Refinement fitted all ten:',spec['id'],flush=True)
    R.table(root/'models.csv',fitted)
    R.dump(root/'model_freeze.json',dict(count=len(fitted),sha256=R.file_hash(root/'models.csv'),
                                      all_fits_before_refinement_probe_scoring=True))
    results=[]; reference_error=0.; core_error=0.; fisher_error=0.
    for row in fitted:
        seed=row['seed']; s=next(r for r in splits if r['seed']==seed)
        g,q=np.array(s['gallery']),np.array(s['probe'])
        assert not set(g)&set(q) and sorted(np.r_[g,q])==list(range(400))
        np.testing.assert_array_equal(np.bincount(R.LABELS[g])[1:],np.full(40,2))
        np.testing.assert_array_equal(np.bincount(R.LABELS[q])[1:],np.full(40,8))
        assert R.file_hash(root/row['model'])==row['sha256']
        model=np.load(root/row['model'])
        data=np.asarray(R.load_data(root,row['input'],seed),dtype=float)
        core,query=V.reference_core(data[g],model),V.reference_core(data[q],model)
        np.testing.assert_allclose(core,model['gallery_core'],rtol=1e-10,atol=1e-8)
        core_error=max(core_error,float(np.max(np.abs(core-model['gallery_core']))))
        if row['start'] in starts and seed in [20,29]:
            reference_error=max(reference_error,V.check_trajectory(data[g],row['start'],model))
        mean=core.mean(axis=0); sw=np.zeros((core.shape[1],core.shape[1])); sb=sw.copy()
        for label in range(1,41):
            x=core[R.LABELS[g]==label]; delta=x-x.mean(axis=0)
            sw+=delta.T@delta
            delta=x.mean(axis=0)-mean;sb+=len(x)*np.outer(delta,delta)
        w=model['lda_basis']
        residual=float(np.linalg.norm(sb@w-(sw@w)*model['eigenvalues'])/max(1.,np.linalg.norm(sb@w)))
        assert residual<1e-8
        fisher_error=max(fisher_error,residual)
        predictions=[];neighbors=[]
        for policy in R.POLICIES:
            w=model['lda_basis'].copy()
            if policy['scale']=='unit':w/=np.linalg.norm(w,axis=0)
            origin=model['lda_mean'] if policy['origin']=='gallery' else 0.
            a,b=(query-origin)@w,(core-origin)@w
            chosen=cdist(a,b,metric=policy['metric']).argmin(axis=1)
            if policy['metric']=='cosine':
                independent=((a/np.linalg.norm(a,axis=1)[:,None])@(b/np.linalg.norm(b,axis=1)[:,None]).T).argmax(axis=1)
            else:
                independent=np.array([np.sum((b-v)**2,axis=1).argmin() for v in a])
            np.testing.assert_array_equal(chosen,independent)
            pred=R.LABELS[g[chosen]]; correct=int(np.sum(pred==R.LABELS[q]))
            results.append(dict(input=row['input'],seed=seed,start=row['start'],policy=policy['id'],
                                correct=correct,probe_count=320,accuracy_percent=correct/3.2))
            predictions.append(pred);neighbors.append(g[chosen])
        path=root/'predictions'/Path(row['model']).name
        np.savez_compressed(path,probe=q,predicted=np.array(predictions),neighbor=np.array(neighbors))
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['predicted'],predictions)
            np.testing.assert_array_equal(saved['neighbor'],neighbors)
    R.table(root/'results.csv',results)
    summary=[]
    for spec in specs:
        for method in methods:
            for policy in R.POLICIES:
                a=[r for r in results if (r['input'],r['start'],r['policy'])==(spec['id'],method,policy['id'])]
                assert sorted(r['seed'] for r in a)==R.SEEDS
                values=np.array([r['accuracy_percent'] for r in a])
                target=84.1 if method in starts else (82.2 if method==R.CONTROLS[0] else None)
                summary.append(dict(input=spec['id'],start=method,policy=policy['id'],n_splits=10,
                    mean_percent=float(values.mean()),std_pp=float(values.std(ddof=1)),min_percent=float(values.min()),
                    max_percent=float(values.max()),paper_percent=target if target is not None else '',
                    mean_minus_paper_pp=float(values.mean()-target) if target is not None else ''))
    R.table(root/'summary.csv',summary)
    best=max([r for r in summary if r['start'] in starts],key=lambda r:r['mean_percent'])
    single=max([r for r in results if r['start'] in starts],key=lambda r:r['accuracy_percent'])
    matched=[r for r in summary if r['input']==best['input'] and r['policy']==best['policy']]
    R.dump(root/'analysis.json',dict(best_mean=best,highest_single_run=single,matched_settings=matched,
        selection=protocol['selection'],stopping=protocol['stopping'],limits=protocol['limits']))
    assert all(R.file_hash(parent/name)==digest for name,digest in parent_hashes.items())
    assert all(R.file_hash(R.BASE/name)==digest for name,digest in protocol['source_sha256'].items())
    assert all(R.file_hash(root/name)==digest for name,digest in R.read(root/'input_freeze.json').items())
    assert R.file_hash(root/'models.csv')==R.read(root/'model_freeze.json')['sha256']
    R.dump(root/'verification.json',dict(passed=True,models=len(fitted),CSA_models=120,PCA_models=120,
        scores=len(results),predictions=len(results)*320,summary_rows=len(summary),
        independent_NN_matches=True,saved_prediction_readback_matches=True,
        max_reference_projector_error=reference_error,max_gallery_core_abs_error=core_error,
        max_independent_fisher_residual=fisher_error,all_ten_splits_retained=True,
        parent_evidence_unchanged=True,source_input_model_hashes_match=True,
        winner_is_exploratory=True,author_settings_recovered=False))
    print(json.dumps(dict(best=best,max_run=single),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/orl_gray_recovery_20261030/refinement')
    args=parser.parse_args()
    with threadpool_limits(limits=2):run(R.BASE/args.output)


if __name__=='__main__':main()
