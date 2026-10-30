"""Read back frozen ORL gray recovery models, predictions, and historical evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

import run_orl_gray_recovery as R


def reference_core(images, model):
    if 'u0' in model:
        return np.einsum('ha,nhw,wb->nab', model['u0'], images-model['mean'],
                         model['u1'], optimize=False).reshape(len(images), 22)
    return (images.reshape(len(images), -1)-model['mean']) @ model['basis']


def check_trajectory(images, start, saved):
    """Independent sample-matrix covariance route for selected full trajectories."""
    centered = start=='hosvd_centered'
    z = images-images.mean(axis=0) if centered else images.copy()
    if start=='full_identity':
        u, v = np.eye(56), np.eye(46)
    elif start.startswith('hosvd'):
        _, a = eigh(sum(x@x.T for x in z)); u = a[:, -11:]
        _, a = eigh(sum(x.T@x for x in z)); v = a[:, -2:]
    else:
        u, v = np.eye(56, 11), np.eye(46, 2)
    previous = np.inf
    for _ in range(3):
        projected = [x@v for x in z]
        c = sum(x@x.T for x in projected)
        _, a = eigh(c); u = a[:, -11:]
        projected = [u.T@x for x in z]
        c = sum(x.T@x for x in projected)
        _, a = eigh(c); v = a[:, -2:]
        error = sum(np.sum((x-u@(u.T@x@v)@v.T)**2) for x in z)
        assert error <= previous+1e-4
        previous = error
    errors = [float(np.linalg.norm(a@a.T-b@b.T))
              for a,b in [(u,saved['u0']), (v,saved['u1'])]]
    assert max(errors)<2e-7, (start, errors)
    return max(errors)


def verify(root):
    p = R.read(root/'protocol.json')
    for name, digest in p['source_sha256'].items():
        assert R.file_hash(R.BASE/name)==digest
        assert R.file_hash(root/'source_snapshot'/name)==digest
    for name, digest in R.read(root/'input_freeze.json').items():
        assert R.file_hash(root/name)==digest
    assert R.file_hash(root/'models.csv')==R.read(root/'model_freeze.json')['sha256']
    assert all(R.file_hash(R.BASE/name)==digest for name,digest in
               R.read(root/'historical_evidence_hashes.json').items())
    splits = {r['seed']:r for r in R.read(root/'splits.json')}
    scores = {(r['input'],int(r['seed']),r['start'],r['policy']):r
              for r in R.rows(root/'results.csv')}
    assert len(scores)==4320
    analysis = R.read(root/'analysis.json')
    best = analysis['best_mean']
    reference_error, core_error, fisher_error = 0., 0., 0.
    checked = 0
    for row in R.rows(root/'models.csv'):
        seed = int(row['seed'])
        s = splits[seed]
        g, q = np.array(s['gallery']), np.array(s['probe'])
        assert len(g)==80 and len(q)==320 and not set(g)&set(q)
        assert sorted(np.r_[g,q])==list(range(400))
        np.testing.assert_array_equal(np.bincount(R.LABELS[g])[1:],np.full(40,2))
        np.testing.assert_array_equal(np.bincount(R.LABELS[q])[1:],np.full(40,8))
        assert R.file_hash(root/row['model'])==row['sha256']
        model = np.load(root/row['model'])
        np.testing.assert_array_equal(model['gallery'],g)
        data = np.asarray(R.load_data(root,row['input'],seed),dtype=float)
        gallery, query = reference_core(data[g],model), reference_core(data[q],model)
        core_error = max(core_error,float(np.max(np.abs(gallery-model['gallery_core']))))
        np.testing.assert_allclose(gallery,model['gallery_core'],rtol=1e-10,atol=1e-8)
        if row['start'] in R.STARTS:
            assert model['u0'].shape==(56,11) and model['u1'].shape==(46,2)
            for u in [model['u0'],model['u1']]:
                np.testing.assert_allclose(u.T@u,np.eye(u.shape[1]),atol=1e-10)
            if row['input'] in {'historical_shared',best['input']}:
                reference_error = max(reference_error,check_trajectory(data[g],row['start'],model))
        mean = gallery.mean(axis=0)
        sw = np.zeros((gallery.shape[1],gallery.shape[1])); sb = sw.copy()
        for label in range(1,41):
            x = gallery[R.LABELS[g]==label]
            delta=x-x.mean(axis=0)
            sw+=delta.T@delta
            delta=x.mean(axis=0)-mean
            sb+=len(x)*np.outer(delta,delta)
        w=model['lda_basis']; ev=model['eigenvalues']
        residual=float(np.linalg.norm(sb@w-(sw@w)*ev)/max(1.,np.linalg.norm(sb@w)))
        assert residual<1e-8
        fisher_error=max(fisher_error,residual)
        saved=np.load(root/'predictions'/Path(row['model']).name)
        np.testing.assert_array_equal(saved['probe'],q)
        for i,policy in enumerate(R.POLICIES):
            w=model['lda_basis'].copy()
            if policy['scale']=='unit':w/=np.sqrt(np.sum(w*w,axis=0))
            origin=mean if policy['origin']=='gallery' else 0.
            a,b=(query-origin)@w,(gallery-origin)@w
            if policy['metric']=='cosine':
                distances=1-(a/np.linalg.norm(a,axis=1)[:,None])@(b/np.linalg.norm(b,axis=1)[:,None]).T
            else:
                distances=np.array([np.sum((b-v)**2,axis=1) for v in a])
            neighbors=g[distances.argmin(axis=1)]
            pred=R.LABELS[neighbors]
            np.testing.assert_array_equal(pred,saved['predicted'][i])
            np.testing.assert_array_equal(neighbors,saved['neighbor'][i])
            score=scores[(row['input'],seed,row['start'],policy['id'])]
            assert int(score['correct'])==int(np.sum(pred==R.LABELS[q]))
            assert abs(float(score['accuracy_percent'])-100*int(score['correct'])/320)<1e-12
            checked+=1
    summaries=R.rows(root/'summary.csv')
    for r in summaries:
        values=[float(scores[(r['input'],seed,r['start'],r['policy'])]['accuracy_percent']) for seed in R.SEEDS]
        np.testing.assert_allclose([np.mean(values),np.std(values,ddof=1),min(values),max(values)],
            [float(r[k]) for k in ['mean_percent','std_pp','min_percent','max_percent']],atol=1e-12)
    assert len(summaries)==432 and checked==4320
    table=[]
    anchor=R.read(root/'feature_manifest.json')[0]['historical_sources']
    for seed in R.SEEDS:
        policy=anchor[str(seed)]['shared_policy']
        name=('unit' if policy['scaling']=='unit_columns' else 'sw')+'_'+('gallery' if policy['lda_center'] else 'linear')+'_'+policy['metric']
        old=float(scores[('historical_shared',seed,'rectangular_identity',name)]['accuracy_percent'])
        new=float(scores[(best['input'],seed,best['start'],best['policy'])]['accuracy_percent'])
        controls=[float(scores[(best['input'],seed,k,best['policy'])]['accuracy_percent']) for k in R.CONTROLS]
        table.append(dict(seed=seed,old_shared=old,best_CSA=new,change_pp=new-old,
                          matched_PCA30_LDA18=controls[0],matched_PCA22_LDA16=controls[1]))
    R.table(root/'best_paired_comparison.csv',table)
    result=dict(passed=True,models=720,CSA_models=480,PCA_models=240,scores=checked,
        predictions=checked*320,summary_rows=len(summaries),
        max_reference_projector_error=reference_error,max_gallery_core_abs_error=core_error,
        max_independent_fisher_residual=fisher_error,
        all_saved_predictions_recomputed=True,source_input_model_hashes_match=True,
        historical_evidence_unchanged=True,all_splits_retained=True,
        improvement_splits=sum(r['change_pp']>0 for r in table),
        tied_splits=sum(r['change_pp']==0 for r in table),
        lower_splits=sum(r['change_pp']<0 for r in table),
        winner_is_exploratory=True,author_settings_recovered=False)
    R.dump(root/'verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/orl_gray_recovery_20261030')
    args=parser.parse_args()
    with threadpool_limits(limits=2):verify(R.BASE/args.output)


if __name__=='__main__':main()
