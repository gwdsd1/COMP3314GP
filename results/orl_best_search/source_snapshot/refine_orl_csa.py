"""Targeted G2/P8 refinement; all choices use only each split's gallery.

CSA-specific screening, richer mode ranks and training-derived HOSVD starts.
Ordinary Fisher and shrinkage remain separate; no seed is discarded.
"""
import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from csa import CSA, PCA, LDA, unfold
from gabor import file_hash
from audit_orl_protocol import LABELS, SEEDS, POLICIES, folds, split, prefix_nn
from optimize_csa_lda import fisher, coordinates, normalize as old_normalize, CASES as OLD_CASES
from run_available_reproduction import transform_batches
from run_orl_gray import write_csv

BASE = Path(__file__).resolve().parent
ROOT = BASE/'results/orl_csa_refined'
PREVIOUS = BASE/'results/csa_lda_optimized'
NORMS = ['none', 'channel_rms', 'sqrt', 'jet_l2', 'tensor_l2']
ALPHAS = [0., .01, .05, .15, .35, .65, 1.]
PARTIAL_FOLDS = [0, 4, 8, 12]
RANKS = [(4,2,3), (4,2,4), (3,3,3), (2,2,6), (2,3,4), (3,2,4),
         (2,4,4), (4,1,8), (2,2,8), (3,2,5), (2,3,5), (7,1,5)]
CASES = [{'algorithm':'pca','ranks':33,'center':True,'init':'identity'}]
CASES += [{'algorithm':'csa','ranks':r,'center':center,'init':'identity'}
          for r,center in [((4,2,3),False),((4,2,3),True),((4,2,4),False),((3,3,3),False)]]
CASES += [{'algorithm':'csa','ranks':r,'center':center,'init':'hosvd'} for r in RANKS for center in [False,True]]
CASES += [{'algorithm':'csa','ranks':(4,2,3),'center':center,'init':'random'} for center in [False,True]]
HIGH = [{'algorithm':'csa','ranks':(15,1,26),'center':center,'init':init}
        for center,init in [(False,'identity'),(False,'hosvd'),(True,'hosvd')]]


def dump(path, x):
    path.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def normalization(data, norm, learning):
    if norm in NORMS[:3]:
        return old_normalize({'gabor':data}, norm, learning)[0]['gabor']
    if norm == 'jet_l2':
        divisor = np.maximum(np.sqrt(np.sum(data**2, axis=-1, keepdims=True)), 1e-12)
    elif norm == 'tensor_l2':
        divisor = np.maximum(np.sqrt(np.sum(data**2, axis=(1,2,3), keepdims=True)), 1e-12)
    else:
        raise ValueError(norm)
    return (data/divisor).astype(np.float32)


def full_hosvd(x, center):
    a = np.asarray(x, dtype=float)
    if center:
        a = a-a.mean(0)
    bases = []
    for k in range(3):
        b = unfold(a, k)
        covariance = np.einsum('nac,nbc->ab', b, b, optimize=True)
        _, u = eigh(covariance)
        bases.append(u[:, ::-1])
    return bases


def fitted(data, learn, query, case, cache):
    if case['algorithm'] == 'pca':
        model = PCA(case['ranks'], center=True, solver='gram')
    else:
        initial = None
        if case['init'] == 'hosvd':
            if case['center'] not in cache:
                cache[case['center']] = full_hosvd(data[learn], case['center'])
            initial = [u[:, :r] for u,r in zip(cache[case['center']], case['ranks'])]
        model = CSA(case['ranks'], center=case['center'], init=case['init'], seed=3314,
                    max_iter=3, initial_bases=initial)
    model.fit(data[learn])
    return model, transform_batches(model,data,learn), transform_batches(model,data,query)


def prepare():
    ROOT.mkdir(exist_ok=False)
    catalogue = read(PREVIOUS/'catalogue.json')
    geometries = ['yunet_d22_zero','yunet_d20_replicate','yunet_d18_replicate','yunet_d16_replicate']
    kernels = ['adaptive_zero','fixed39_zero','adaptive_reflect','sigma_pi_zero','wavelength2_zero','mean2_zero']
    eligible = [e['id'] for e in catalogue if e['geometry'] in geometries and e['kernel'] in kernels]
    dump(ROOT/'catalogue.json', catalogue)
    dump(ROOT/'protocol.json', {'seeds':SEEDS,'gallery':2,'probe':8,'Tmax':3,'cases':CASES,'pure_cases':HIGH,
        'normalizations':NORMS,'alphas':ALPHAS,'policies':POLICIES,'eligible_input_ids':eligible,
        'screen':'CSA24 HOSVD noncentered fitted to own80gallery; centered cosine leave-self-out score; approximate input pruning only',
        'input_shortlist':'previous own ordinary+shrinkage input/norm anchors; best screen and best jet_l2/tensor_l2 screen; at most4',
        'partial_CV':'folds0/4/8/12,20 held-out gallery images,75learning/5validation each; all40classes retained',
        'case_shortlist':'top2 ordinary and top2 shrinkage per input, plus paper case/PCA/own previous cases',
        'full_CV':'all16 folds; each own gallery image validated once; partial counts reused, remaining12 replayed',
        'objective':'maximize Gabor CSA+LDA inner correct; ties favor earlier input, original case, alpha0, paper dimension23 then smaller dimension',
        'comparisons':'within each profile PCA/CSA390/CSA+LDA share input,norm,NN policy; PCA33 and CSA390 have their own training-selected parameters',
        'LDA_pre_limit':'<=35 in every inner fold and <=40 outside; shrinkage retains the same constraint',
        'initialization':'HOSVD bases from learning images only; fixed random3314; arbitrary orthogonal starts permitted by Procedure1, not asserted author configuration',
        'reporting':'ordinary LDA separate from shrinkage extension; original paper ranks retained as an anchor',
        'limitations':'finite two-stage pruning may miss optimum; historical probes seen in earlier development; not blind; no test-based seed/config selection',
        'references':['https://www.kolda.net/publication/koba09/','https://cenl.ucsd.edu/pubs/ieee-1997.pdf'],
        'sources':{n:file_hash(BASE/n) for n in ['refine_orl_csa.py','csa.py','optimize_csa_lda.py']}})
    (ROOT/'source_snapshot').mkdir()
    for n in ['refine_orl_csa.py','csa.py','optimize_csa_lda.py']:
        shutil.copy2(BASE/n, ROOT/'source_snapshot'/n)


def old_choices(seed):
    return read(PREVIOUS/f'seed{seed}/joint_selected.json')['profiles']


def old_case(choice):
    setting = next(m for m in choice['methods'] if m['method']=='CSA-3+LDA')
    old = OLD_CASES[setting['case_index']]
    return next(i for i,c in enumerate(CASES) if c['algorithm']=='csa' and c['init']=='identity'
                and tuple(c['ranks'])==tuple(old[3]) and c['center']==old[4]), setting


def screen(seed, catalogue):
    train,_ = split(LABELS,2,seed)
    rows = []
    for identifier in read(ROOT/'protocol.json')['eligible_input_ids']:
        entry = catalogue[identifier]
        gallery = np.asarray(np.load(BASE/entry['gabor_path'],mmap_mode='r')[train])
        for norm in NORMS:
            x = normalization(gallery,norm,np.arange(80))
            _,g,_ = fitted(x,np.arange(80),np.arange(1),
                           {'algorithm':'csa','ranks':(4,2,3),'center':False,'init':'hosvd'}, {})
            z = g-g.mean(0)
            matrix = cdist(z,z,'cosine')
            np.fill_diagonal(matrix,np.inf)
            count = int(np.sum(LABELS[train[np.argmin(matrix,axis=1)]]==LABELS[train]))
            rows.append({'entry_id':identifier,'normalization':norm,'gallery_LOO_correct':count})
    candidates = [{'entry_id':c['entry_id'],'normalization':c['normalization']} for c in old_choices(seed)]
    candidates = list({(c['entry_id'],c['normalization']):c for c in candidates}.values())
    winners = [max(rows,key=lambda r:r['gallery_LOO_correct']),
               max([r for r in rows if r['normalization'] in ['jet_l2','tensor_l2']],key=lambda r:r['gallery_LOO_correct'])]
    for winner in winners:
        candidate = {k:winner[k] for k in ['entry_id','normalization']}
        if candidate not in candidates:
            candidates.append(candidate)
    return rows,candidates


def cv(seed, candidate, indices, stage, catalogue, partial=None):
    folder = ROOT/f'seed{seed}'/stage
    folder.mkdir(exist_ok=True)
    path = folder/f'{candidate["entry_id"]}_{candidate["normalization"]}.npz'
    if path.exists():
        a = np.load(path)
        np.testing.assert_array_equal(a['cases'],indices)
        return a['counts'],path
    counts = np.zeros((6,len(indices),7,35),dtype=np.int16)
    if partial is not None:
        for j,case_index in enumerate(indices):
            counts[:,j] = partial[:,case_index]
    train,probe = split(LABELS,2,seed)
    original = np.load(BASE/catalogue[candidate['entry_id']]['gabor_path'],mmap_mode='r')
    chosen_folds = PARTIAL_FOLDS if stage=='partial' else [k for k in range(16) if k not in PARTIAL_FOLDS]
    for fi,(learn,valid) in enumerate(folds(train,seed)):
        if fi not in chosen_folds:
            continue
        assert not np.intersect1d(learn,probe).size and not np.intersect1d(valid,probe).size
        data = normalization(original,candidate['normalization'],learn)
        cache = {}
        for j,case_index in enumerate(indices):
            _,g,q = fitted(data,learn,valid,CASES[case_index],cache)
            assert g.shape[1] <= len(learn)-40
            base = LDA().fit(g,LABELS[learn])
            for ai,alpha in enumerate(ALPHAS):
                lda = fisher(g,LABELS[learn],alpha,base=base)
                for pi,policy in enumerate(POLICIES.values()):
                    a,b = coordinates(g,q,lda,policy)
                    pred,_ = prefix_nn(a,b,LABELS[learn],policy['metric'])
                    counts[pi,j,ai,1:a.shape[1]] += np.sum(pred[:,1:]==LABELS[valid,None],axis=0)
    np.savez_compressed(path,counts=counts,cases=np.array(indices),alphas=np.array(ALPHAS))
    return counts,path


def best_setting(counts, indices, policy_index, alphas, algorithm='csa'):
    pool = []
    for j,case_index in enumerate(indices):
        case = CASES[case_index]
        if case['algorithm'] != algorithm:
            continue
        p = int(np.prod(case['ranks']))
        paper_d = 25 if algorithm=='pca' else 23
        dimensions = ([paper_d] if paper_d <= p else [])+[d for d in range(2,p+1) if d!=paper_d]
        for ai in alphas:
            for d in dimensions:
                pool.append({'case_index':case_index,'alpha':ALPHAS[ai],'dimension':d,
                             'correct':int(counts[policy_index,j,ai,d-1])})
    return max(pool,key=lambda r:r['correct'])


def pure_cv(seed,candidate,catalogue):
    path = ROOT/f'seed{seed}'/f'pure_{candidate["entry_id"]}_{candidate["normalization"]}.npz'
    if path.exists():
        return np.load(path)['counts']
    counts = np.zeros((6,len(HIGH)),dtype=np.int16)
    train,_ = split(LABELS,2,seed)
    original = np.load(BASE/catalogue[candidate['entry_id']]['gabor_path'],mmap_mode='r')
    for learn,valid in folds(train,seed):
        data = normalization(original,candidate['normalization'],learn)
        cache = {}
        for ci,case in enumerate(HIGH):
            _,g,q = fitted(data,learn,valid,case,cache)
            for pi,policy in enumerate(POLICIES.values()):
                mean = g.mean(0) if policy['lda_center'] else 0.
                near = np.argmin(cdist(q-mean,g-mean,policy['metric']),axis=1)
                counts[pi,ci] += np.sum(LABELS[learn[near]]==LABELS[valid])
    np.savez_compressed(path,counts=counts)
    return counts


def select(seeds):
    catalogue = {e['id']:e for e in read(ROOT/'catalogue.json')}
    for seed in seeds:
        start = time.perf_counter()
        folder = ROOT/f'seed{seed}'
        folder.mkdir(exist_ok=True)
        assert not (folder/'selected.json').exists(), 'Refusing to replace frozen settings'
        train,probe = split(LABELS,2,seed)
        dump(folder/'indices.json',{'gallery':train.tolist(),'probe':probe.tolist()})
        if (folder/'screen.json').exists():
            screened = read(folder/'screen.json')
            candidates = screened['candidates']
        else:
            rows,candidates = screen(seed,catalogue)
            dump(folder/'screen.json',{'scores':rows,'candidates':candidates})
        options = []
        for candidate in candidates:
            partial,_ = cv(seed,candidate,list(range(len(CASES))),'partial',catalogue)
            keep = {0,1}
            for alphas in [[0],list(range(7))]:
                ranked = []
                for ci in range(1,len(CASES)):
                    value = max(int(np.max(partial[pi,ci,alphas,1:])) for pi in range(6))
                    ranked.append((value,ci))
                keep.update(ci for _,ci in sorted(ranked,key=lambda v:-v[0])[:2])
            for previous in old_choices(seed):
                if all(previous[k]==candidate[k] for k in ['entry_id','normalization']):
                    keep.add(old_case(previous)[0])
            indices = sorted(keep)
            full,path = cv(seed,candidate,indices,'full',catalogue,partial)
            for previous in old_choices(seed):
                if all(previous[k]==candidate[k] for k in ['entry_id','normalization']):
                    ci,setting = old_case(previous)
                    got = full[list(POLICIES).index(previous['policy']),indices.index(ci),ALPHAS.index(setting['alpha']),setting['dimension']-1]
                    assert int(got)==setting['correct'], (seed,candidate,int(got),setting)
            for profile,alphas in [('ordinary',[0]),('shrinkage',list(range(7)))]:
                for pi,policy in enumerate(POLICIES):
                    setting = best_setting(full,indices,pi,alphas)
                    options.append({**candidate,'profile':profile,'policy':policy,'CSA':setting,
                                    'PCA':best_setting(full,indices,pi,alphas,'pca'),
                                    'score_file':path.relative_to(ROOT).as_posix(),'score_sha256':file_hash(path)})
            print(f'seed{seed}: completed full gallery CV {candidate}, cases={indices}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
        winners = []
        for profile in ['ordinary','shrinkage']:
            winner = max([r for r in options if r['profile']==profile],key=lambda r:r['CSA']['correct'])
            high_counts = pure_cv(seed,winner,catalogue)
            pi = list(POLICIES).index(winner['policy'])
            hi = int(np.argmax(high_counts[pi]))
            winners.append({**winner,'pure':{'case_index':hi,'correct':int(high_counts[pi,hi])}})
        dump(folder/'candidate_scores.json',options)
        dump(folder/'selected.json',{'seed':seed,'training_only':True,'profiles':winners,
                                    'elapsed_seconds':time.perf_counter()-start})
        print(f'seed{seed} frozen: '+str([(w['profile'],w['CSA']['correct'],w['pure']['correct']) for w in winners]),flush=True)


def evaluate():
    assert all((ROOT/f'seed{s}/selected.json').exists() for s in SEEDS)
    out = ROOT/'evaluation'
    out.mkdir(exist_ok=False)
    (out/'embeddings').mkdir()
    freeze = {str(s):file_hash(ROOT/f'seed{s}/selected.json') for s in SEEDS}
    dump(out/'selection_freeze.json',freeze)
    catalogue = {e['id']:e for e in read(ROOT/'catalogue.json')}
    rows,inner = [],[]
    for seed in SEEDS:
        train,probe = split(LABELS,2,seed)
        for choice in read(ROOT/f'seed{seed}/selected.json')['profiles']:
            entry = catalogue[choice['entry_id']]
            assert file_hash(BASE/entry['gabor_path'])==entry['gabor_sha256']
            original = np.load(BASE/entry['gabor_path'],mmap_mode='r')
            policy = POLICIES[choice['policy']]
            for name,setting,cases in [('CSA-3+LDA',choice['CSA'],CASES),('PCA+LDA (Gabor)',choice['PCA'],CASES),('CSA-3',choice['pure'],HIGH)]:
                case = cases[setting['case_index']]
                count = 0
                for learn,valid in folds(train,seed):
                    data = normalization(original,choice['normalization'],learn)
                    _,g,q = fitted(data,learn,valid,case,{})
                    if name != 'CSA-3':
                        lda = fisher(g,LABELS[learn],setting['alpha'])
                        g,q = coordinates(g,q,lda,policy)
                        g,q = g[:,:setting['dimension']],q[:,:setting['dimension']]
                    else:
                        mean = g.mean(0) if policy['lda_center'] else 0.
                        g,q = g-mean,q-mean
                    near = np.argmin(cdist(q,g,policy['metric']),axis=1)
                    count += int(np.sum(LABELS[learn[near]]==LABELS[valid]))
                assert count==setting['correct'], (seed,name,count,setting)
                inner.append({'seed':seed,'profile':choice['profile'],'method':name,'correct':count,'n':80})
                data = normalization(original,choice['normalization'],train)
                model,g,q = fitted(data,train,probe,case,{})
                variants = [(choice['profile'],setting,case,model,g,q)]
                if name=='CSA-3+LDA' and choice['profile']=='ordinary':
                    anchor,ag,aq = fitted(data,train,probe,CASES[1],{})
                    variants.append(('same_input_paper_case',{'alpha':0.,'dimension':23},CASES[1],anchor,ag,aq))
                if name=='CSA-3+LDA' and choice['profile']=='shrinkage':
                    variants.append(('shrinkage_alpha0',{**setting,'alpha':0.},case,model,g,q))
                for profile,actual,case,model,g,q in variants:
                    if name=='CSA-3':
                        mean = g.mean(0) if policy['lda_center'] else 0.
                        a,b = g-mean,q-mean
                    else:
                        lda = fisher(g,LABELS[train],actual['alpha'],check=True)
                        a,b = coordinates(g,q,lda,policy)
                        a,b = a[:,:actual['dimension']],b[:,:actual['dimension']]
                    dist = cdist(b,a,policy['metric'])
                    assert np.isfinite(dist).all()
                    near = np.argmin(dist,axis=1)
                    predicted = LABELS[train[near]]
                    identifier = f'{seed}_{profile}_{name.replace(" ","_").replace("+","_")}'
                    saved = {'gallery':a,'probe':b,'train':train,'query':probe,'nearest':near,'predicted':predicted,
                             'core_gallery':g,'core_probe':q,'mean':model.mean}
                    if name!='CSA-3':
                        basis = lda.basis.copy()
                        if policy['scaling']=='unit_columns':
                            basis /= np.linalg.norm(basis,axis=0)
                        saved.update(lda_basis=basis[:,:actual['dimension']],sw=lda.sw,sb=lda.sb,
                                     eigenvalues=lda.eigenvalues[:actual['dimension']],
                                     lda_mean=lda.mean if policy['lda_center'] else np.zeros(g.shape[1]))
                    if case['algorithm']=='csa':
                        saved.update({f'axis{k}':u for k,u in enumerate(model.bases)})
                    else:
                        saved['pca_basis'] = model.basis
                    np.savez_compressed(out/'embeddings'/f'{identifier}.npz',**saved)
                    rows.append({'seed':seed,'profile':profile,'method':name,'entry_id':entry['id'],
                                 'normalization':choice['normalization'],'policy':choice['policy'],
                                 'ranks':str(case['ranks']),'center':case['center'],'init':case['init'],
                                 'alpha':actual.get('alpha',''),'dimension':actual.get('dimension',390),
                                 'inner_correct':count if profile==choice['profile'] else '',
                                 'correct':int(np.sum(predicted==LABELS[probe])),'probe_count':320,
                                 'accuracy_percent':100*float(np.mean(predicted==LABELS[probe])),
                                 'embedding':f'embeddings/{identifier}.npz'})
        write_csv(out/'results.csv',rows)
        print(f'Outer evaluation seed{seed}: all selected inner scores replayed',flush=True)
    assert freeze=={str(s):file_hash(ROOT/f'seed{s}/selected.json') for s in SEEDS}
    write_csv(out/'inner_scores.csv',inner)
    verify()


def verify():
    out = ROOT/'evaluation'
    rows = csv_rows(out/'results.csv')
    inner = {(r['seed'],r['profile'],r['method']):int(r['correct']) for r in csv_rows(out/'inner_scores.csv')}
    freeze = read(out/'selection_freeze.json')
    assert freeze=={str(s):file_hash(ROOT/f'seed{s}/selected.json') for s in SEEDS}
    catalogue = {e['id']:e for e in read(ROOT/'catalogue.json')}
    for seed in SEEDS:
        folder = ROOT/f'seed{seed}'
        options = []
        for candidate in read(folder/'screen.json')['candidates']:
            path = folder/'full'/f'{candidate["entry_id"]}_{candidate["normalization"]}.npz'
            counts = np.load(path)
            indices = counts['cases'].tolist()
            for profile,alphas in [('ordinary',[0]),('shrinkage',list(range(7)))]:
                for pi,policy in enumerate(POLICIES):
                    options.append({**candidate,'profile':profile,'policy':policy,
                                    'CSA':best_setting(counts['counts'],indices,pi,alphas),
                                    'PCA':best_setting(counts['counts'],indices,pi,alphas,'pca')})
        for choice in read(folder/'selected.json')['profiles']:
            expected = max([r for r in options if r['profile']==choice['profile']],key=lambda r:r['CSA']['correct'])
            for key in ['entry_id','normalization','policy','CSA','PCA']:
                assert choice[key]==expected[key]
            assert file_hash(ROOT/choice['score_file'])==choice['score_sha256']
            high = np.load(folder/f'pure_{choice["entry_id"]}_{choice["normalization"]}.npz')['counts'][list(POLICIES).index(choice['policy'])]
            assert choice['pure']=={'case_index':int(np.argmax(high)),'correct':int(high.max())}
    groups,shared = defaultdict(list),defaultdict(set)
    total,max_residual = 0,0.
    for r in rows:
        train,probe = split(LABELS,2,int(r['seed']))
        emb = np.load(out/r['embedding'])
        np.testing.assert_array_equal(train,emb['train'])
        np.testing.assert_array_equal(probe,emb['query'])
        a,b = emb['gallery'],emb['probe']
        if 'lda_basis' in emb:
            g,q = emb['core_gallery'],emb['core_probe']
            assert g.shape[1]<=len(train)-40
            sw,sb = np.zeros((g.shape[1],g.shape[1])),np.zeros((g.shape[1],g.shape[1]))
            for label in range(1,41):
                samples = g[LABELS[train]==label]
                delta = samples-samples.mean(0)
                sw += delta.T@delta
                d = samples.mean(0)-g.mean(0)
                sb += len(samples)*np.outer(d,d)
            np.testing.assert_allclose(sw,emb['sw'],rtol=1e-10,atol=1e-7)
            np.testing.assert_allclose(sb,emb['sb'],rtol=1e-10,atol=1e-7)
            alpha = float(r['alpha'])
            regularized = (1-alpha)*sw+alpha*np.trace(sw)/len(sw)*np.eye(len(sw))
            basis = emb['lda_basis']
            residual = np.linalg.norm(sb@basis-(regularized@basis)*emb['eigenvalues'])/max(1,np.linalg.norm(sb@basis))
            assert residual<1e-8
            max_residual = max(max_residual,float(residual))
            np.testing.assert_allclose((g-emb['lda_mean'])@basis,a,rtol=1e-9,atol=1e-8)
            np.testing.assert_allclose((q-emb['lda_mean'])@basis,b,rtol=1e-9,atol=1e-8)
        if POLICIES[r['policy']]['metric']=='cosine':
            distance = 1-(b/np.linalg.norm(b,axis=1)[:,None])@(a/np.linalg.norm(a,axis=1)[:,None]).T
        else:
            distance = np.sum(b*b,axis=1)[:,None]+np.sum(a*a,axis=1)[None,:]-2*b@a.T
        near = np.argmin(distance,axis=1)
        np.testing.assert_array_equal(near,emb['nearest'])
        np.testing.assert_array_equal(LABELS[train[near]],emb['predicted'])
        count = int(np.sum(emb['predicted']==LABELS[probe]))
        assert count==int(r['correct']) and abs(100*count/320-float(r['accuracy_percent']))<1e-10
        if r['inner_correct']:
            assert inner[r['seed'],r['profile'],r['method']]==int(r['inner_correct'])
        if r['profile'] in ['ordinary','same_input_paper_case','shrinkage_alpha0'] and r['alpha']:
            assert float(r['alpha'])==0.
        original = np.load(BASE/catalogue[r['entry_id']]['gabor_path'],mmap_mode='r')
        data = normalization(original,r['normalization'],train)
        for indices,key in [(train,'core_gallery'),(probe,'core_probe')]:
            x = np.asarray(data[indices],dtype=float)
            if 'pca_basis' in emb:
                encoded = (x.reshape(len(x),-1)-emb['mean'])@emb['pca_basis']
            else:
                encoded = np.einsum('nhwc,ha,wb,cd->nabd',x-emb['mean'],emb['axis0'],emb['axis1'],emb['axis2'],optimize=True).reshape(len(x),-1)
                for k in range(3):
                    u = emb[f'axis{k}']
                    np.testing.assert_allclose(u.T@u,np.eye(u.shape[1]),atol=1e-10)
            np.testing.assert_allclose(encoded,emb[key],rtol=1e-9,atol=1e-8)
        groups[r['profile'],r['method']].append(float(r['accuracy_percent']))
        shared[r['seed'],r['profile']].add((r['entry_id'],r['normalization'],r['policy']))
        total += 320
    assert len(rows)==80 and total==25600 and all(len(v)==1 for v in shared.values())
    summary = [{'profile':p,'method':m,'n_splits':len(v),'mean_percent':float(np.mean(v)),
                'std_pp':float(np.std(v,ddof=1)),'min_percent':min(v),'max_percent':max(v)} for (p,m),v in groups.items()]
    dump(out/'summary.json',summary)
    dump(out/'verification.json',{'passed':True,'scores':len(rows),'outer_predictions':total,'selected_inner_scores_replayed':len(inner),
                                  'independent_NN':True,'independent_encoding':True,'selection_freeze_unchanged':True,
                                  'independent_scatter_and_Fisher_equation':True,'max_relative_Fisher_residual':max_residual,
                                  'all_training_objectives_recomputed':True,
                                  'shared_input_and_distance':True,'source_sha256':file_hash(Path(__file__))})
    print(json.dumps(summary,indent=2),flush=True)


def randomness_plan():
    assert not (ROOT/'evaluation').exists(), 'Plan must precede current outer scoring'
    path = ROOT/'randomness_plan.json'
    assert not path.exists()
    choice = next(r for r in read(ROOT/'seed20/selected.json')['profiles'] if r['profile']=='ordinary')
    dump(path,{'seeds':list(range(1000,1100)),'input':choice['entry_id'],'normalization':choice['normalization'],
               'policy':choice['policy'],'refined_setting':choice['CSA'],
               'refined_case':CASES[choice['CSA']['case_index']],
               'selection_source':'seed20 ordinary winner selected only on its gallery; frozen before current outer results',
               'comparisons':'paper CSA24/LDA23; seed20 fixed ordinary config; CSA390 identity, noncentered',
               'purpose':'descriptive random-split sensitivity; never choose a seed/config from these scores',
               'limits':'same400images reused; not100 independent datasets or a fresh blind test'})


def randomness():
    assert read(ROOT/'evaluation/verification.json')['passed']
    plan = read(ROOT/'randomness_plan.json')
    out = ROOT/'randomness'
    out.mkdir(exist_ok=False)
    (out/'predictions').mkdir()
    catalogue = {e['id']:e for e in read(ROOT/'catalogue.json')}
    original = np.load(BASE/catalogue[plan['input']]['gabor_path'],mmap_mode='r')
    policy = POLICIES[plan['policy']]
    rows = []
    for seed in plan['seeds']:
        train,probe = split(LABELS,2,seed)
        data = normalization(original,plan['normalization'],train)
        cases = [('CSA24_LDA23',CASES[1],23),
                 ('fixed_refined_LDA',plan['refined_case'],plan['refined_setting']['dimension']),
                 ('CSA390',HIGH[0],None)]
        cache = {}
        for name,case,d in cases:
            _,g,q = fitted(data,train,probe,case,cache)
            if d is not None:
                lda = fisher(g,LABELS[train],0.)
                g,q = coordinates(g,q,lda,policy)
                g,q = g[:,:d],q[:,:d]
            else:
                mean = g.mean(0) if policy['lda_center'] else 0.
                g,q = g-mean,q-mean
            matrix = cdist(q,g,policy['metric'])
            near = np.argmin(matrix,axis=1)
            if policy['metric']=='cosine':
                independent = 1-(q/np.linalg.norm(q,axis=1)[:,None])@(g/np.linalg.norm(g,axis=1)[:,None]).T
            else:
                independent = np.sum(q*q,axis=1)[:,None]+np.sum(g*g,axis=1)[None,:]-2*q@g.T
            np.testing.assert_array_equal(near,np.argmin(independent,axis=1))
            pred = LABELS[train[near]]
            count = int(np.sum(pred==LABELS[probe]))
            np.savez_compressed(out/'predictions'/f'{seed}_{name}.npz',train=train,probe=probe,predicted=pred,nearest=near)
            rows.append({'seed':seed,'method':name,'correct':count,'n':320,'accuracy_percent':100*count/320})
        if seed%20==19:
            print(f'Fixed-configuration randomness study: {seed-999}/100 splits complete',flush=True)
    write_csv(out/'results.csv',rows)
    summary = []
    for name in dict.fromkeys(r['method'] for r in rows):
        group = [r for r in rows if r['method']==name]
        values = [r['accuracy_percent'] for r in group]
        target = 94.4 if name=='CSA390' else 90.9
        summary.append({'method':name,'n_splits':len(values),'mean_percent':float(np.mean(values)),
                        'std_pp':float(np.std(values,ddof=1)),'min_percent':min(values),'max_percent':max(values),
                        'max_seed':max(group,key=lambda r:r['accuracy_percent'])['seed'],
                        'at_least_paper_count':sum(v>=target for v in values),'paper_target':target})
    for row in csv_rows(out/'results.csv'):
        saved = np.load(out/'predictions'/f'{row["seed"]}_{row["method"]}.npz')
        train,probe = split(LABELS,2,int(row['seed']))
        np.testing.assert_array_equal(train,saved['train'])
        np.testing.assert_array_equal(probe,saved['probe'])
        np.testing.assert_array_equal(LABELS[train[saved['nearest']]],saved['predicted'])
        assert int(np.sum(saved['predicted']==LABELS[probe]))==int(row['correct'])
    dump(out/'summary.json',summary)
    dump(out/'verification.json',{'passed':True,'scores':len(rows),'predictions':len(rows)*320,
                                  'independent_NN':True,'all_splits_preserved':True,
                                  'plan_sha256':file_hash(ROOT/'randomness_plan.json')})
    print(json.dumps(summary,indent=2),flush=True)


def analyze():
    from decimal import Decimal, ROUND_HALF_UP
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def percent(v):
        return str(Decimal(str(v)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
    assert read(ROOT/'evaluation/verification.json')['passed']
    assert read(ROOT/'randomness/verification.json')['passed']
    current = csv_rows(ROOT/'evaluation/results.csv')
    previous = csv_rows(PREVIOUS/'evaluation/results.csv')
    fixed = csv_rows(BASE/'results/orl_sample_size/results.csv')
    paired = []
    for row in current:
        if row['method']!='CSA-3+LDA' or row['profile'] not in ['ordinary','shrinkage']:
            continue
        old_profile = 'optimized_ordinary_LDA' if row['profile']=='ordinary' else 'optimized_shrinkage_LDA'
        old = next(r for r in previous if r['seed']==row['seed'] and r['profile']==old_profile and r['method']==row['method'])
        control = next(r for r in current if r['seed']==row['seed'] and r['profile']==('same_input_paper_case' if row['profile']=='ordinary' else 'shrinkage_alpha0'))
        paired.append({'seed':row['seed'],'profile':row['profile'],'previous_percent':float(old['accuracy_percent']),
                       'current_percent':float(row['accuracy_percent']),
                       'change_pp':float(row['accuracy_percent'])-float(old['accuracy_percent']),
                       'same_input_control_percent':float(control['accuracy_percent']),
                       'control_change_pp':float(row['accuracy_percent'])-float(control['accuracy_percent'])})
    write_csv(ROOT/'paired_changes.csv',paired)
    stats = []
    for profile in ['ordinary','shrinkage']:
        group = [r for r in paired if r['profile']==profile]
        changes = np.array([r['change_pp'] for r in group])
        stats.append({'profile':profile,'mean_change_pp':float(changes.mean()),
                      'improved_splits':int(np.sum(changes>0)),'tied_splits':int(np.sum(changes==0)),
                      'worse_splits':int(np.sum(changes<0)),
                      'same_input_control_mean_change_pp':float(np.mean([r['control_change_pp'] for r in group]))})
    settings = []
    for seed in SEEDS:
        for choice in read(ROOT/f'seed{seed}/selected.json')['profiles']:
            case = CASES[choice['CSA']['case_index']]
            settings.append({'seed':seed,'profile':choice['profile'],'input':choice['entry_id'],
                             'normalization':choice['normalization'],'policy':choice['policy'],
                             'ranks':str(case['ranks']),'center':case['center'],'init':case['init'],
                             'LDA_dimension':choice['CSA']['dimension'],'alpha':choice['CSA']['alpha'],
                             'inner_correct':choice['CSA']['correct'],'inner_total':80})
    write_csv(ROOT/'selected_settings.csv',settings)
    energy_rows = []
    catalogue = {e['id']:e for e in read(ROOT/'catalogue.json')}
    for row in current:
        if row['method']!='CSA-3+LDA' or row['profile'] not in ['ordinary','same_input_paper_case']:
            continue
        emb = np.load(ROOT/'evaluation'/row['embedding'])
        train = emb['train']
        data = np.load(BASE/catalogue[row['entry_id']]['gabor_path'],mmap_mode='r')
        x = normalization(data,row['normalization'],train)[train].astype(float).reshape(80,-1)
        g = emb['core_gallery']
        total = np.sum((x-x.mean(0))**2)
        between_x,between_g = 0.,0.
        for label in range(1,41):
            mask = LABELS[train]==label
            between_x += mask.sum()*np.sum((x[mask].mean(0)-x.mean(0))**2)
            between_g += mask.sum()*np.sum((g[mask].mean(0)-g.mean(0))**2)
        energy_rows.append({'seed':row['seed'],'profile':row['profile'],
                            'centered_total_retained_percent':100*np.sum((g-g.mean(0))**2)/total,
                            'between_class_retained_percent':100*between_g/between_x})
    write_csv(ROOT/'training_energy.csv',energy_rows)
    dump(ROOT/'analysis.json',{'paired':stats,'selected_HOSVD_count':sum(r['init']=='hosvd' for r in settings),
                              'selected_centered_count':sum(r['center'] for r in settings),
                              'training_energy_means':{p:{key:float(np.mean([r[key] for r in energy_rows if r['profile']==p]))
                                  for key in ['centered_total_retained_percent','between_class_retained_percent']}
                                  for p in ['ordinary','same_input_paper_case']},
                              'all10_seeds_retained':True,'ranking_not_based_on_probe':True})
    def values(rows,**filter):
        return np.array([float(r['accuracy_percent']) for r in rows if all(r[k]==v for k,v in filter.items())])
    series = [('Fixed CSA390',values(fixed,context='anchor_P8',method='CSA390'),94.4),
              ('Refined CSA390',values(current,profile='ordinary',method='CSA-3'),94.4),
              ('Fixed CSA24/LDA23',values(fixed,context='anchor_P8',method='CSA24_LDA23'),90.9),
              ('Previous ordinary',values(previous,profile='optimized_ordinary_LDA',method='CSA-3+LDA'),90.9),
              ('Refined ordinary',values(current,profile='ordinary',method='CSA-3+LDA'),90.9),
              ('Previous shrinkage',values(previous,profile='optimized_shrinkage_LDA',method='CSA-3+LDA'),90.9),
              ('Refined shrinkage',values(current,profile='shrinkage',method='CSA-3+LDA'),90.9)]
    assert all(len(v)==10 for _,v,_ in series)
    fig,ax = plt.subplots(figsize=(11.8,5.5),layout='constrained')
    for i,(name,v,target) in enumerate(series):
        color = '#bc782b' if 'shrinkage' in name else '#246d9b'
        ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),fmt='o',color=color,capsize=5,zorder=3)
        ax.scatter(np.full(10,i)+np.linspace(-.12,.12,10),v,color=color,alpha=.4,s=18)
        ax.text(i,v.mean()+v.std(ddof=1)+.7,f'{percent(v.mean())}%',ha='center',fontsize=10)
    ax.axhline(94.4,color='#528d4e',ls='--',label='Paper CSA: 94.4%')
    ax.axhline(90.9,color='#9d5284',ls='--',label='Paper ordinary CSA+LDA: 90.9%')
    ax.set_xticks(range(len(series)),[n.replace(' ','\n',1) for n,_,_ in series],fontsize=9)
    ax.set_ylabel('Accuracy (%)');ax.set_ylim(72,100)
    ax.set_title('ORL G2/P8: all 10 splits; dots = runs, error bars = sample SD')
    ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left')
    fig.savefig(ROOT/'refinement_comparison.png',dpi=170);plt.close(fig)
    random_rows = csv_rows(ROOT/'randomness/results.csv')
    delta = values(random_rows,method='fixed_refined_LDA')-values(random_rows,method='CSA24_LDA23')
    analysis = read(ROOT/'analysis.json')
    analysis['fixed100_paired_change'] = {'mean_pp':float(delta.mean()),'improved_splits':int(np.sum(delta>0)),
                                        'tied_splits':int(np.sum(delta==0)),'worse_splits':int(np.sum(delta<0))}
    analysis['analysis_source_sha256'] = file_hash(Path(__file__))
    dump(ROOT/'analysis.json',analysis)
    fig,axes = plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,name,target in zip(axes,['CSA24_LDA23','fixed_refined_LDA','CSA390'],[90.9,90.9,94.4]):
        v = values(random_rows,method=name)
        ax.hist(v,bins=np.arange(65,99,.9375),color='#246d9b',alpha=.85)
        ax.axvline(v.mean(),color='#bc782b',label=f'Mean {percent(v.mean())}%')
        ax.axvline(target,color='#9d5284',ls='--',label=f'Paper {target}%')
        ax.set_title(f'{name}\nMax {v.max():.2f}%, SD {v.std(ddof=1):.2f}pp')
        ax.set_xlabel('Accuracy (%)');ax.set_ylabel('Number of splits');ax.legend(fontsize=8)
    fig.suptitle('100 predeclared random splits, one fixed input and configuration')
    fig.savefig(ROOT/'randomness_distribution.png',dpi=170);plt.close(fig)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['prepare','select','evaluate','verify','randomness-plan','randomness','analyze'])
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    parser.add_argument('--output',default='results/orl_csa_refined')
    args = parser.parse_args()
    ROOT = (BASE/args.output).resolve()
    assert ROOT.is_relative_to(BASE) and all(s in SEEDS for s in args.seeds)
    with threadpool_limits(limits=2):
        if args.stage=='prepare':
            prepare()
        elif args.stage=='select':
            select(args.seeds)
        elif args.stage=='evaluate':
            evaluate()
        elif args.stage=='randomness-plan':
            randomness_plan()
        elif args.stage=='randomness':
            randomness()
        elif args.stage=='analyze':
            analyze()
        else:
            verify()
