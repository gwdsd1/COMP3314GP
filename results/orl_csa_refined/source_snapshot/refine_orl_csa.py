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
    groups,shared = defaultdict(list),defaultdict(set)
    total = 0
    for r in rows:
        train,probe = split(LABELS,2,int(r['seed']))
        emb = np.load(out/r['embedding'])
        np.testing.assert_array_equal(train,emb['train'])
        np.testing.assert_array_equal(probe,emb['query'])
        a,b = emb['gallery'],emb['probe']
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
                                  'shared_input_and_distance':True,'source_sha256':file_hash(Path(__file__))})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['prepare','select','evaluate','verify'])
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
        else:
            verify()
