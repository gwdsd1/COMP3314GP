"""Finite exploratory ORL best-result search, explicitly selected on probes.

Keep the previous gallery-selected evaluation intact. All 200 planned splits,
all valid score cells and near-paper runs are retained; never stop on success.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import refine_orl_csa as r
from csa import CSA, LDA, PCA
from gabor import file_hash
from run_available_reproduction import transform_batches
from run_orl_gray import write_csv

BASE = Path(__file__).resolve().parent
ROOT = BASE/'results/orl_best_search'
SEEDS = list(range(3000,3200))
BRANCHES = [
    {'input':'yunet_d20_replicate_fixed39_zero','normalization':'none'},
    {'input':'yunet_d20_replicate_fixed39_zero','normalization':'power075'},
    {'input':'yunet_d18_zero_fixed39_zero','normalization':'sqrt'},
]
CASES = [
    {'family':'CSA+LDA','ranks':(4,2,3),'center':False,'init':'identity'},
    {'family':'CSA+LDA','ranks':(4,2,4),'center':False,'init':'identity'},
    {'family':'CSA+LDA','ranks':(4,1,8),'center':False,'init':'hosvd'},
    {'family':'CSA+LDA','ranks':(2,2,8),'center':True,'init':'hosvd'},
    {'family':'CSA+LDA','ranks':(2,1,16),'center':False,'init':'hosvd'},
    {'family':'CSA+LDA','ranks':(3,1,11),'center':True,'init':'hosvd'},
    {'family':'CSA+LDA','ranks':(4,2,5),'center':False,'init':'hosvd'},
    {'family':'CSA+LDA','ranks':(4,2,5),'center':True,'init':'hosvd'},
    {'family':'CSA+LDA','ranks':(4,1,10),'center':False,'init':'hosvd'},
    {'family':'CSA+LDA','ranks':(4,1,10),'center':True,'init':'hosvd'},
    {'family':'CSA','ranks':(15,1,26),'center':False,'init':'identity'},
    {'family':'CSA','ranks':(15,1,26),'center':False,'init':'hosvd'},
    {'family':'CSA','ranks':(14,2,20),'center':False,'init':'hosvd'},
    {'family':'CSA','ranks':(12,3,20),'center':False,'init':'hosvd'},
    {'family':'CSA','ranks':(14,4,20),'center':False,'init':'hosvd'},
]
POLICIES = list(r.POLICIES)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def prepare():
    ROOT.mkdir(exist_ok=False)
    catalogue = {e['id']:e for e in read(BASE/'results/csa_lda_optimized/catalogue.json')}
    entries = [catalogue[name] for name in dict.fromkeys(b['input'] for b in BRANCHES)]
    for entry in entries:
        assert file_hash(BASE/entry['gabor_path'])==entry['gabor_sha256']
    sources = ['search_orl_best.py','csa.py','refine_orl_csa.py','optimize_csa_lda.py',
               'audit_orl_protocol.py','run_available_reproduction.py']
    dump(ROOT/'protocol.json',{'seeds':SEEDS,'branches':BRANCHES,'cases':CASES,'policies':POLICIES,
        'gallery_per_class':2,'probe_per_class':8,'N':80,'Nc':40,'Tmax':3,
        'LDA':'ordinary Fisher only, alpha=0; pre dimension <=40, final <=39; numerical nullspace removal is unchanged',
        'LDA_dimensions':'all valid dimensions 2 through min(pre dimension,39)',
        'pure_policies':'first three policies only; LDA column scaling is inapplicable to pure CSA',
        'inputs':'existing fixed eye-aligned 56x46 gray / 28x23x40 Gabor, five scales and eight orientations',
        'new_conventions':'power0.75 amplitudes; alternative mode ranks; training-derived HOSVD start; pure dimensions390/560/720/1120',
        'selection':'EXPLORATORY: maxima across configurations AND splits selected using probe accuracy; no independent validation claim',
        'reporting':'all scores saved; report best brackets, seed, mean and SD for the corresponding fixed config, grid size and split count',
        'near_threshold':'within1 percentage point of paper or higher; paper90.9 for ordinary CSA+LDA and94.4 for pure CSA',
        'limits':'previous probes were seen; repeated use of same400images; maxima and winning-config statistics are optimistic; not original author protocol',
        'stopping':'complete all200 predeclared splits regardless of crossing a target',
        'source_sha256':{name:file_hash(BASE/name) for name in sources}})
    dump(ROOT/'catalogue.json',entries)
    (ROOT/'scores').mkdir()
    (ROOT/'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(BASE/name,ROOT/'source_snapshot'/name)


def data_arrays():
    catalogue = {e['id']:e for e in read(ROOT/'catalogue.json')}
    arrays = []
    for branch in BRANCHES:
        entry = catalogue[branch['input']]
        assert file_hash(BASE/entry['gabor_path'])==entry['gabor_sha256']
        x = np.load(BASE/entry['gabor_path'],mmap_mode='r')
        if branch['normalization']=='power075':
            x = np.power(x,.75).astype(np.float32)
        else:
            x = r.normalization(x,branch['normalization'],np.arange(80))
        assert x.shape==(400,28,23,40) and np.isfinite(x).all()
        arrays.append(x)
    return arrays


def model_fit(x,train,probe,case,cache):
    return r.fitted(x,train,probe,{'algorithm':'csa',**case},cache)


def run(seeds):
    protocol = read(ROOT/'protocol.json')
    assert protocol['seeds']==SEEDS and protocol['branches']==BRANCHES
    assert protocol['cases']==json.loads(json.dumps(CASES)) and protocol['policies']==POLICIES
    assert protocol['source_sha256']['search_orl_best.py']==file_hash(Path(__file__)), 'Frozen search source changed'
    plan_hash = file_hash(ROOT/'protocol.json')
    data = data_arrays()
    start = time.perf_counter()
    for completed,seed in enumerate(seeds,1):
        path = ROOT/'scores'/f'{seed}.npz'
        if path.exists():
            saved = np.load(path)
            assert str(saved['protocol_sha256'])==plan_hash
            continue
        train,probe = r.split(r.LABELS,2,seed)
        assert len(train)==80 and len(probe)==320 and not np.intersect1d(train,probe).size
        counts = np.full((len(BRANCHES),len(CASES),6,39),-1,dtype=np.int16)
        sweeps = np.zeros((len(BRANCHES),len(CASES)),dtype=np.int8)
        sw_rank = np.zeros_like(sweeps)
        for bi,x in enumerate(data):
            cache = {}
            for ci,case in enumerate(CASES):
                model,g,q = model_fit(x,train,probe,case,cache)
                sweeps[bi,ci] = len(model.history)
                assert sweeps[bi,ci]<=3
                if case['family']=='CSA+LDA':
                    assert g.shape[1]<=len(train)-40
                    lda = LDA().fit(g,r.LABELS[train])
                    sw_rank[bi,ci] = lda.sw_rank
                    for pi,name in enumerate(POLICIES):
                        a,b = r.coordinates(g,q,lda,r.POLICIES[name])
                        predicted,_ = r.prefix_nn(a,b,r.LABELS[train],r.POLICIES[name]['metric'])
                        counts[bi,ci,pi,1:a.shape[1]] = np.sum(predicted[:,1:]==r.LABELS[probe,None],axis=0)
                else:
                    for pi,name in enumerate(POLICIES[:3]):
                        policy = r.POLICIES[name]
                        mean = g.mean(0) if policy['lda_center'] else 0.
                        nearest = np.argmin(cdist(q-mean,g-mean,policy['metric']),axis=1)
                        counts[bi,ci,pi,0] = np.sum(r.LABELS[train[nearest]]==r.LABELS[probe])
        np.savez_compressed(path,counts=counts,train=train,probe=probe,sweeps=sweeps,
                            sw_rank=sw_rank,protocol_sha256=np.array(plan_hash))
        if completed%5==0 or completed==len(seeds):
            print(f'batch {seeds[0]}: {completed}/{len(seeds)} splits complete, elapsed={time.perf_counter()-start:.1f}s',flush=True)


def row(seed,bi,ci,pi,di,count):
    case = CASES[ci]
    return {'seed':int(seed),'family':case['family'],'branch':int(bi),'case':int(ci),'policy_index':int(pi),
            'input':BRANCHES[bi]['input'],'normalization':BRANCHES[bi]['normalization'],
            'ranks':str(tuple(case['ranks'])),'CSA_dimension':int(np.prod(case['ranks'])),
            'center':case['center'],'init':case['init'],'Tmax':3,
            'LDA_dimension':di+1 if case['family']=='CSA+LDA' else '',
            'policy':POLICIES[pi],'alpha':0 if case['family']=='CSA+LDA' else '',
            'correct':int(count),'probe_count':320,'accuracy_percent':100*int(count)/320}


def summarize():
    assert all((ROOT/'scores'/f'{seed}.npz').exists() for seed in SEEDS)
    all_counts = np.stack([np.load(ROOT/'scores'/f'{seed}.npz')['counts'] for seed in SEEDS])
    near,best,statistics = [],[],[]
    for bi in range(len(BRANCHES)):
        for ci,case in enumerate(CASES):
            target = 90.9 if case['family']=='CSA+LDA' else 94.4
            for pi in range(6):
                for di in range(39):
                    counts = all_counts[:,bi,ci,pi,di]
                    if np.all(counts==-1):
                        continue
                    assert np.all(counts>=0) and np.all(counts<=320), 'A valid config must have all200 results'
                    values = counts*100/320
                    i = int(np.argmax(counts))
                    item = row(SEEDS[i],bi,ci,pi,di,counts[i])
                    stat = {**item,'n_splits':200,'mean_percent':float(values.mean()),
                            'std_pp':float(values.std(ddof=1)),'min_percent':float(values.min()),
                            'max_percent':float(values.max()),'paper_target':target,
                            'at_least_paper_count':int(np.sum(values>=target)),
                            'near_paper_count':int(np.sum(values>=target-1.))}
                    statistics.append(stat)
                    for j in np.flatnonzero(values>=target-1.):
                        near.append({**row(SEEDS[j],bi,ci,pi,di,counts[j]),'paper_target':target,
                                     'difference_pp':float(values[j]-target),'at_least_paper':bool(values[j]>=target)})
    write_csv(ROOT/'configuration_summary.csv',statistics)
    write_csv(ROOT/'near_or_exceed.csv',sorted(near,key=lambda v:(v['family'],-v['accuracy_percent'],v['seed'])))
    for family in ['CSA+LDA','CSA']:
        candidates = [item for item in statistics if item['family']==family]
        winner = max(candidates,key=lambda item:item['max_percent'])
        best.append(winner)
    dump(ROOT/'best_results.json',{'selection':'exploratory maximum selected on probes across configurations AND splits',
                                  'seeds':SEEDS,'scores_evaluated':int(np.sum(all_counts>=0)),
                                  'valid_fixed_configurations':len(statistics),'near_or_exceed_records':len(near),
                                  'best':best,'protocol_sha256':file_hash(ROOT/'protocol.json')})
    top = []
    for family in ['CSA+LDA','CSA']:
        selected_seeds = set()
        for item in sorted([n for n in near if n['family']==family],key=lambda v:-v['accuracy_percent']):
            if item['seed'] not in selected_seeds:
                selected_seeds.add(item['seed'])
                top.append(item)
            if len(selected_seeds)==10:
                break
    # Always replay both winners even if no run was within the near threshold.
    for winner in best:
        if not any(all(t[k]==winner[k] for k in ['seed','branch','case','policy_index','LDA_dimension']) for t in top):
            top.append(winner)
    dump(ROOT/'replay_selection.json',top)
    print(json.dumps(read(ROOT/'best_results.json'),indent=2),flush=True)


def replay(item,data,save=None):
    seed,bi,ci,pi = (int(item[k]) for k in ['seed','branch','case','policy_index'])
    case,policy = CASES[ci],r.POLICIES[POLICIES[pi]]
    train,probe = r.split(r.LABELS,2,seed)
    model,g,q = model_fit(data[bi],train,probe,case,{})
    # Independently encode the actual fitted tensor bases, using one contraction.
    for indices,actual in [(train,g),(probe,q)]:
        independent = np.einsum('nhwc,ha,wb,cd->nabd',data[bi][indices].astype(float)-model.mean,
                                *model.bases,optimize=True).reshape(len(indices),-1)
        np.testing.assert_allclose(actual,independent,rtol=1e-9,atol=1e-8)
    saved = {'train':train,'probe':probe,'core_gallery':g,'core_probe':q,'mean':model.mean,
             **{f'axis{k}':u for k,u in enumerate(model.bases)}}
    residual = None
    if case['family']=='CSA+LDA':
        lda = LDA().fit(g,r.LABELS[train])
        sw,sb = np.zeros_like(lda.sw),np.zeros_like(lda.sb)
        for label in range(1,41):
            samples = g[r.LABELS[train]==label]
            delta = samples-samples.mean(0)
            sw += delta.T@delta
            diff = samples.mean(0)-g.mean(0)
            sb += len(samples)*np.outer(diff,diff)
        np.testing.assert_allclose(sw,lda.sw,rtol=1e-10,atol=1e-7)
        np.testing.assert_allclose(sb,lda.sb,rtol=1e-10,atol=1e-7)
        residual = np.linalg.norm(sb@lda.basis-(sw@lda.basis)*lda.eigenvalues)/max(1.,np.linalg.norm(sb@lda.basis))
        assert residual<1e-7
        d = int(item['LDA_dimension'])
        g,q = r.coordinates(g,q,lda,policy)
        g,q = g[:,:d],q[:,:d]
        saved.update(sw=sw,sb=sb,lda_basis=lda.basis,lda_mean=lda.mean,eigenvalues=lda.eigenvalues)
    else:
        mean = g.mean(0) if policy['lda_center'] else 0.
        g,q = g-mean,q-mean
    distance = cdist(q,g,policy['metric'])
    near = np.argmin(distance,axis=1)
    if policy['metric']=='cosine':
        distance2 = 1-(q/np.linalg.norm(q,axis=1)[:,None])@(g/np.linalg.norm(g,axis=1)[:,None]).T
    else:
        distance2 = np.sum(q*q,axis=1)[:,None]+np.sum(g*g,axis=1)[None,:]-2*q@g.T
    np.testing.assert_array_equal(near,np.argmin(distance2,axis=1))
    pred = r.LABELS[train[near]]
    count = int(np.sum(pred==r.LABELS[probe]))
    assert count==int(item['correct'])
    saved.update(gallery=g,queries=q,predicted=pred,nearest=near)
    if save is not None:
        np.savez_compressed(save,**saved)
    return {'seed':seed,'branch':bi,'case':ci,'policy_index':pi,'LDA_dimension':item['LDA_dimension'],
            'correct':count,'accuracy_percent':100*count/320,'relative_Fisher_residual':residual}


def verify():
    expected = file_hash(ROOT/'protocol.json')
    total_scores = 0
    grids = {}
    for seed in SEEDS:
        saved = np.load(ROOT/'scores'/f'{seed}.npz')
        assert str(saved['protocol_sha256'])==expected
        train,probe = r.split(r.LABELS,2,seed)
        np.testing.assert_array_equal(train,saved['train'])
        np.testing.assert_array_equal(probe,saved['probe'])
        assert np.all((saved['sweeps']>=1)&(saved['sweeps']<=3))
        total_scores += int(np.sum(saved['counts']>=0))
        grids[seed] = saved['counts']
    assert total_scores==read(ROOT/'best_results.json')['scores_evaluated']
    folder = ROOT/'verified_best'
    folder.mkdir(exist_ok=True)
    checks = []
    arrays = data_arrays()
    for i,item in enumerate(read(ROOT/'replay_selection.json')):
        checks.append(replay(item,arrays,folder/f'{i:02d}.npz'))
    write_csv(folder/'replay_checks.csv',checks)
    # Reread every CSV summary and near-paper index against the complete grid.
    for filename in ['configuration_summary.csv','near_or_exceed.csv']:
        with (ROOT/filename).open(encoding='utf-8-sig',newline='') as f:
            for item in csv.DictReader(f):
                counts = grids[int(item['seed'])]
                di = int(item['LDA_dimension'])-1 if item['LDA_dimension'] else 0
                count = int(counts[int(item['branch']),int(item['case']),int(item['policy_index']),di])
                assert count==int(item['correct']) and 100*count/320==float(item['accuracy_percent'])
    dump(ROOT/'verification.json',{'passed':True,'planned_splits':200,'complete_score_cells':total_scores,
        'independently_refitted_best_runs':len(checks),'best_predictions_replayed':320*len(checks),
        'independent_encoding_scatter_Fisher_NN':True,'all_CSV_scores_match_grid':True,
        'protocol_sha256':expected,'source_sha256':file_hash(Path(__file__)),
        'limit':'all score cells recorded; independent model replay restricted to top10 distinct seeds per family and winners'})
    print(json.dumps(read(ROOT/'verification.json'),indent=2),flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert read(ROOT/'verification.json')['passed']
    fig,axes = plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
    for ax,winner in zip(axes,read(ROOT/'best_results.json')['best']):
        bi,ci,pi = (int(winner[k]) for k in ['branch','case','policy_index'])
        di = int(winner['LDA_dimension'])-1 if winner['LDA_dimension']!='' else 0
        scores = [np.load(ROOT/'scores'/f'{s}.npz')['counts'][bi,ci,pi,di]*100/320 for s in SEEDS]
        ax.hist(scores,bins=np.arange(65,100,.9375),color='#246d9b')
        ax.axvline(winner['paper_target'],color='#9d5284',ls='--',label='Paper reference')
        ax.axvline(np.mean(scores),color='#bc782b',label=f'Mean {np.mean(scores):.2f}%')
        ax.set_title(f'{winner["family"]}: best {max(scores):.2f}%\nFixed-config SD {np.std(scores,ddof=1):.2f}pp')
        ax.set_xlabel('Accuracy (%)');ax.set_ylabel('Number of splits');ax.legend(fontsize=9)
    fig.suptitle('Exploratory winners chosen on probes; all 200 splits retained')
    fig.savefig(ROOT/'best_distribution.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['prepare','run','summarize','verify','plot','replay'])
    parser.add_argument('--start',type=int,default=3000)
    parser.add_argument('--stop',type=int,default=3200)
    parser.add_argument('--family',choices=['CSA','CSA+LDA'],default='CSA+LDA')
    parser.add_argument('--output',default='results/orl_best_search')
    args = parser.parse_args()
    ROOT = (BASE/args.output).resolve()
    assert ROOT.is_relative_to(BASE)
    with threadpool_limits(limits=2):
        if args.stage=='prepare':
            prepare()
        elif args.stage=='run':
            seeds = list(range(args.start,args.stop))
            assert seeds and all(s in SEEDS for s in seeds)
            run(seeds)
        elif args.stage=='summarize':
            summarize()
        elif args.stage=='verify':
            verify()
        elif args.stage=='plot':
            plot()
        else:
            item = next(v for v in read(ROOT/'best_results.json')['best'] if v['family']==args.family)
            print(json.dumps(replay(item,data_arrays()),indent=2))
