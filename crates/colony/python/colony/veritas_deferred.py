"""VERITAS-2: explore, rank spatial hypotheses, then confirm them independently.

Ablation against VERITAS's immediate-claim protocol on NEW heldout seeds.
No trained localization model: rank observed alpha, not hidden source coordinates.
"""
from __future__ import annotations
import argparse,json,math,statistics
from dataclasses import dataclass,asdict
from pathlib import Path
from .signalroot import World,N,dist
from .signalroot_dfa import score
from .trustroot import METHODS,START,policy_position,bootstrap_difference
from .veritas import FAMILIES,episode as immediate_episode,get_field,calibrate,project_affordable

@dataclass(frozen=True)
class DeferredOutcome:
    seed:int
    family:str
    method:str
    claimed:bool
    true_claim:bool
    false_claim:bool
    claim_point:tuple[int,int]|None
    claim_distance:float|None
    acquired:int
    distinct_places:int
    repetitions:int
    traveled:float


def deferred_episode(seed,family,method,primary,replicate,threshold,*,budget=24,travel_cap=85,confirm_reserve=6):
    if family not in FAMILIES or method not in METHODS:raise ValueError('invalid experiment')
    if budget<5+confirm_reserve or not(0<confirm_reserve<budget):raise ValueError('invalid confirmation budget')
    if len(primary)!=N*N or set(primary)!=set(replicate):raise ValueError('incomplete paired fields')
    observed={};valid=[];current=START[0];travel=0.
    for i in range(budget-confirm_reserve):
        if i<len(START):p=START[i]
        else:p,_=policy_position(method,observed,current,i,valid)
        if p is None:break
        if p in observed:raise AssertionError('revisited probe')
        distance=dist(current,p)
        if travel+distance>travel_cap+1e-9:
            p=project_affordable(p,observed,current,travel_cap-travel)
            if p is None:break
            distance=dist(current,p)
        travel+=distance;current=p
        alpha=primary[p]
        observed[p]=None if alpha is None else score(alpha[0])
        valid.append(alpha is not None)
    # Rank only measured evidence. No access to hidden source or independent
    # confirmation values until the candidate is selected for re-measurement.
    hypotheses=[(p,primary[p][0]) for p in observed if primary[p] is not None and primary[p][0]>threshold]
    hypotheses.sort(key=lambda t:(-t[1],t[0][1],t[0][0]))
    confirmations=[]
    for p,first in hypotheses[:confirm_reserve]:
        repeated=replicate[p]
        if repeated is not None and repeated[0]>threshold:
            confirmations.append((p,min(first,repeated[0])))
    # Choose strongest independently confirmed location, not the first claim.
    p=max(confirmations,key=lambda x:(x[1],-x[0][1],-x[0][0]))[0] if confirmations else None
    source=World(seed,'compact').source  # evaluation-only oracle
    actual_dist=None if p is None else dist(p,source)
    is_true=p is not None and family!='shuffled' and actual_dist<=2.0
    return DeferredOutcome(seed,family,method,p is not None,is_true,p is not None and not is_true,
                           p,actual_dist,len(observed)+min(len(hypotheses),confirm_reserve),
                           len(observed),min(len(hypotheses),confirm_reserve),round(travel,6))


def paired_report(*,seeds=8,seed_start=3000,budget=24,travel_cap=85,command=None):
    if command is None:raise ValueError('real Rust DFA bridge required')
    if seeds<1 or seed_start<0:raise ValueError('invalid seed range')
    calibration=calibrate(command)
    if any(seed_start<=s<seed_start+seeds for s in calibration['seeds']):raise ValueError('overlap')
    rows=[]
    for family in FAMILIES:
        for seed in range(seed_start,seed_start+seeds):
            first=get_field(seed,family,0,command)
            second=get_field(seed,family,1,command)
            for method in METHODS:
                a=immediate_episode(seed,family,method,first,second,calibration['threshold_alpha'],budget=budget,travel_cap=travel_cap)
                b=deferred_episode(seed,family,method,first,second,calibration['threshold_alpha'],budget=budget,travel_cap=travel_cap)
                rows.append({'family':family,'seed':seed,'method':method,'immediate':asdict(a),'deferred':asdict(b)})
    summary={}
    for family in FAMILIES:
        summary[family]={}
        for method in METHODS:
            group=[r for r in rows if r['family']==family and r['method']==method]
            immediate=[r['immediate'] for r in group]; deferred=[r['deferred'] for r in group]
            summary[family][method]={
             'immediate_true_rate':statistics.mean(r['true_claim'] for r in immediate),
             'immediate_false_rate':statistics.mean(r['false_claim'] for r in immediate),
             'deferred_true_rate':statistics.mean(r['true_claim'] for r in deferred),
             'deferred_false_rate':statistics.mean(r['false_claim'] for r in deferred),
             'deferred_mean_windows':statistics.mean(r['acquired'] for r in deferred),
             'deferred_mean_travel':statistics.mean(r['traveled'] for r in deferred),
             'paired_true_gain':bootstrap_difference([int(r['true_claim']) for r in deferred],[int(r['true_claim']) for r in immediate],seed=333),
            }
    return {'format':'colony-veritas-deferred/0.1','calibration':calibration,'seed_start':seed_start,'seeds':seeds,
      'budget':budget,'travel_cap':travel_cap,'summary':summary,'episodes':rows,
      'caveats':['No learned source-inference model; ranks sampled alpha only.',
       'False TARGET attribution may still happen even when structural change is real.',
       'The same synthetic generator produces discovery and confirmation; independent noise is not independent model specification.',
       'This is not Struktura Guard or real sensor-health validation.']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--seed-start',type=int,default=3000)
    p.add_argument('--budget',type=int,default=24)
    p.add_argument('--travel-cap',type=float,default=85.)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    obj=paired_report(seeds=a.seeds,seed_start=a.seed_start,budget=a.budget,travel_cap=a.travel_cap,
                command=['cargo','run','--quiet','--example','colony_dfa_bridge'])
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({'calibration':obj['calibration'],'summary':obj['summary']},indent=2))
if __name__=='__main__':main()
