"""VERITAS: independent temporal confirmation of anomaly-guided exploration.

Calibrate a Struktura DFA alpha cutoff on independent shuffled/no-source
windows, then require TWO independent real DFA measurements before claiming a
spatially localized structural discovery. These are synthetic traces; no Guard
calibration or real-world fault interpretation is supplied.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .signalroot import World, N, dist
from .signalroot_dfa import run_bridge, score, N_WINDOW
from .trustroot import METHODS, START, policy_position

FAMILIES = ('correlation', 'shuffled', 'dropout')


def independent_trace(seed: int, family: str, p: tuple[int,int], replica: int) -> list[float]:
    if family not in FAMILIES or replica not in (0,1):
        raise ValueError('unsupported family/replica')
    x, y = p
    world=World(seed,'compact')
    influence=math.exp(-dist(p,world.source)**2/(2*3.5**2))
    rho=0.05+0.90*influence
    rng=random.Random(seed*100_003+x*703+y*911+replica*1_000_000_007+17)
    state=0.
    values=[]
    for _ in range(N_WINDOW):
        state=rho*state+math.sqrt(max(.001,1-rho*rho))*rng.gauss(0,1)
        values.append(state)
    if family=='shuffled':
        random.Random(seed*55_555+x*2909+y*71+replica*1777).shuffle(values)
    if family=='dropout' and ((x+2*y+seed+replica)%7 in (0,1)):
        values[20]=float('nan')
    return values


def bridge_rows(seed, family, replica):
    for y in range(N):
        for x in range(N):
            series=independent_trace(seed,family,(x,y),replica)
            yield f'{x},{y},'+','.join(format(v,'.11g') for v in series)


def get_field(seed,family,replica,command):
    return run_bridge('\n'.join(bridge_rows(seed,family,replica))+'\n',command)


def calibrate(command, seeds=(6000,6001), percentile=0.95):
    if not(0.9 <= percentile <= 0.99) or not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('bad calibration protocol')
    null=[]
    for seed in seeds:
        for reading in get_field(seed,'shuffled',0,command).values():
            if reading is not None:
                null.append(reading[0])
    if len(null)<100:
        raise ValueError('too few null-calibration windows')
    null.sort()
    k=max(0,min(len(null)-1,math.ceil(percentile*len(null))-1))
    return {'threshold_alpha':null[k], 'valid_calibration_windows':len(null),
            'seeds':list(seeds),'percentile':percentile,
            'meaning':'one-window null quantile, not calibrated Guard probability'}


@dataclass(frozen=True)
class Outcome:
    method: str
    seed: int
    family: str
    claim: bool
    true_claim: bool
    false_claim: bool
    first_claim_at: int | None
    readings: int
    primary_locations: int
    confirm_reads: int
    valid_readings: int
    travel: float


def project_affordable(point, observed, current, remaining):
    affordable=((x,y) for y in range(N) for x in range(N)
                if (x,y) not in observed and dist(current,(x,y))<=remaining+1e-9)
    return min(affordable,key=lambda q:(dist(q,point),dist(q,current),q[1],q[0]),default=None)


def episode(seed,family,method,primary,confirm,threshold,*,budget=24,travel_cap=85):
    if family not in FAMILIES or method not in METHODS:raise ValueError('invalid policy/world')
    if not (math.isfinite(threshold) and budget>=len(START) and travel_cap>0):raise ValueError('bad protocol')
    if set(primary)!=set(confirm) or len(primary)!=N*N:raise ValueError('incomplete paired fields')
    observations={}
    health=[]
    current=START[0]
    total=0.;used=0;replicas=0;valid=0;claim=None;first=None
    oracle=World(seed,'compact')
    for i in range(budget):
        if used>=budget:break
        if i<len(START):point=START[i]
        else:point,_=policy_position(method,observations,current,i,health)
        if point is None:break
        if point in observations:raise AssertionError('duplicate primary location')
        d=dist(current,point)
        if total+d>travel_cap+1e-9:
            point=project_affordable(point,observations,current,travel_cap-total)
            if point is None:break
            d=dist(current,point)
        current=point;total+=d
        alpha=primary[point]
        used+=1
        observations[point]=None if alpha is None else score(alpha[0])
        ok=alpha is not None; health.append(ok);valid+=int(ok)
        if ok and alpha[0]>threshold and used<budget:
            verification=confirm[point]
            used+=1;replicas+=1
            valid+=int(verification is not None)
            if verification is not None and verification[0]>threshold:
                claim=point
                first=used
                break
    is_true=claim is not None and family!='shuffled' and dist(claim,oracle.source)<=2
    return Outcome(method,seed,family,claim is not None,is_true,claim is not None and not is_true,
                   first,used,len(observations),replicas,valid,round(total,6))


def run(*,seeds=8,seed_start=2000,budget=24,travel_cap=85,command=None):
    if command is None:raise ValueError('real Struktura DFA bridge mandatory')
    if seeds<1 or budget<5 or not(math.isfinite(travel_cap) and travel_cap>0):raise ValueError('invalid budget')
    calibration=calibrate(command)
    if any(seed_start<=s<seed_start+seeds for s in calibration['seeds']):
        raise ValueError('calibration and holdout overlap')
    outcomes=[]
    for family in FAMILIES:
        for seed in range(seed_start,seed_start+seeds):
            field=get_field(seed,family,0,command)
            repeats=get_field(seed,family,1,command)
            for method in METHODS:
                outcomes.append(episode(seed,family,method,field,repeats,
                        calibration['threshold_alpha'],budget=budget,travel_cap=travel_cap))
    summary={}
    for family in FAMILIES:
        summary[family]={}
        for method in METHODS:
            group=[o for o in outcomes if o.family==family and o.method==method]
            summary[family][method]={
                'verified_target_rate':statistics.mean(o.true_claim for o in group),
                'false_claim_rate':statistics.mean(o.false_claim for o in group),
                'any_claim_rate':statistics.mean(o.claim for o in group),
                'mean_windows':statistics.mean(o.readings for o in group),
                'mean_confirmations':statistics.mean(o.confirm_reads for o in group),
                'mean_travel':statistics.mean(o.travel for o in group),
            }
    return {'format':'colony-veritas/0.1', 'source':'real_struktura_dfa_on_synthetic_traces',
      'calibration':calibration, 'heldout_seed_start':seed_start,'heldout_seeds_per_family':seeds,
      'window_budget':budget,'travel_cap':travel_cap,'summary':summary,'episodes':[asdict(o) for o in outcomes],
      'caveats':['Independent replicate per location is simulated, not real sensing.',
       'Anomaly threshold is an independent empirical null-quantile, not Guard severity.',
       'Confirmed event away from hidden source is a false TARGET claim, but could be a genuine nuisance signal.',
       'Shuffled data contains no designed structural source; any confirmation is a false discovery.',
       'Observation-location travel is Euclidean, without real navigation, power or risk.']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--seed-start',type=int,default=2000)
    p.add_argument('--budget',type=int,default=24)
    p.add_argument('--travel-cap',type=float,default=85.)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    obj=run(seeds=a.seeds,seed_start=a.seed_start,budget=a.budget,travel_cap=a.travel_cap,
            command=['cargo','run','--quiet','--example','colony_dfa_bridge'])
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({'calibration':obj['calibration'],'summary':obj['summary']},indent=2))
if __name__=='__main__':main()
