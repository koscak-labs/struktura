"""TRUSTROOT: adversarial active sensing with *actual* Struktura DFA windows.

A missing window is an abstention, not a zero score. `trustroot` is deliberately
an explicit deterministic health-aware heuristic: do not claim emergent recovery.
The only use of hidden targets is within the scenario generator and evaluator.
No physics, Guard, or hardware actuation.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .signalroot import N, World, dist, proposal
from .signalroot_dfa import rows_for, run_bridge, score, trace, N_WINDOW

METHODS = ('coverage', 'costed_coverage', 'signal_sca', 'signal_greedy', 'fixed_mix', 'trustroot')
FAMILIES = ('correlation', 'dropout', 'burst_dropout', 'correlated_decoy', 'shuffled')
START = ((12, 12), (12, 13), (13, 12), (11, 12), (12, 11))


def field_trace(seed: int, family: str, loc: tuple[int,int]) -> list[float]:
    if family == 'correlation':
        return trace(seed, 'correlation', loc)
    if family == 'dropout':
        return trace(seed, 'dropout', loc)
    if family == 'burst_dropout':
        return trace(seed, 'correlation', loc)
    if family == 'shuffled':
        series = trace(seed, 'correlation', loc)
        x, y = loc
        random.Random(901_003*seed + 7919*x + 59*y + 17).shuffle(series)
        return series
    if family == 'correlated_decoy':
        # A strong but irrelevant correlation source may attract both policies.
        world = World(seed, 'compact')
        source = math.exp(-dist(loc, world.source)**2/(2*3.5**2))
        decoy = math.exp(-dist(loc, world.decoy)**2/(2*2.5**2))
        rho = min(.97, .05 + .65*source + .90*decoy)
        x, y = loc
        rand = random.Random(seed*100_003 + x*703 + y*911 + 17)
        state, values = 0., []
        for _ in range(N_WINDOW):
            state = rho*state + math.sqrt(1-rho*rho)*rand.gauss(0, 1)
            values.append(state)
        return values
    raise ValueError(f'unknown world family {family}')


def spatial_rows(seed: int, family: str):
    for y in range(N):
        for x in range(N):
            vals = field_trace(seed, family, (x,y))
            yield f'{x},{y},' + ','.join(format(v, '.11g') for v in vals)


def costed_position(observed: dict, current: tuple[int,int], invalid_avoidance=False):
    """A stronger coverage baseline, optimizing novelty AND travel from current pose."""
    used = set(observed)
    if not used: raise ValueError('requires prior observations')
    missing = [p for p, v in observed.items() if v is None]
    candidates = ((x,y) for y in range(N) for x in range(N) if (x,y) not in used)
    def merit(p):
        novelty = min(dist(p,q) for q in used)
        cost = dist(p,current)
        risk = (max(0,2.5-min(dist(p,q) for q in missing))*0.6) if invalid_avoidance and missing else 0.
        return (novelty - .40*cost - risk, -cost,-p[1],-p[0])
    return max(candidates,key=merit,default=None)


def trust_position(observed: dict, current: tuple[int,int], *, last_valid: list[bool]):
    """Conservative failover: retain SCA until sustained sensor loss.

    Naive switching on *any* invalid reading performed badly on training pilots.
    This is a fixed explicit heuristic, NOT recovered autonomy or learning.
    """
    tail=last_valid[-8:]
    if len(tail)>=6 and sum(not x for x in tail)>=5:
        return costed_position(observed,current,invalid_avoidance=True), 'repair'
    return proposal('signal_sca', observed, random.Random(0), current), 'attract'


def policy_position(method, observed, current, iteration, health):
    if method == 'costed_coverage':
        return costed_position(observed,current), method
    if method == 'trustroot':
        return trust_position(observed, current, last_valid=health)
    if method == 'fixed_mix':
        other = 'coverage' if iteration % 3 == 0 else 'signal_sca'
        return proposal(other,observed,random.Random(iteration),current), other
    if method in METHODS:
        return proposal(method,observed,random.Random(iteration),current), method
    raise ValueError(f'unknown method {method}')

@dataclass(frozen=True)
class Episode:
    seed: int
    family: str
    method: str
    criterion: str
    target_found: bool
    first_hit: int | None
    first_hit_travel: float | None
    valid_readings: int
    readings: int
    travel: float
    repair_steps: int
    invalid_readings: int


def episode(seed: int, family: str, method: str, field: dict, *,
            budget: int, travel_cap: float | None = None) -> Episode:
    if family not in FAMILIES or method not in METHODS:
        raise ValueError('bad family/method')
    if not isinstance(budget,int) or not len(START)<=budget<=N*N:
        raise ValueError('bad budget')
    if travel_cap is not None and (not math.isfinite(travel_cap) or travel_cap<=0):
        raise ValueError('bad travel budget')
    world = World(seed, 'compact')  # Hidden oracle seen ONLY by evaluator below.
    observed = {}
    health = []
    current = START[0]
    travel = 0.
    first_hit = None
    first_hit_travel = None
    repairs = 0
    for i in range(budget):
        if i<len(START):
            point = START[i]
            mode = 'initial'
        else:
            point, mode = policy_position(method, observed, current, i, health)
        if point is None:
            break
        assert point not in observed and 0<=point[0]<N and 0<=point[1]<N
        delta = dist(current, point) if i else 0.
        if travel_cap is not None and travel+delta>travel_cap+1e-9:
            # Common constraint projection: every planner may move to a reachable
            # intermediate observation, rather than unfairly stopping a long-hop
            # proposal. This projection is part of the experimental protocol.
            remaining = travel_cap - travel
            affordable = [(x,y) for y in range(N) for x in range(N)
                          if (x,y) not in observed and dist(current,(x,y))<=remaining+1e-9]
            if not affordable:
                break
            point = min(affordable, key=lambda q:(dist(q,point), dist(q,current),q[1],q[0]))
            delta = dist(current,point)
        travel += delta
        current = point
        reading = field[point]  # oracle reveals ONLY the queried value.
        if family=='burst_dropout' and 9<=i<19:
            reading=None  # temporary sensor outage independent of where policy probes
        observed[point] = None if reading is None else score(reading[0])
        valid = reading is not None
        health.append(valid)
        if mode=='repair':
            repairs+=1
        # Scorer knows truth, but planner was never given it.
        if first_hit is None and valid and dist(point, world.source)<=2:
            first_hit = i+1
            first_hit_travel = travel
    return Episode(seed, family, method, 'equal_travel' if travel_cap is not None else 'equal_observations',
                   first_hit is not None, first_hit, first_hit_travel,
                   sum(health),len(observed),round(travel,6),repairs,len(health)-sum(health))


def bootstrap_difference(a, b, seed=177, reps=1500):
    if len(a)!=len(b) or not a: raise ValueError('need paired arrays')
    delta=[x-y for x,y in zip(a,b)]
    rng=random.Random(seed)
    means=sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(reps))
    return {'paired_delta':statistics.mean(delta), 'percentile95': [means[int(.025*reps)],means[int(.975*reps)-1]],'paired_n':len(delta)}


def summarize(episodes):
    result={}
    for family in FAMILIES:
        result[family]={}
        for criterion in ('equal_observations','equal_travel'):
            group=[r for r in episodes if r.family==family and r.criterion==criterion]
            if not group: continue
            result[family][criterion]={}
            keyed={(r.seed,r.method):r for r in group}
            for method in METHODS:
                vals=[r for r in group if r.method==method]
                if not vals:continue
                vals.sort(key=lambda r:r.seed)
                baseline=[keyed[(r.seed,'coverage')] for r in vals]
                result[family][criterion][method]={
                    'hit_rate':statistics.mean(r.target_found for r in vals),
                    'mean_travel':statistics.mean(r.travel for r in vals),
                    'mean_readings':statistics.mean(r.readings for r in vals),
                    'mean_valid_readings':statistics.mean(r.valid_readings for r in vals),
                    'mean_repair_steps':statistics.mean(r.repair_steps for r in vals),
                    'delta_vs_coverage':bootstrap_difference([int(r.target_found) for r in vals],
                                                             [int(x.target_found) for x in baseline]),
                }
    return result


def experiment(*, seeds: int, budget: int, travel_cap: float, seed_start: int = 1000,
               bridge_command: list[str] | None = None):
    if bridge_command is None:raise ValueError('real Rust DFA bridge command required')
    if seeds<1 or seeds>1000 or not isinstance(seed_start,int) or seed_start<0:raise ValueError('invalid seeds')
    started=time.perf_counter()
    all_eps=[]
    for family in FAMILIES:
        for seed in range(seed_start,seed_start+seeds):
            rows='\n'.join(spatial_rows(seed,family))+'\n'
            field=run_bridge(rows, bridge_command)
            for method in METHODS:
                for cap in (None,travel_cap):
                    all_eps.append(episode(seed,family,method,field,budget=budget,travel_cap=cap))
    return {'format':'colony-trustroot/0.1','actual_detector':'struktura::dfa_short (not Guard)',
            'worlds':len(FAMILIES)*seeds,'paired_seeds_per_family':seeds,'seed_start':seed_start,
            'budget':budget,'travel_cap':travel_cap,'elapsed_seconds':round(time.perf_counter()-started,3),
            'summary':summarize(all_eps),'episodes':[asdict(r) for r in all_eps],
            'limitations':['All location windows are scored offline by the real Struktura DFA bridge.',
             'This is NOT calibrated Struktura Guard and cannot decide sensor trust.',
             'Missing windows, not R² values, drive the health-aware fallback.',
             'Travel is direct Euclidean movement on an unobstructed synthetic grid.',
             'Results are synthetic and do not establish spaceflight safety or algorithmic novelty.']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=20)
    p.add_argument('--seed-start',type=int,default=1000)
    p.add_argument('--budget',type=int,default=24)
    p.add_argument('--travel-cap',type=float,default=85.)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    result=experiment(seeds=a.seeds,budget=a.budget,travel_cap=a.travel_cap,seed_start=a.seed_start,
                      bridge_command=['cargo','run','--quiet','--example','colony_dfa_bridge'])
    if a.out:
        a.out.parent.mkdir(exist_ok=True,parents=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
