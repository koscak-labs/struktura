"""SIGNALROOT: reproducible active sensing experiments, not a robot controller.

Observations are revealed to planners one at a time. Hidden source locations
exist only inside the environment/evaluator. The `cue` is an explicitly labeled
synthetic anomaly proxy, NOT a Struktura Guard alarm.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from random import Random
import math
import json
import statistics
import argparse

N=25
Point=tuple[int,int]
MODES=("raster","random","coverage","signal_sca","signal_greedy")

@dataclass(frozen=True)
class Trial:
    method: str
    seed: int
    family: str
    hit: bool
    first_hit: int|None
    best_distance: float
    steps: int
    length: float
    measured_max: float
    cue_max: float

class World:
    """Hidden truth is intentionally absent from policy inputs."""
    def __init__(self,seed:int,family:str):
        if family not in ("compact","diffuse","decoy","dropout"): raise ValueError(family)
        self.seed=seed;self.family=family
        r=Random(seed*7919+23)
        self.source=(r.randint(4,20),r.randint(4,20))
        self.decoy=(r.randint(3,21),r.randint(3,21))
        self.sigma=2.0 if family in ("compact","decoy","dropout") else 5.0
        self.r=Random(seed*2371+91)
    def sample(self,p:Point,iteration:int):
        x,y=p
        d2=(x-self.source[0])**2+(y-self.source[1])**2
        strength=math.exp(-d2/(2*self.sigma**2))
        if self.family=="decoy":
            dd=(x-self.decoy[0])**2+(y-self.decoy[1])**2
            strength+=0.60*math.exp(-dd/(2*3**2))
        noise=self.r.gauss(0,0.045)
        # Missing measurements never become a fabricated zero reading.
        healthy=not(self.family=="dropout" and iteration>=10 and (x+2*y)%5 in (0,1))
        return None if not healthy else max(0., strength+noise)
    def distance(self,p:Point):
        return math.dist(p,self.source)

def walkable(p):return 0<=p[0]<N and 0<=p[1]<N
def dist(a,b):return math.dist(a,b)
def startup():
    # Equal initial observations for every policy. No ground truth, only geometry.
    return [(0,0),(24,0),(0,24),(24,24),(12,12)]
def seed_for(seed,method):
    return seed*131+{"raster":1,"random":2,"coverage":3,"signal_sca":4,"signal_greedy":5}[method]

def proposal(method,positions:dict[Point,float|None],rng:Random,current:Point):
    used=set(positions)
    free=[(x,y) for y in range(N) for x in range(N) if (x,y) not in used]
    if not free:return None
    if method=="random": return rng.choice(free)
    if method=="raster":return min(free,key=lambda p:(p[1],p[0]))
    if method=="coverage":
        return max(free,key=lambda p:(min(dist(p,q) for q in used),-dist(p,current),-p[1],-p[0]))
    valid={p:v for p,v in positions.items() if v is not None}
    if not valid: return proposal("coverage",positions,rng,current)
    if method=="signal_greedy":
        # Exploit best observed signal, but never a hidden source position.
        center=max(valid,key=lambda p:(valid[p],-p[1],-p[0]))
        return min(free,key=lambda p:(dist(center,p),dist(current,p),p[1],p[0]))
    if method!="signal_sca":raise ValueError(method)
    # Multiple growing tips: nominate high-signal observations; advance toward
    # unexplored attraction points, with an exploration floor to prevent collapse.
    ranked=sorted(valid,key=lambda p:(-valid[p],p[1],p[0]))[:7]
    frontier=[]
    for tip in ranked:
        directions=[(x,y) for x,y in free if 0<dist(tip,(x,y))<=5.0]
        if not directions:continue
        # Space-colonization-style summed unit directions, weighted by measured cue.
        attractors=sorted(directions,key=lambda p:(-min(dist(p,q) for q in used),p[1],p[0]))[:14]
        vx=sum((p[0]-tip[0])/dist(p,tip) for p in attractors)
        vy=sum((p[1]-tip[1])/dist(p,tip) for p in attractors)
        magnitude=math.hypot(vx,vy)
        if magnitude<1e-9:continue
        target=(tip[0]+2*vx/magnitude,tip[1]+2*vy/magnitude)
        candidates=sorted(directions,key=lambda p:(dist(p,target),p[1],p[0]))
        p=candidates[0]
        novelty=min(dist(p,q) for q in used)
        score=1.8*max(0,valid[tip])+0.24*novelty-0.018*dist(current,p)
        frontier.append((score,p))
    if not frontier:return proposal("coverage",positions,rng,current)
    # Seeded tie breakers are deterministic, no true field values are inspected.
    return sorted(frontier,key=lambda t:(-t[0],t[1][1],t[1][0]))[0][1]

def run(seed:int,family:str,method:str,budget:int=45)->Trial:
    if method not in MODES:raise ValueError(method)
    if budget<len(startup()):raise ValueError("budget must include startup observations")
    world=World(seed,family);rng=Random(seed_for(seed,method))
    observations={}
    current=startup()[0];length=0.;best=999.;first=None
    for i in range(budget):
        p=startup()[i] if i<len(startup()) else proposal(method,observations,rng,current)
        if p is None:break
        if i: length+=dist(current,p)
        current=p
        observed=world.sample(p,i)
        observations[p]=observed
        d=world.distance(p)
        if d<best:best=d
        if observed is not None and d<=2.0 and first is None:first=i+1
    valid=[v for v in observations.values() if v is not None]
    return Trial(method,seed,family,first is not None,first,best,len(observations),length,max(valid,default=0),max(valid,default=0))

def benchmark(seeds=40,budget=45):
    if not(1<=seeds<=1000 and 6<=budget<=N*N):raise ValueError("invalid benchmark")
    data=[asdict(run(seed,family,method,budget))
          for family in ("compact","diffuse","decoy","dropout")
          for seed in range(seeds) for method in MODES]
    results={}
    for family in ("compact","diffuse","decoy","dropout"):
        results[family]={}
        for method in MODES:
            rows=[r for r in data if r["family"]==family and r["method"]==method]
            results[family][method]={
                "hit_rate":sum(r["hit"] for r in rows)/len(rows),
                "mean_min_distance":statistics.mean(r["best_distance"] for r in rows),
                "mean_travel":statistics.mean(r["length"] for r in rows),
                "mean_first_hit_successes":statistics.mean([r["first_hit"] for r in rows if r["hit"]]) if any(r["hit"] for r in rows) else None,
            }
    return {"format":"colony-signalroot/0.1","source":"synthetic_spatial_cue_proxy_not_struktura","seeds":seeds,"budget":budget,"results":results,"episodes":data,
            "caveats":["Sensor cue is simulated, not a validated Struktura detector.",
                       "True source used by evaluation oracle only.",
                       "All policies use equal observation budgets, not equal travel or compute.",
                       "No physical robot, camera, dynamics or telemetry integration."]}
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,default=40)
    parser.add_argument("--budget",type=int,default=45)
    parser.add_argument("--out")
    a=parser.parse_args()
    result=benchmark(a.seeds,a.budget)
    if a.out:
        from pathlib import Path
        p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["results"],indent=2))
