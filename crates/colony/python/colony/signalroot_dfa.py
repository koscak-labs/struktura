"""Two-stage offline experiment: actual Rust Struktura DFA outputs feed COLONY.

Phase A generates windows WITHOUT giving the planner access to hidden labels.
Phase B parses the bridge's scores and reveals each location only on visitation.
DFA alpha measures temporal structure; it is NOT a calibrated Guard event.
"""
from __future__ import annotations
import argparse,csv,io,json,math,random,statistics,subprocess
from pathlib import Path
from .signalroot import N,MODES,startup,proposal,dist,seed_for,World

N_WINDOW=96
FAMILIES=('correlation','amplitude_decoy','dropout')

def trace(seed:int,family:str,location:tuple[int,int])->list[float]:
    if family not in FAMILIES:raise ValueError(family)
    x,y=location
    world=World(seed,'compact')
    # Signal strength tied to hidden source, but only signal generator knows it.
    influence=math.exp(-dist(location,world.source)**2/(2*3.5**2))
    decoy=math.exp(-dist(location,world.decoy)**2/(2*3.5**2))
    rng=random.Random(seed*100003+x*703+y*911+17)
    state=0.0;values=[]
    rho=0.05+0.90*influence
    for t in range(N_WINDOW):
        innovation=rng.gauss(0,1)
        state=rho*state+math.sqrt(max(0.001,1-rho*rho))*innovation
        amp=1+3.0*decoy if family=='amplitude_decoy' else 1
        values.append(state*amp)
    if family=='dropout' and (x+2*y+seed)%7 in (0,1):
        values[20]=float('nan')
    return values

def rows_for(seed:int,family:str):
    if family not in FAMILIES:raise ValueError(family)
    for y in range(N):
        for x in range(N):
            vals=trace(seed,family,(x,y))
            yield f"{x},{y},"+','.join(format(z,'.11g') for z in vals)

def parse_scores(text:str):
    data={}
    for row in csv.DictReader(io.StringIO(text)):
        loc=(int(row['x']),int(row['y']))
        if loc in data:raise ValueError('duplicate position')
        if not (0<=loc[0]<N and 0<=loc[1]<N):raise ValueError('invalid position')
        valid=row['confidence']=='1' and row['alpha']!='' and row['r_squared']!=''
        if valid:
            alpha=float(row['alpha']);r2=float(row['r_squared'])
            if not(math.isfinite(alpha) and 0.5<=r2<=1.00001):raise ValueError('invalid DFA')
            data[loc]=(alpha,r2)
        else:data[loc]=None
    if len(data)!=N*N:raise ValueError('incomplete bridge output')
    return data

def score(alpha:float,reference:float=0.5)->float:
    # Alpha departure is a hypothesis, not a probability or calibrated detection score.
    return min(3.0,max(0.0,alpha-reference))

def run_bridge(input_csv:str,command:list[str]):
    p=subprocess.run(command,input=input_csv,text=True,capture_output=True,check=False,timeout=180)
    if p.returncode!=0:raise RuntimeError(f'Rust bridge failed: {p.stderr[-1000:]}')
    return parse_scores(p.stdout)

def episode(seed,family,method,data,budget=35):
    if method not in MODES:raise ValueError(method)
    rng=random.Random(seed_for(seed,method));observed={};current=startup()[0]
    world=World(seed,'compact');first=None;travel=0.;valid_reads=0
    for i in range(budget):
        p=startup()[i] if i<len(startup()) else proposal(method,observed,rng,current)
        if p is None:break
        if i:travel+=dist(current,p)
        current=p
        result=data[p]
        observed[p]=None if result is None else score(result[0])
        if result is not None:valid_reads+=1
        if first is None and result is not None and dist(p,world.source)<=2:first=i+1
    return {'seed':seed,'family':family,'method':method,'hit':first is not None,
            'first_hit':first,'travel':travel,'readings':len(observed),'valid_readings':valid_reads}

def experiment(seeds=4,budget=35,cmd=None):
    if cmd is None:raise ValueError('real Struktura bridge command required')
    results=[]
    for family in FAMILIES:
        for seed in range(seeds):
            input_csv='\n'.join(rows_for(seed,family))+'\n'
            data=run_bridge(input_csv,cmd)
            for method in ('coverage','signal_sca','signal_greedy'):
                results.append(episode(seed,family,method,data,budget))
    report={}
    for family in FAMILIES:
        report[family]={}
        for method in ('coverage','signal_sca','signal_greedy'):
            rows=[r for r in results if r['family']==family and r['method']==method]
            report[family][method]={'hit_rate':sum(r['hit'] for r in rows)/len(rows),
              'mean_travel':statistics.mean(r['travel'] for r in rows),'mean_valid_readings':statistics.mean(r['valid_readings'] for r in rows)}
    return {'format':'colony-struktura-dfa/0.1','actual_dfa':True,'detector':'struktura::dfa_short NOT Guard',
            'seeds':seeds,'budget':budget,'summary':report,'episodes':results,
            'limitations':['Entire spatial field scored offline by Rust; planner sees only visited cells.',
            'Not a calibrated anomaly or fault detector.',
            'Controlled synthetic temporal-correlation source, not real robot telemetry.',
            'Equal readings do not imply equal travel, wall-clock CPU or power.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=4)
    parser.add_argument('--budget',type=int,default=35);parser.add_argument('--out',type=Path)
    a=parser.parse_args()
    output=experiment(a.seeds,a.budget,cmd=['cargo','run','--quiet','--example','colony_dfa_bridge'])
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output['summary'],indent=2))
