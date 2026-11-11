"""Rebuild bounded full-coordinate native witnesses, no training checkpoint changes."""
import argparse
import json
from pathlib import Path

import numpy as np

from scratchgpt.checkpoint import named_parameters
from scratchgpt.model import GPT
from scratchgpt.optim import Adam
from scratchgpt.remat_audit import audit, SCHEMA
from scratchgpt.tensor import cross_entropy

ROOT=Path(__file__).resolve().parents[1]


def build():
    reports=[]
    for seed in range(6):
        layers=1+seed%3;d=2*(1+seed%3);heads=1+seed%2
        if d%heads:heads=1
        c=dict(vocab_size=5+seed%3,block_size=4,n_layer=layers,n_head=heads,d_model=d)
        m=GPT(**c,rng=np.random.default_rng(seed+100))
        rng=np.random.default_rng(seed+200)
        x=rng.integers(0,c['vocab_size'],(1+seed%2,2+seed%3))
        y=rng.integers(0,c['vocab_size'],x.shape)
        # Half the fixtures contain weights after real original eager Adam steps.
        steps=3 if seed%2 else 0
        opt=Adam(m.parameters(),lr=.005)
        for _ in range(steps):
            opt.zero_grad();cross_entropy(m(x),y).backward();opt.step()
        for group in range(layers+1):
            obj=dict(schema=SCHEMA,config=c,group_size=group,learning_rate=.01,
                     tokens=x.tolist(),targets=y.tolist(),weights=[dict(name=n,shape=list(p.shape),values=p.data.reshape(-1).tolist()) for n,p in named_parameters(m)])
            report=audit(obj);report['fixture']=dict(seed=seed,original_eager_training_steps=steps)
            if report['comparison']['max_gradient_delta']>1e-6:raise AssertionError('native gradient parity')
            reports.append(report)
    return dict(schema='scratchgpt.remat.witnesses.v1',reports=reports,
                count=len(reports),parameter_coordinates=sum(r['comparison']['parameter_coordinates'] for r in reports))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    out=build();dest=ROOT/'docs/remat-witnesses.json'
    if args.check:
        old=json.loads(dest.read_text())
        def compare(a,b):
            if isinstance(a,(int,float)) and not isinstance(a,bool):
                if abs(a-b)>2e-6*max(1,abs(a),abs(b)):raise AssertionError((a,b))
            elif isinstance(a,list):
                assert len(a)==len(b)
                for x,y in zip(a,b):compare(x,y)
            elif isinstance(a,dict):
                assert set(a)==set(b)
                for k in a:compare(a[k],b[k])
            else:assert a==b
        compare(out,old)
    else:dest.write_text(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(witnesses=out['count'],parameter_coordinates=out['parameter_coordinates'],checked=args.check)))


if __name__=='__main__':main()
