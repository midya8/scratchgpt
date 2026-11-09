"""Reproduce complete training witnesses and restart counterexamples, offline."""
import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import numpy as np

from scratchgpt.checkpoint import Training, named_parameters
from scratchgpt.replay import forward, adam_scalar

TEXT = 'red fox\nblue bird\nred bird\nblue fox\n'*60
CFG = dict(vocab_size=len(set(TEXT)),block_size=6,n_layer=1,n_head=2,d_model=8)
OPT = dict(batch=3,lr=.003,warmup=5,clip=.4,eval_every=3,eval_batches=2,seed=7)
ROOT=Path(__file__).resolve().parents[1]


def weights(a):return {n:p.data for n,p in named_parameters(a.model)}


def max_delta(a,b):return max(float(np.max(np.abs(p.data.astype(float)-q.data))) for p,q in zip(a.model.parameters(),b.model.parameters()))


def build():
    a=Training(TEXT,CFG,OPT)
    records=[];max_adam=0.;max_forward=0.;scalar_count=0;forward_count=0
    for _ in range(12):
        row=a.update(witness=True)
        for item,p,m,v in zip(row['before'],a.model.parameters(),a.opt.m,a.opt.v):
            for values in zip(item['p'].flat,item['g'].flat,item['m'].flat,item['v'].flat,p.data.flat,m.flat,v.flat):
                pp,gg,mm,vv,pa,ma,va=map(float,values)
                ref=adam_scalar(pp,gg,mm,vv,row['step'],row['lr'])
                max_adam=max(max_adam,*[abs(x-y) for x,y in zip(ref,(pa,ma,va))]);scalar_count+=3
            item['shape']=list(p.shape)
            for k in ('p','g','m','v'):item[k]=item[k].reshape(-1).tolist()
            item['after']={'p':p.data.reshape(-1).tolist(),'m':m.reshape(-1).tolist(),'v':v.reshape(-1).tolist()}
        row['logits']=a.model(np.array(row['x'])).data.tolist()
        for tokens,logits in zip(row['x'],row['logits']):
            ref=forward(weights(a),CFG,np.array(tokens))
            max_forward=max(max_forward,float(np.max(np.abs(ref-logits))));forward_count+=ref.size
        row['state_sha256']=a.digest();row['val_loss']=a.history[-1][2] if a.step%3==0 else None
        records.append(row)
    full=a;boundaries=[]
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'state.npz'
        for split in range(13):
            cut=Training(TEXT,CFG,OPT)
            for _ in range(split):cut.update()
            cut.save(path)
            resume=Training.load(path,TEXT,CFG,OPT)
            # These ablations change one state category, not the training algorithm.
            legacy=Training.load(path,TEXT,CFG,OPT)
            legacy.rng=Training(TEXT,CFG,OPT).rng
            legacy.eval_rng=np.random.default_rng(OPT['seed']+1)
            moments=Training.load(path,TEXT,CFG,OPT)
            for buffer in moments.opt.m+moments.opt.v:buffer.fill(0)
            evalonly=Training.load(path,TEXT,CFG,OPT)
            evalonly.eval_rng=np.random.default_rng(OPT['seed']+1)
            for _ in range(12-split):
                for run in (resume,legacy,moments,evalonly):run.update()
            if resume.digest()!=full.digest():raise AssertionError('exact continuation failed')
            boundaries.append(dict(split=split,exact_state=True,exact_sha256=resume.digest(),
                                   legacy_weight_delta=max_delta(full,legacy),
                                   missing_moment_delta=max_delta(full,moments),
                                   eval_only_weight_delta=max_delta(full,evalonly),
                                   eval_only_history_equal=evalonly.history==full.history,
                                   legacy_history_equal=legacy.history==full.history))
    if max_adam>2e-7 or max_forward>2e-5:raise AssertionError((max_adam,max_forward))
    shipped=np.load(ROOT/'checkpoints/final.npz',allow_pickle=False)
    shipped_count=sum(shipped[k].size for k in shipped.files if k.startswith('p'))
    if shipped_count!=419328:raise AssertionError(shipped_count)
    return dict(schema='scratchgpt.replay.v1',text=TEXT[:len(TEXT)//60],repeats=60,
                config=CFG,options=OPT,parameters=full.model.n_params,vocabulary=full.vocab.itos,
                records=records,boundaries=boundaries,
                checks=dict(adam_scalar_values=scalar_count,adam_max_abs=max_adam,
                            forward_values=forward_count,forward_max_abs=max_forward),
                shipped=dict(parameters=shipped_count,checkpoint_sha256=hashlib.sha256((ROOT/'checkpoints/final.npz').read_bytes()).hexdigest()),
                note='Synthetic replay is a separate small model. Historical Shakespeare checkpoint is not retrained. Browser recomputes forward inference and Adam from recorded gradients, not backward/training.')


def compare(a,b,path='root'):
    if isinstance(a,(int,float)) and not isinstance(a,bool):
        if not isinstance(b,(int,float)) or not np.isclose(a,b,rtol=1e-6,atol=1e-6):
            raise AssertionError(path)
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError(path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(i))
    elif isinstance(a,dict):
        keys=set(a)-{'state_sha256','exact_sha256'}
        if keys!=set(b)-{'state_sha256','exact_sha256'}:raise AssertionError(path)
        for k in keys:compare(a[k],b[k],path+'.'+k)
    elif a!=b:raise AssertionError(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    data=build();p=ROOT/'docs/replay.json'
    if args.check:
        compare(data,json.loads(p.read_text()))
    else:p.write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(check=args.check,steps=len(data['records']),boundaries=len(data['boundaries']),parameters=data['parameters'],checks=data['checks'],shipped=data['shipped'])))


if __name__=='__main__':main()
