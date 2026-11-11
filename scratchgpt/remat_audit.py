"""Bounded, replayable native rematerialization audit. python -m scratchgpt.remat_audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import config_validate, fields, integer, finite, named_parameters
from .model import GPT
from .remat import forward, graph_report
from .tensor import cross_entropy

SCHEMA = 'scratchgpt.remat.input.v1'


def fixture(seed=7, layers=2, group=1):
    c = dict(vocab_size=5, block_size=4, n_layer=layers, n_head=2, d_model=4)
    m = GPT(**c, rng=np.random.default_rng(seed))
    return dict(schema=SCHEMA, config=c, group_size=group, learning_rate=.01,
                tokens=[[0,1,2]], targets=[[1,2,3]],
                weights=[dict(name=n, shape=list(p.shape), values=p.data.reshape(-1).tolist()) for n,p in named_parameters(m)])


def parse(obj):
    fields(obj, ('schema','config','group_size','learning_rate','tokens','targets','weights'), 'input')
    if obj['schema'] != SCHEMA: raise ValueError('unsupported input schema')
    c = obj['config']; config_validate(c)
    if c['d_model'] > 8 or c['n_layer'] > 3 or c['vocab_size'] > 12 or c['block_size'] > 6:
        raise ValueError('audit bounds: d_model<=8, layers<=3, vocabulary<=12, block_size<=6')
    integer(obj['group_size'], 0, c['n_layer'], 'group_size')
    finite(obj['learning_rate'], 0, .1, 'learning_rate')
    arrays = []
    for k in ('tokens','targets'):
        rows = obj[k]
        if type(rows) is not list or not 1 <= len(rows) <= 2 or any(type(r) is not list for r in rows):
            raise ValueError(k+': expected 1-2 rows')
        width = len(rows[0])
        if not 1 <= width <= c['block_size'] or any(len(r) != width for r in rows):
            raise ValueError(k+': invalid sequence dimensions')
        for r in rows:
            for v in r: integer(v, 0, c['vocab_size']-1, k)
        arrays.append(np.array(rows, dtype=np.int64))
    if arrays[0].shape != arrays[1].shape: raise ValueError('tokens and targets must have identical shapes')
    m = GPT(**c, rng=np.random.default_rng(0))
    if type(obj['weights']) is not list or len(obj['weights']) != len(m.parameters()):
        raise ValueError('weight list must match every named parameter')
    for row, (name,p) in zip(obj['weights'], named_parameters(m)):
        fields(row, ('name','shape','values'), 'weight')
        if row['name'] != name or row['shape'] != list(p.shape) or any(type(v) is not int for v in row['shape']):
            raise ValueError('weight name/shape/order mismatch')
        if type(row['values']) is not list or len(row['values']) != p.data.size: raise ValueError('weight length mismatch')
        for v in row['values']: finite(v, -10, 10, 'weight')
        p.data[...] = np.array(row['values'], dtype=np.float32).reshape(p.shape)
    return m, *arrays


def audit(obj):
    m,x,y = parse(obj)
    arms=[]
    for size in (0, obj['group_size']):
        for p in m.parameters(): p.grad=None
        events=[]
        logits=forward(m,x,size,events)
        loss=cross_entropy(logits,y)
        retained=graph_report(loss,[*m.parameters(),m.mask(x.shape[1])])
        loss.backward()
        params=[]
        for n,p in named_parameters(m):
            updated=p.data-obj['learning_rate']*p.grad
            params.append(dict(name=n,shape=list(p.shape),gradient=p.grad.reshape(-1).tolist(),
                               updated=updated.reshape(-1).tolist()))
        arms.append(dict(loss=float(loss.data),logits=logits.data.tolist(),retained=retained,events=events,parameters=params))
    gradient_delta=max(abs(a-b) for p,q in zip(arms[0]['parameters'],arms[1]['parameters']) for a,b in zip(p['gradient'],q['gradient']))
    update_delta=max(abs(a-b) for p,q in zip(arms[0]['parameters'],arms[1]['parameters']) for a,b in zip(p['updated'],q['updated']))
    return dict(schema='scratchgpt.remat.report.v1',input=obj,eager=arms[0],rematerialized=arms[1],
                comparison=dict(max_gradient_delta=gradient_delta,max_update_delta=update_delta,
                                parameter_coordinates=m.n_params,logical_elements_removed=arms[0]['retained']['logical_elements']-arms[1]['retained']['logical_elements']),
                scope='First-order deterministic original GPT. Logical Tensor.data elements, not allocator bytes/RSS or complete saved-backward storage; parameter/mask nodes, closure arrays, gradients, audit copies and Python/BLAS buffers excluded. Views counted per Tensor. SGD arithmetic is a one-step comparison, not a new training trajectory. Original trainer/checkpoint format unchanged.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',nargs='?')
    ap.add_argument('--fixture',action='store_true')
    args=ap.parse_args()
    try:
        if args.fixture:
            if args.input: raise ValueError('use --fixture without an input file')
            out=fixture()
        else:
            if not args.input: raise ValueError('provide input JSON, or --fixture to print a complete example')
            path=Path(args.input)
            if path.stat().st_size>500_000: raise ValueError('input exceeds 500 KB')
            out=audit(json.loads(path.read_text(),parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON'))))
        print(json.dumps(out,sort_keys=True,allow_nan=False))
    except (ValueError,TypeError,KeyError,OSError,OverflowError) as e:
        ap.exit(2,f'rematerialization audit error: {e}\n')


if __name__ == '__main__': main()
