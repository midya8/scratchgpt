import json
from pathlib import Path
import numpy as np
import pytest
from scratchgpt.replay import forward,adam_scalar
from scratchgpt.checkpoint import Training,named_parameters
from scratchgpt.tensor import cross_entropy

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'docs/replay.json').read_text())


def test_complete_adam_witness():
    count=0
    for r in DATA['records']:
        for item in r['before']:
            for values in zip(item['p'],item['g'],item['m'],item['v'],item['after']['p'],item['after']['m'],item['after']['v']):
                p,g,m,v,pp,mm,vv=values
                np.testing.assert_allclose(adam_scalar(p,g,m,v,r['step'],r['lr']),(pp,mm,vv),rtol=1e-6,atol=2e-7)
                count+=3
    assert count==DATA['checks']['adam_scalar_values']==38016


def test_all_independent_forward_values():
    count=0
    for r in DATA['records']:
        w={i['name']:np.array(i['after']['p']).reshape(i['shape']) for i in r['before']}
        for x,expected in zip(r['x'],r['logits']):
            got=forward(w,DATA['config'],np.array(x))
            np.testing.assert_allclose(got,expected,rtol=1e-5,atol=2e-5)
            count+=got.size
    assert count==2592


@pytest.mark.parametrize('seed',[0,1,9])
@pytest.mark.parametrize('layers',[1,2])
def test_independent_forward_random_models(seed,layers):
    from scratchgpt.model import GPT
    cfg=dict(DATA['config']);cfg['n_layer']=layers
    model=GPT(rng=np.random.default_rng(seed),**cfg)
    w={n:p.data for n,p in named_parameters(model)}
    for t in [1,3,6]:
        x=np.random.default_rng(seed+t).integers(0,cfg['vocab_size'],size=t)
        np.testing.assert_allclose(forward(w,cfg,x),model(x[None]).data[0],rtol=1e-5,atol=2e-5)


def test_counterexamples_retained():
    b=DATA['boundaries']
    assert all(x['exact_state'] for x in b)
    assert sum(x['legacy_weight_delta']>0 for x in b)==11
    assert sum(x['missing_moment_delta']>0 for x in b)==11
    assert all(x['eval_only_weight_delta']==0 for x in b)
    assert not b[4]['eval_only_history_equal']
    assert b[0]['legacy_weight_delta']==b[12]['legacy_weight_delta']==0


def test_model_gradient_independent_forward_central_difference():
    run=Training(DATA['text']*DATA['repeats'],DATA['config'],DATA['options'])
    x=np.array([[0,1,2,3,4,5]]);y=np.array([[1,2,3,4,5,6]])
    loss=cross_entropy(run.model(x),y);loss.backward()
    w={n:p.data.astype(float).copy() for n,p in named_parameters(run.model)}
    def objective():
        logits=forward(w,DATA['config'],x[0]);z=logits-logits.max(axis=1,keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(axis=1))-z[np.arange(6),y[0]]))
    for n,p in named_parameters(run.model):
        for index in [0,p.data.size//2,p.data.size-1]:
            a=w[n].flat;v=a[index];h=1e-5
            a[index]=v+h;plus=objective();a[index]=v-h;minus=objective();a[index]=v
            np.testing.assert_allclose((plus-minus)/(2*h),p.grad.flat[index],rtol=3e-3,atol=2e-5)
