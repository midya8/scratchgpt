import copy
import gc

import numpy as np
import pytest

from scratchgpt.model import GPT
from scratchgpt.nn import Block
from scratchgpt.optim import Adam
from scratchgpt.tensor import Tensor, cross_entropy
from scratchgpt.remat import forward, graph_report, topology, _region


def model(seed=7, layers=3, d=8, heads=2):
    return GPT(7, 6, layers, heads, d, np.random.default_rng(seed))


def gradients(m, x, y, group):
    for p in m.parameters():
        p.grad = None
    events = []
    logits = forward(m, x, group, events)
    loss = cross_entropy(logits, y)
    report = graph_report(loss, [*m.parameters(), m.mask(x.shape[1])])
    loss.backward()
    return logits.data.copy(), float(loss.data), [p.grad.copy() for p in m.parameters()], report, events


def test_every_coordinate_all_groups_and_adam_trajectory():
    rng = np.random.default_rng(91)
    x, y = [rng.integers(0, 7, (2, 5)) for _ in range(2)]
    runs = [model() for _ in range(4)]
    opts = [Adam(m.parameters(), lr=.002) for m in runs]
    for step in range(5):
        results = [gradients(m, x, y, g) for g, m in enumerate(runs)]
        for g, result in enumerate(results[1:], 1):
            np.testing.assert_array_equal(results[0][0], result[0])
            assert results[0][1] == result[1]
            for a, b in zip(results[0][2], result[2]):
                np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-5)
            assert result[3]['logical_elements'] < results[0][3]['logical_elements']
            events = result[4]
            assert [e['region'] for e in events if e['phase'] == 'recompute'] == list(reversed([e['region'] for e in events if e['phase'] == 'forward']))
            assert sum(e['blocks'] for e in events if e['phase'] == 'recompute') == 3
            before = {e['region']: e['interior'] for e in events if e['phase'] == 'forward'}
            assert all(e['interior'] == before[e['region']] for e in events if e['phase'] == 'recompute')
        for opt in opts:
            opt.step()
        for m in runs[1:]:
            for p, q in zip(runs[0].parameters(), m.parameters()):
                np.testing.assert_allclose(p.data, q.data, atol=3e-7, rtol=1e-4)


def test_seeded_shapes_batches_and_nonuniform_upstream():
    rng = np.random.default_rng(333)
    for i in range(30):
        layers = 1 + i % 4
        m = model(i, layers, 2*(1+i%4), 2)
        x = rng.integers(0, 7, (1+i%3, 1+i%6))
        weights = rng.normal(size=(*x.shape, 7)).astype('float32')
        ref = m(x)
        (ref * Tensor(weights)).sum().backward()
        gs = [p.grad.copy() for p in m.parameters()]
        for g in range(1, layers+1):
            for p in m.parameters(): p.grad = None
            out = forward(m, x, g)
            np.testing.assert_array_equal(out.data, ref.data)
            (out*Tensor(weights)).sum().backward()
            for a, p in zip(gs, m.parameters()):
                np.testing.assert_allclose(a, p.grad, atol=1e-5, rtol=1e-4)


def test_shared_fanout_existing_parameter_grads_and_input_graph():
    rng = np.random.default_rng(4)
    b = Block(4, 2, rng)
    raw = rng.normal(size=(1, 3, 4)).astype('float32')
    mask = model().mask(3)
    for remat in (False, True):
        leaf = Tensor(raw)
        x = leaf * 1.2
        for p in b.parameters(): p.grad = np.full(p.shape, .125, dtype='float32')
        if remat:
            a = _region(x, [b], mask, [], 0)
            c = _region(x, [b], mask, [], 1)
        else:
            a, c = b(x, mask), b(x, mask)
        ((a*a).sum()+c.sum()).backward()
        now = [leaf.grad.copy(), *[p.grad.copy() for p in b.parameters()]]
        if not remat: ref = now
        else:
            for v, w in zip(ref, now): np.testing.assert_allclose(v, w, atol=1e-6, rtol=2e-5)


@pytest.mark.parametrize('which', ['input','weight','mask','metadata'])
def test_mutation_rejected_before_local_vjp(which):
    m = model(layers=1)
    x = Tensor(np.ones((1, 3, 8)))
    mask = m.mask(3)
    out = _region(x, m.blocks, mask, [], 0)
    if which == 'input': x.data[0,0,0] += .1
    elif which == 'weight': m.blocks[0].attn.qkv.w.data[0,0] += .1
    elif which == 'mask': mask.data[0,0,0,0] += .1
    else: m.blocks[0].attn.n_head = 1
    with pytest.raises(ValueError, match='changed'):
        out.sum().backward()
    assert all(p.grad is None for p in m.parameters())


def test_actual_discarded_forward_graph_not_retained_by_closure():
    m = model(layers=2)
    x = np.array([[0,1,2]])
    gc.collect()
    before = {id(o) for o in gc.get_objects() if isinstance(o, Tensor)}
    out = forward(m, x, 2)
    gc.collect()
    outer = {id(o) for o in topology(out)}
    live_new = [o for o in gc.get_objects() if isinstance(o, Tensor) and id(o) not in before]
    # Only the outer graph and model's newly cached mask survive. No private
    # parameter clones, input clones or internal forward Tensor nodes remain.
    assert all(id(o) in outer or o is m.mask(3) for o in live_new)
    assert sum(o._op == 'rematerialize' for o in topology(out)) == 1


@pytest.mark.parametrize('bad', [-1,4,True,1.2,None])
def test_bad_group(bad):
    with pytest.raises(ValueError): forward(model(), np.array([[1]]), bad)


@pytest.mark.parametrize('x', [[], [[-1]], [[7]], [[1.0]], [[True]], [[[1]]], [list(range(7))]])
def test_bad_tokens(x):
    with pytest.raises(ValueError): forward(model(), x)


def test_dropout_and_tied_region_rejected():
    m = model()
    m.blocks[0].attn.dropout = .1
    with pytest.raises(ValueError, match='dropout'): forward(m, [[1]])
    b = Block(4, 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match='tied'):
        _region(Tensor(np.ones((1,2,4))), [b,b], model().mask(2), [], 0)


def test_topology_shared_node_order():
    x = Tensor([2.]); a = x*x; b = a+a; c = a*b
    order = topology(c)
    assert len(order) == len(set(map(id,order)))
    pos = {id(t):i for i,t in enumerate(order)}
    assert all(pos[id(child)] < pos[id(t)] for t in order for child in t._prev)


def test_all_parameter_gradients_independent_float64_finite_differences():
    from scratchgpt.checkpoint import named_parameters
    from scratchgpt.replay import forward as reference
    m=model(5,2,4,2)
    tokens=np.array([[0,2,4]])
    upstream=np.random.default_rng(1).normal(size=(3,7))
    weights={n:p.data.astype('float64') for n,p in named_parameters(m)}
    cfg=dict(vocab_size=7,block_size=6,n_layer=2,n_head=2,d_model=4)
    out=forward(m,tokens,1)
    (out*Tensor(upstream[None])).sum().backward()
    eps=1e-5
    for n,p in named_parameters(m):
        numerical=np.zeros(p.shape)
        for idx in np.ndindex(p.shape):
            old=weights[n][idx]
            weights[n][idx]=old+eps
            plus=float((reference(weights,cfg,tokens[0])*upstream).sum())
            weights[n][idx]=old-eps
            minus=float((reference(weights,cfg,tokens[0])*upstream).sum())
            weights[n][idx]=old
            numerical[idx]=(plus-minus)/(2*eps)
        np.testing.assert_allclose(p.grad,numerical,atol=5e-6,rtol=2e-3,err_msg=n)


def test_original_trained_checkpoint_all_419328_gradients_and_logits():
    import json
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'checkpoints/final.npz'
    with np.load(path,allow_pickle=False) as z:
        cfg=json.loads(str(z['config']))
        m=GPT(**cfg,rng=np.random.default_rng(0))
        for i,p in enumerate(m.parameters()):p.data[...]=z[f'p{i}']
    assert m.n_params==419328
    x=np.array([[1,2,3,4,5,6,7,8]])
    y=np.array([[2,3,4,5,6,7,8,9]])
    ref=gradients(m,x,y,0)
    for group in range(1,m.n_layer+1):
        got=gradients(m,x,y,group)
        np.testing.assert_array_equal(ref[0],got[0])
        for a,b in zip(ref[2],got[2]):np.testing.assert_allclose(a,b,atol=1e-7,rtol=2e-5)


def test_single_use_no_external_mask_gradient_and_replaced_parameter_guard():
    m=model(layers=1)
    out=forward(m,[[1,2,3]],1)
    loss=out.sum();loss.backward()
    assert m.mask(3).grad is None
    with pytest.raises(ValueError,match='fresh forward'):loss.backward()
    m=model(layers=1);out=forward(m,[[1,2]],1)
    m.blocks[0].ln1.w=Tensor(m.blocks[0].ln1.w.data.copy())
    with pytest.raises(ValueError,match='changed'):out.sum().backward()
