"""Opt-in deterministic activation rematerialization for the original GPT.

This is separate from durable training checkpoints. Tensor graph logical-data
counts are diagnostics, NOT allocator/RSS bytes or complete backward storage.
"""
from __future__ import annotations

import copy
import hashlib
from collections import Counter

import numpy as np

from .nn import Block, Module
from .tensor import Tensor, _acc, gather, layer_norm


def topology(root):
    """Iterative postorder, including shared leaves exactly once."""
    order, seen, stack = [], set(), [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
        elif id(node) not in seen:
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(node._prev))
    return order


def graph_report(root, excluded=()):
    """Reachable Tensor.data logical sizes. Views count per node, not per allocation.

    Excludes declared parameters/mask; does not inspect ndarray captures in
    backward closures, gradients, Python objects, BLAS buffers or audit arrays.
    """
    exclude = {id(p) for p in excluded}
    nodes = [n for n in topology(root) if id(n) not in exclude]
    return dict(tensor_nodes=len(nodes), logical_elements=sum(n.data.size for n in nodes),
                operations=dict(sorted(Counter(n._op or 'leaf' for n in nodes).items())))


def _digest(t):
    return (t.shape, str(t.data.dtype), hashlib.sha256(t.data.tobytes()).hexdigest())


def _bind(module, leaves):
    """Same original module classes/forward rules, fresh private Tensor leaves."""
    clone = copy.copy(module)
    for key, value in module.__dict__.items():
        if isinstance(value, Tensor):
            setattr(clone, key, next(leaves))
        elif isinstance(value, Module):
            setattr(clone, key, _bind(value, leaves))
    return clone


def _block_signature(block):
    return (type(block), type(block.attn), type(block.mlp), block.attn.n_head,
            block.attn.dropout, block.mlp.dropout, tuple(id(p) for p in block.parameters()))


def _region(x, blocks, mask, events, index):
    # Only the known original modules, deterministic no-dropout path. No arbitrary
    # user callable or implicit/global random state is accepted by this interface.
    if any(type(b) is not Block or b.attn.dropout or b.mlp.dropout for b in blocks):
        raise ValueError('rematerialization requires original deterministic Block modules with dropout=0')
    ps = [p for b in blocks for p in b.parameters()]
    if any(p._prev for p in ps):
        raise ValueError('parameters must be leaf Tensors')
    if len(set(map(id, ps))) != len(ps):
        raise ValueError('tied parameters inside a region are not supported')
    original = [x, *ps, mask]
    signatures = [_block_signature(b) for b in blocks]
    digests = [_digest(t) for t in original]

    def run():
        # Copies are ephemeral and confined to this invocation. They are not
        # retained by the outer graph. Never clear or overwrite original grads.
        xx = Tensor(x.data.copy())
        pp = [Tensor(p.data.copy()) for p in ps]
        it = iter(pp)
        yy = xx
        mm = Tensor(mask.data.copy())
        for block in blocks:
            yy = _bind(block, it)(yy, mm)
        return xx, pp, mm, yy

    xx, pp, mm, result = run()
    event = dict(region=index, blocks=len(blocks), phase='forward',
                 interior=graph_report(result, [*pp, mm]))
    events.append(event)
    output = Tensor(result.data, (x, *ps), 'rematerialize')
    expected = _digest(output)
    # Do not capture result, its graph, cloned parameter arrays or input copies.
    del result, xx, pp, mm
    used = False

    def backward(node):
        nonlocal used
        if used:
            raise ValueError('use a fresh forward graph for each backward pass')
        used = True
        if signatures != [_block_signature(b) for b in blocks] or digests != [_digest(t) for t in original]:
            raise ValueError('rematerialization inputs, parameters, mask or block settings changed before backward')
        xx, pp, mm, yy = run()
        if _digest(yy) != expected:
            raise ValueError('rematerialized output differs from forward')
        events.append(dict(region=index, blocks=len(blocks), phase='recompute',
                           interior=graph_report(yy, [*pp, mm])))
        yy.grad = node.grad.copy()
        for local in reversed(topology(yy)):
            if local.grad is not None:
                local._backward(local)
        # Publish only after a successful complete local VJP.
        _acc(x, xx.grad)
        for original_p, private_p in zip(ps, pp):
            if private_p.grad is not None:
                _acc(original_p, private_p.grad)

    output._backward = backward
    return output


def forward(model, tokens, group_size=1, events=None):
    """Real original GPT forward, checkpointing contiguous groups of blocks.

    group_size=0 is the original eager path. Positive groups recompute during
    backward. First-order differentiation only, deterministic dropout=0, no
    mutation between forward/backward. Standard Tensor.backward is unchanged.
    """
    if type(group_size) is not int or not 0 <= group_size <= model.n_layer:
        raise ValueError('group_size must be an integer from 0 to n_layer')
    a = np.asarray(tokens)
    if a.dtype.kind not in 'iu' or a.ndim != 2 or a.shape[0] < 1 or not 1 <= a.shape[1] <= model.block_size:
        raise ValueError('tokens must be a nonempty rank-2 integer array within block_size')
    if np.any(a < 0) or np.any(a >= model.vocab_size):
        raise ValueError('token outside vocabulary')
    if model.dropout or any(b.attn.dropout or b.mlp.dropout for b in model.blocks):
        raise ValueError('this API supports dropout=0 only')
    if group_size == 0:
        return model(a)
    events = events if events is not None else []
    x = gather(model.tok_emb.w, a) + gather(model.pos_emb.w, np.arange(a.shape[1]))
    mask = model.mask(a.shape[1])
    for start in range(0, model.n_layer, group_size):
        x = _region(x, model.blocks[start:start+group_size], mask, events, start)
    x = layer_norm(x, model.ln_f.w, model.ln_f.b)
    return x @ model.head.w
