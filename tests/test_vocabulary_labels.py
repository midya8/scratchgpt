"""Finite token-label consistency, not an independent transformer oracle."""
import copy

import numpy as np

from scratchgpt.inference import Inference
from scratchgpt.model import GPT
from scratchgpt.optim import Adam
from scratchgpt.remat import forward
from scratchgpt.tensor import cross_entropy


def relabel(model, new_to_old):
    other = copy.deepcopy(model)
    other.tok_emb.w.data[:] = model.tok_emb.w.data[new_to_old]
    other.head.w.data[:] = model.head.w.data[:, new_to_old]
    return other


def mapped(model, values, permutation):
    # Parameter identity matters: square internal matrices need no relabeling.
    out = []
    for p, a in zip(model.parameters(), values):
        if p is model.tok_emb.w:
            a = a[permutation]
        elif p is model.head.w:
            a = a[:, permutation]
        out.append(a)
    return out


def close(a, b):
    np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5)


def pair(seed):
    rng = np.random.default_rng(seed)
    model = GPT(7, 6, 1 + seed % 3, 1 + seed % 2, 8, rng)
    permutation = rng.permutation(7)
    if np.array_equal(permutation, np.arange(7)):
        permutation = np.roll(permutation, 1)
    return model, relabel(model, permutation), permutation, np.argsort(permutation), rng


def gradients(model, x, y, group):
    for p in model.parameters():
        p.grad = None
    z = forward(model, x, group)
    loss = cross_entropy(z, y)
    loss.backward()
    return z.data, loss.data, [p.grad.copy() for p in model.parameters()]


def test_vocabulary_labels_preserve_all_gradients_and_adam_buffers():
    for seed in range(12):
        a, b, permutation, inverse, rng = pair(seed)
        opts = [Adam(m.parameters(), lr=.0013) for m in (a, b)]
        for step in range(3):
            # Repeated IDs exercise embedding-gradient accumulation; vocabulary
            # size seven differs from hidden width eight, avoiding axis guesses.
            x = rng.integers(0, 4, (2, 5))
            y = rng.integers(0, 7, (2, 5))
            za, la, ga = gradients(a, x, y, 0)
            zb, lb, gb = gradients(b, inverse[x], inverse[y], 0)
            close(za[..., permutation], zb)
            close(la, lb)
            for expected, got in zip(mapped(a, ga, permutation), gb):
                close(expected, got)
            for opt in opts:
                opt.step()
            assert opts[0].t == opts[1].t == step + 1
            for attr in ('m', 'v'):
                for expected, got in zip(mapped(a, getattr(opts[0], attr), permutation), getattr(opts[1], attr)):
                    close(expected, got)
            for expected, p in zip(mapped(a, [p.data for p in a.parameters()], permutation), b.parameters()):
                close(expected, p.data)


def test_vocabulary_labels_preserve_rematerialized_parameter_gradients():
    for seed in range(6):
        a, b, permutation, inverse, rng = pair(seed + 20)
        x = rng.integers(0, 7, (2, 6))
        y = rng.integers(0, 7, (2, 6))
        za, la, ga = gradients(a, x, y, 0)
        for group in range(1, a.n_layer + 1):
            zb, lb, gb = gradients(b, inverse[x], inverse[y], group)
            close(za[..., permutation], zb)
            close(la, lb)
            for expected, got in zip(mapped(a, ga, permutation), gb):
                close(expected, got)


def test_cached_rollover_relabels_logits_but_not_kv_or_work_counts():
    for seed in range(12):
        a, b, permutation, inverse, rng = pair(seed + 40)
        fa, fb = Inference(a), Inference(b)
        sa, sb = fa.session(), fb.session()
        tokens = rng.integers(0, 7, (2, 17))
        start = 0
        for end in (2, 3, 6, 8, 11, 12, 17):
            za = sa.advance(tokens[:, start:end])
            zb = sb.advance(inverse[tokens[:, start:end]])
            close(za[..., permutation], zb)
            window = tokens[:, max(0, end - 6):end]
            close(za[:, -1], a(window).data[:, -1])
            close(zb[:, -1], b(inverse[window]).data[:, -1])
            assert sa.length == sb.length == min(end, 6)
            assert sa.counts == sb.counts
            assert sa.cache_nbytes == sb.cache_nbytes
            for pa, pb in zip(sa._past, sb._past):
                for ka, kb in zip(pa, pb):
                    np.testing.assert_array_equal(ka, kb)
            start = end
        assert sa.counts['window_rebuilds'] == 4


def test_one_sided_relabeling_is_not_a_valid_control():
    a, b, permutation, inverse, rng = pair(100)
    x = np.tile(np.arange(6), (2, 1))
    expected = a(x).data[..., permutation]
    close(expected, b(inverse[x]).data)
    wrong_input = b(x).data
    wrong_head = copy.deepcopy(b)
    wrong_head.head.w.data[:] = a.head.w.data
    assert float(np.max(np.abs(expected - wrong_input))) > 1e-4
    assert float(np.max(np.abs(expected - wrong_head(inverse[x]).data))) > 1e-4
