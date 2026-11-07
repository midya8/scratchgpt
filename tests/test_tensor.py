import numpy as np

from scratchgpt.tensor import (Tensor, cross_entropy, dropout, gather,
                               gelu, layer_norm, log_softmax, pick_last,
                               softmax)

rng = np.random.default_rng(0)


def numgrad(f, x, eps=1e-2):
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        ix = it.multi_index
        old = x[ix]
        x[ix] = old + eps
        fp = f()
        x[ix] = old - eps
        fm = f()
        x[ix] = old
        g[ix] = (fp - fm) / (2 * eps)
    return g


def check(build, x, tol=5e-2):
    x64 = x.astype(np.float64)
    t = Tensor(x.copy())
    out = build(t)
    out.backward()
    num = numgrad(lambda: float(build(Tensor(x64)).data), x64)
    assert np.allclose(t.grad, num, atol=tol, rtol=tol), \
        f"max abs diff {np.abs(t.grad - num).max():.4g}"


def test_arithmetic():
    x = rng.normal(size=(4, 3)).astype(np.float32)
    y = rng.normal(size=(4, 3)).astype(np.float32) + 3.0
    check(lambda t: (t * 2.0 + 1.0).sum(), x)
    check(lambda t: (t * t).sum(), x)
    check(lambda t: (t / 3.0).sum(), x)
    check(lambda t: (t / (t * 0.5 + y)).sum(), x)


def test_matmul():
    x = rng.normal(size=(3, 4)).astype(np.float32)
    w = rng.normal(size=(4, 5)).astype(np.float32)
    check(lambda t: (t @ w).sum(), x)
    xb = rng.normal(size=(2, 3, 4)).astype(np.float32)
    wb = rng.normal(size=(2, 4, 5)).astype(np.float32)
    check(lambda t: (t @ wb).sum(), xb)
    check(lambda t: (xb @ t).sum(), wb)


def test_activations():
    x = (rng.normal(size=(4, 5)) + 0.5).astype(np.float32)
    check(lambda t: t.tanh().sum(), x)
    check(lambda t: t.sigmoid().sum(), x)
    check(lambda t: gelu(t).sum(), x)
    xr = rng.choice(np.array([-2.0, -1.0, 1.0, 3.0]), size=(4, 5)).astype(np.float32)
    check(lambda t: t.relu().sum(), xr)
    pos = (np.abs(x) + 0.5).astype(np.float32)
    check(lambda t: (t.log() + t.sqrt()).sum(), pos)
    check(lambda t: t.pow(3).sum(), x)


def test_reductions_and_shapes():
    x = rng.normal(size=(3, 4, 5)).astype(np.float32)
    check(lambda t: t.sum(axis=1).sum(), x)
    check(lambda t: t.mean(axis=(0, 2)).sum(), x)
    check(lambda t: t.view((4, 15)).sum(), x)
    check(lambda t: t.permute((2, 0, 1)).sum(), x)
    check(lambda t: t.select(1).sum(), x)
    check(lambda t: t.mean(axis=-1, keepdims=True).sum(), x)


def test_softmax_and_cross_entropy():
    x = rng.normal(size=(4, 7)).astype(np.float32)
    w = rng.normal(size=(4, 7)).astype(np.float32)
    check(lambda t: (softmax(t, -1) * w).sum(), x)
    check(lambda t: (log_softmax(t, -1) * w).sum(), x)
    targets = rng.integers(0, 7, size=(4, 3))
    xb = rng.normal(size=(4, 3, 7)).astype(np.float32)
    check(lambda t: cross_entropy(t, targets), xb)
    check(lambda t: cross_entropy(t, targets) * 2.0, xb)


def test_layer_norm_and_gather():
    x = rng.normal(size=(2, 5, 6)).astype(np.float32)
    w = rng.normal(size=(6,)).astype(np.float32)
    b = rng.normal(size=(6,)).astype(np.float32)
    check(lambda t: layer_norm(t, w, b).sum(), x)
    idx = rng.integers(0, 6, size=(2, 3))
    e = rng.normal(size=(6, 4)).astype(np.float32)
    check(lambda t: (gather(t, idx) ** 2).sum(), e)
    ids = rng.integers(0, 5, size=(2, 5))
    check(lambda t: (pick_last(t, ids) ** 2).sum(), x)


def test_dropout_zero_p_is_identity():
    x = rng.normal(size=(3, 4)).astype(np.float32)
    t = Tensor(x.copy())
    out = dropout(t, 0.0, np.random.default_rng(1))
    assert np.allclose(out.data, x)
