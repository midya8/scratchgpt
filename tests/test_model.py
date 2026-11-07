import numpy as np

from scratchgpt.model import GPT
from scratchgpt.optim import Adam
from scratchgpt.tensor import cross_entropy


def tiny_model():
    rng = np.random.default_rng(0)
    return GPT(vocab_size=20, block_size=8, n_layer=1, n_head=2,
               d_model=16, rng=rng)


def test_forward_shape_and_loss():
    m = tiny_model()
    x = np.random.default_rng(1).integers(0, 20, size=(2, 8))
    logits = m(x)
    assert logits.shape == (2, 8, 20)
    loss = cross_entropy(logits, x)
    assert np.isfinite(float(loss.data))


def test_training_reduces_loss():
    rng = np.random.default_rng(2)
    m = tiny_model()
    x = rng.integers(0, 20, size=(4, 8))
    opt = Adam(m.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        loss = cross_entropy(m(x), x)
        if first is None:
            first = float(loss.data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.data) < first - 0.5


def test_no_future_leak():
    m = tiny_model()
    rng = np.random.default_rng(3)
    x1 = rng.integers(0, 20, size=(1, 8))
    x2 = x1.copy()
    x2[0, 5] = (x2[0, 5] + 1) % 20
    l1 = m(x1).data[0, :5]
    l2 = m(x2).data[0, :5]
    assert np.allclose(l1, l2, atol=1e-5)


def test_sampling_stays_in_vocab():
    m = tiny_model()
    rng = np.random.default_rng(4)
    idx = [1, 2, 3]
    for _ in range(10):
        ctx = np.array(idx[-8:])[None]
        logits = m(ctx).data[0, -1]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        idx.append(int(rng.choice(len(p), p=p)))
    assert all(0 <= i < 20 for i in idx[3:])
