import json
from pathlib import Path

import numpy as np
import pytest

from scratchgpt.inference import Inference
from scratchgpt.model import GPT


def model(seed=0, layers=2, heads=2, width=12, batch=1, context=13):
    m = GPT(
        vocab_size=19,
        block_size=context,
        n_layer=layers,
        n_head=heads,
        d_model=width,
        rng=np.random.default_rng(seed),
    )
    x = np.random.default_rng(seed + 20).integers(0, 19, (batch, context))
    return m, x


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


def state(s):
    return (
        s.length,
        s.cache_nbytes,
        s.counts,
        tuple(a.tobytes() for pair in s._past for a in pair),
        None if s._tokens is None else s._tokens.tobytes(),
    )


@pytest.mark.parametrize(
    "layers,heads,width,batch", [(1, 1, 8, 1), (2, 2, 12, 2), (3, 3, 12, 3)]
)
def test_full_and_every_chunk_boundary(layers, heads, width, batch):
    m, x = model(layers=layers, heads=heads, width=width, batch=batch)
    inf = Inference(m)
    expected = m(x).data
    close(inf.forward(x), expected)
    for boundary in range(1, x.shape[1]):
        s = inf.session()
        a = s.prefill(x[:, :boundary])
        b = s.append(x[:, boundary:])
        close(np.concatenate((a, b), axis=1), expected)
        assert s.length == x.shape[1]
        assert s.cache_nbytes == 2 * batch * x.shape[1] * width * layers * 4


def test_100_random_models_and_chunkings():
    rng = np.random.default_rng(919)
    for seed in range(100):
        m, x = model(
            seed,
            layers=1 + seed % 3,
            heads=1 + seed % 2,
            width=12,
            batch=1 + seed % 3,
            context=1 + seed % 17,
        )
        inf, outputs = Inference(m), []
        s = inf.session()
        i = 0
        while i < x.shape[1]:
            end = min(x.shape[1], i + int(rng.integers(1, 5)))
            outputs.append(s.prefill(x[:, i:end]) if i == 0 else s.append(x[:, i:end]))
            i = end
        close(np.concatenate(outputs, axis=1), m(x).data)
        close(inf.forward(x), m(x).data)


def test_prefill_and_chunk_append_are_causal():
    m, x = model(context=10)
    # Larger but finite weights make accidental future attention easy to detect.
    for p in m.parameters():
        p.data *= np.float32(3)
    inf = Inference(m)
    changed = x.copy()
    changed[:, 6:] = (changed[:, 6:] + 5) % 19
    a = inf.session()
    b = inf.session()
    first = a.prefill(x[:, :3])
    other = b.prefill(changed[:, :3])
    np.testing.assert_array_equal(first, other)
    tail = a.append(x[:, 3:])
    tail2 = b.append(changed[:, 3:])
    np.testing.assert_array_equal(tail[:, :3], tail2[:, :3])
    close(tail, m(x).data[:, 3:])
    close(tail2, m(changed).data[:, 3:])
    assert np.max(np.abs(tail[:, 3:] - tail2[:, 3:])) > 1e-4


def test_rollover_rebuilds_all_layers_and_positions():
    m, x = model(layers=3, batch=2, context=5)
    s = Inference(m).session()
    history = x[:, :2].copy()
    s.prefill(history)
    for step in range(19):
        token = np.full((2, 1), step % 19)
        history = np.concatenate((history, token), axis=1)
        actual = s.advance(token)
        expected = m(history[:, -5:]).data
        close(actual[:, -1], expected[:, -1])
        if step >= 3:
            close(actual, expected)
    assert s.counts["window_rebuilds"] == 16
    assert s.length == 5


def test_window_one_rebuild_and_chunk_overflow():
    m, x = model(context=1)
    s = Inference(m).session()
    s.prefill(x)
    for i in range(5):
        close(s.advance([[i]]), m(np.array([[i]])).data)
    assert s.counts["window_rebuilds"] == 5
    m, x = model(context=7)
    s = Inference(m).session()
    s.prefill(x[:, :5])
    close(
        s.advance(x[:, 3:]),
        m(np.concatenate((x[:, :5], x[:, 3:]), axis=1)[:, -7:]).data,
    )


def test_snapshot_is_separate_from_training_and_owns_its_weights():
    m, x = model()
    before = [p.data.copy() for p in m.parameters()]
    for p in m.parameters():
        p.grad = np.ones_like(p.data)
    expected = m(x).data.copy()
    inf = Inference(m)
    close(inf.forward(x), expected)
    for p, v in zip(m.parameters(), before):
        np.testing.assert_array_equal(p.data, v)
        np.testing.assert_array_equal(p.grad, np.ones_like(p.data))
        p.data[:] = 0
    close(inf.forward(x), expected)
    assert not np.allclose(m(x).data, expected)
    assert inf.weight_nbytes == sum(v.nbytes for v in before)
    for a in [
        inf._token,
        inf._position,
        *inf._final,
        inf._head,
        *(v for layer in inf._layers for v in layer),
    ]:
        assert not a.flags.writeable
        assert not any(np.shares_memory(a, p.data) for p in m.parameters())


def test_independent_sessions_input_output_ownership_and_reset():
    m, x = model(batch=2)
    inf = Inference(m)
    s = inf.session()
    other = inf.session()
    a = x[:, :4].copy()
    out = s.prefill(a)
    other.prefill(x[:, :6])
    before = state(other)
    a[:] = 0
    out[:] = 999
    close(s.append(x[:, 4:]), inf.forward(x)[:, 4:])
    assert state(other) == before
    s.reset()
    assert s.length == 0 and s.cache_nbytes == 0 and s.counts == {}
    with pytest.raises(ValueError):
        s.append([[1], [1]])
    close(s.prefill(x), inf.forward(x))


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        [[]],
        [[True]],
        [[True, 1]],
        [[1.0]],
        [["1"]],
        [[None]],
        [[-1]],
        [[19]],
        [[2**64 - 1]],
        np.array([1, 2]),
        np.zeros((1, 1, 1), dtype=int),
        np.zeros((0, 2), dtype=int),
        np.zeros((1, 14), dtype=int),
    ],
)
def test_invalid_tokens_leave_session_untouched(invalid):
    m, x = model()
    s = Inference(m).session()
    s.prefill(x[:, :3])
    before = state(s)
    for action in (s.prefill, s.append, s.advance):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            action(invalid)
        assert state(s) == before


def test_batch_change_and_overflow_are_transactional():
    m, x = model(batch=2, context=5)
    s = Inference(m).session()
    s.prefill(x[:, :4])
    before = state(s)
    for bad in ([[1]], [[1, 2], [3, 4]]):
        with pytest.raises(ValueError):
            s.append(bad)
        assert state(s) == before
    close(s.append(x[:, 4:]), m(x).data[:, 4:])


def test_counts_are_executed_matrix_shapes_not_triangular_slogans():
    m, x = model(layers=2, heads=3, width=12, batch=2, context=7)
    inf = Inference(m)
    _, full = inf.forward_with_counts(x)
    assert full == dict(
        projected_layer_rows=28, attention_score_cells=588, head_rows=14
    )
    s = inf.session()
    s.prefill(x[:, :3])
    s.append(x[:, 3:5])
    s.append(x[:, 5:])
    assert s.counts == dict(
        projected_layer_rows=28,
        attention_score_cells=396,
        head_rows=14,
        prefills=1,
        appends=2,
    )
    # Each matrix includes masked cells; this is not the number of allowed edges.
    assert s.cache_nbytes == 2 * 2 * 7 * 12 * 2 * 4
    d = s.counts
    d["appends"] = 100
    assert s.counts["appends"] == 2


def test_invalid_snapshot_parameters_and_dropout():
    m, _ = model()
    m.dropout = 0.1
    with pytest.raises(ValueError):
        Inference(m)
    m, _ = model()
    m.blocks[0].attn.dropout = 0.1
    with pytest.raises(ValueError):
        Inference(m)
    m, _ = model()
    m.blocks[0].attn.n_head = 3
    with pytest.raises(ValueError):
        Inference(m)
    m, _ = model()
    m.head.w.data[0, 0] = np.nan
    with pytest.raises(ValueError):
        Inference(m)
    m, _ = model()
    m.head.w.data = m.head.w.data.astype(np.float64)
    with pytest.raises(ValueError):
        Inference(m)
    m, _ = model()
    m.d_model = 11
    with pytest.raises(ValueError):
        Inference(m)


def test_shipped_checkpoint_every_step_through_rollover():
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "checkpoints/final.npz", allow_pickle=False) as z:
        cfg = json.loads(str(z["config"]))
        m = GPT(rng=np.random.default_rng(4), dropout=0.0, **cfg)
        for i, p in enumerate(m.parameters()):
            p.data[:] = z[f"p{i}"]
    rng = np.random.default_rng(19)
    tokens = rng.integers(0, m.vocab_size, (1, m.block_size + 9))
    inf = Inference(m)
    s = inf.session()
    s.prefill(tokens[:, :5])
    for end in range(6, tokens.shape[1] + 1):
        got = s.advance(tokens[:, end - 1 : end])[:, -1]
        expected = m(tokens[:, max(0, end - m.block_size) : end]).data[:, -1]
        # Different GEMM/GEMV shapes change float32 rounding on trained weights.
        # Keep the stricter small-model tolerance above; this is a separate
        # absolute logit bound for this frozen checkpoint, not bitwise parity.
        assert float(np.max(np.abs(got - expected))) <= 1e-5

        def probabilities(z):
            e = np.exp(z.astype(np.float64) - z.max())
            return e / e.sum()

        assert (
            float(np.abs(probabilities(got) - probabilities(expected)).sum() / 2)
            <= 3e-6
        )
    assert s.counts["window_rebuilds"] == 9
