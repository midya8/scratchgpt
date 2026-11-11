import io
import json
import zipfile

import numpy as np
import pytest

from scratchgpt.cache_benchmark import benchmark, load_legacy
from scratchgpt.model import GPT


def small():
    c = dict(vocab_size=5, block_size=4, n_layer=1, n_head=2, d_model=4)
    m = GPT(**c, rng=np.random.default_rng(0))
    return c, m


def save(path, config, parameters):
    np.savez(
        path,
        config=json.dumps(config),
        **{f"p{i}": a for i, a in enumerate(parameters)},
    )


def test_three_arm_benchmark_keeps_all_repeats_and_rollover():
    _, m = small()
    r = benchmark(m, prompt_lengths=[1, 4], steps=5, repeats=3, warmups=0)
    assert r["snapshot_weight_payload_bytes"] == sum(
        p.data.nbytes for p in m.parameters()
    )
    assert r["snapshot_setup_ns"] > 0
    assert set(r["environment"]) == {
        "python",
        "numpy",
        "os",
        "machine",
        "blas",
        "cpu_model",
        "threads",
    }
    for case in r["cases"]:
        assert set(case["arms"]) == {"original_autograd", "numpy_full", "numpy_cached"}
        assert len(case["tokens"][0]) == case["prompt_length"] + 4
        for arm in case["arms"].values():
            assert len(arm["elapsed_ns"]) == 3 and min(arm["elapsed_ns"]) > 0
            assert len(arm["last_logits"]) == 5
            assert arm["max_logit_delta"] < 2e-6
        if case["prompt_length"] == 4:
            a = case["arms"]
            assert a["numpy_cached"]["executed_counts"]["window_rebuilds"] == 4
            for key, value in a["numpy_full"]["executed_counts"].items():
                assert a["numpy_cached"]["executed_counts"][key] == value
    # Intentionally no assertion that any median factor exceeds one.


@pytest.mark.parametrize(
    "options",
    [
        dict(steps=0),
        dict(steps=True),
        dict(repeats=0),
        dict(repeats=32),
        dict(warmups=-1),
        dict(prompt_lengths=[]),
        dict(prompt_lengths=[5]),
        dict(prompt_lengths=[True]),
    ],
)
def test_invalid_benchmark_bounds(options):
    _, m = small()
    with pytest.raises(ValueError):
        benchmark(m, **options)


def test_loader_validates_and_copies_legacy_arrays(tmp_path):
    c, m = small()
    p = tmp_path / "good.npz"
    save(p, c, [v.data for v in m.parameters()])
    got, sha = load_legacy(p)
    import hashlib

    assert sha == hashlib.sha256(p.read_bytes()).hexdigest()
    x = np.array([[0, 1, 2]])
    np.testing.assert_array_equal(got(x).data, m(x).data)
    for a, b in zip(got.parameters(), m.parameters()):
        assert not np.shares_memory(a.data, b.data)


def test_loader_rejects_shapes_dtypes_objects_nonfinite_and_extra_fields(tmp_path):
    c, m = small()
    ps = [v.data.copy() for v in m.parameters()]
    p = tmp_path / "bad.npz"
    for bad in [
        np.zeros((1, 1), dtype=np.float32),
        ps[0].astype(np.float64),
        np.full(ps[0].shape, np.nan, dtype=np.float32),
        np.array([object()], dtype=object),
    ]:
        changed = list(ps)
        changed[0] = bad
        save(p, c, changed)
        with pytest.raises(ValueError):
            load_legacy(p)
    np.savez(p, config=json.dumps(c), extra=1, **{f"p{i}": a for i, a in enumerate(ps)})
    with pytest.raises(ValueError):
        load_legacy(p)
    np.savez(
        p, config=np.array([json.dumps(c)]), **{f"p{i}": a for i, a in enumerate(ps)}
    )
    with pytest.raises(ValueError):
        load_legacy(p)
    np.savez(p, config='{"vocab_size":5,"vocab_size":5}')
    with pytest.raises(ValueError):
        load_legacy(p)


def test_loader_checks_npy_headers_before_loading_arrays(tmp_path):
    c, m = small()
    good = tmp_path / "good.npz"
    save(good, c, [v.data for v in m.parameters()])
    with zipfile.ZipFile(good) as z:
        members = {n: z.read(n) for n in z.namelist()}
    fake = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        fake, dict(descr="<f4", fortran_order=False, shape=(10**10, 10**10))
    )
    members["p0.npy"] = fake.getvalue()
    bad = tmp_path / "bad.npz"
    with zipfile.ZipFile(bad, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    with pytest.raises(ValueError):
        load_legacy(bad)


def test_loader_rejects_duplicate_zip_members(tmp_path):
    c, m = small()
    good = tmp_path / "good.npz"
    save(good, c, [v.data for v in m.parameters()])
    with zipfile.ZipFile(good, "a") as z:
        with pytest.warns(UserWarning):
            z.writestr("p0.npy", b"not an array")
    with pytest.raises(ValueError):
        load_legacy(good)
