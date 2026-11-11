import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from scratchgpt.cache_audit import audit, fixture, parse, parse_json


def test_every_report_coordinate_and_stale_shift_counterexample():
    r = audit(fixture())
    assert len(r["steps"]) == 6
    assert r["comparison"]["numpy_full_delta"] == 0
    assert r["comparison"]["max_logit_delta"] < 2e-6
    assert r["comparison"]["max_kv_delta"] < 2e-6
    assert r["comparison"]["stale_shift_max_delta"] > 1e-5
    assert [s["mode"] for s in r["steps"]] == [
        "prefill",
        "append",
        "append",
        "append",
        "rebuild",
        "rebuild",
    ]
    for s in r["steps"]:
        np.testing.assert_allclose(
            s["cached_logits"], s["original_logits"], atol=2e-6, rtol=2e-5
        )
        for cache, reference in zip(s["cache"], s["original_cache"]):
            assert cache["shape"] == reference["shape"]
            for key in ("key", "value"):
                np.testing.assert_allclose(
                    cache[key], reference[key], atol=2e-6, rtol=2e-5
                )
        for pos, row in zip(s["positions"], s["causal_mask"]):
            assert row == [int(k <= pos) for k in range(len(row))]
    assert r["cached_counts"] == dict(
        appends=3,
        attention_score_cells=380,
        head_rows=18,
        prefills=3,
        projected_layer_rows=36,
        window_rebuilds=2,
    )
    assert r["numpy_full_counts"] == dict(
        attention_score_cells=584, head_rows=28, projected_layer_rows=56
    )


def test_all_32_short_stream_partitions():
    for mask in range(32):
        obj = fixture()
        obj["tokens"] = [[0, 1, 2, 3, 4, 0]]
        obj["chunks"] = []
        last = 0
        for cut in range(1, 6):
            if mask >> (cut - 1) & 1:
                obj["chunks"].append(cut - last)
                last = cut
        obj["chunks"].append(6 - last)
        r = audit(obj)
        assert r["comparison"]["max_logit_delta"] < 2e-6
        assert r["comparison"]["max_kv_delta"] < 2e-6
        assert r["cached_counts"].get("window_rebuilds", 0) == 0


def test_30_seeded_weight_and_layer_controls():
    for seed in range(30):
        obj = fixture(seed=seed, layers=1 + seed % 3)
        r = audit(obj)
        assert r["comparison"]["max_logit_delta"] < 2e-6
        assert r["comparison"]["max_kv_delta"] < 2e-6


def test_audit_does_not_mutate_input_or_global_rng():
    obj = fixture()
    original = copy.deepcopy(obj)
    before = np.random.get_state()
    a = audit(obj)
    b = audit(obj)
    assert a == b and obj == original
    after = np.random.get_state()
    assert (
        before[0] == after[0]
        and np.array_equal(before[1], after[1])
        and before[2:] == after[2:]
    )


def test_strict_json_transport():
    obj = fixture()
    s = json.dumps(obj)
    assert parse_json(s.encode()) == obj
    for bad in [
        b'{"a":1,"a":2}',
        b'{"a":1,"\\u0061":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1} trailing',
        b"[ " * 49 + b"0" + b"] " * 49,
        b" " * 2_000_001,
        b"\xff",
    ]:
        with pytest.raises((ValueError, UnicodeError)):
            parse_json(bad)
    assert parse_json(b'{"a":"[{}]\\""}') == {"a": '[{}]"'}


def test_invalid_schema_shapes_types_weights_and_chunks():
    original = fixture()
    cases = []
    for key, value in [
        ("schema", "other"),
        ("chunks", [2, True, 6]),
        ("chunks", [9]),
        ("chunks", [1]),
        ("tokens", [[0, 1.0]]),
        ("tokens", [[0], [1, 2]]),
        ("tokens", [[5]]),
        ("tokens", []),
        ("weights", []),
    ]:
        obj = copy.deepcopy(original)
        obj[key] = value
        cases.append(obj)
    obj = copy.deepcopy(original)
    obj["unknown"] = 1
    cases.append(obj)
    obj = copy.deepcopy(original)
    obj["weights"][0]["name"] = "head.w"
    cases.append(obj)
    obj = copy.deepcopy(original)
    obj["weights"][0]["shape"][0] = True
    cases.append(obj)
    obj = copy.deepcopy(original)
    obj["weights"][0]["values"][0] = float("inf")
    cases.append(obj)
    obj = copy.deepcopy(original)
    obj["weights"][0]["values"][0] = True
    cases.append(obj)
    obj = copy.deepcopy(original)
    obj["config"]["n_layer"] = 4
    cases.append(obj)
    for bad in cases:
        with pytest.raises(ValueError):
            parse(bad)


def test_cli_exact_roundtrip_and_error_stdout(tmp_path):
    command = [sys.executable, "-m", "scratchgpt.cache_audit"]
    obj = json.loads(subprocess.check_output(command + ["--fixture"]))
    p = tmp_path / "input.json"
    p.write_text(json.dumps(obj))
    assert json.loads(subprocess.check_output(command + [str(p)])) == audit(obj)
    p.write_text('{"schema":0,"schema":1}')
    r = subprocess.run(command + [str(p)], capture_output=True)
    assert r.returncode == 2 and r.stdout == b"" and b"duplicate JSON key" in r.stderr
    p.write_text(" " * 2_000_001)
    r = subprocess.run(command + [str(p)], capture_output=True)
    assert r.returncode == 2 and r.stdout == b""
