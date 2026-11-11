"""Complete small cached-inference witnesses. Timings are deliberately separate."""

import argparse
import json
from pathlib import Path

import numpy as np

from scratchgpt.cache_audit import audit, fixture
from scratchgpt.checkpoint import named_parameters
from scratchgpt.model import GPT
from scratchgpt.optim import Adam
from scratchgpt.tensor import cross_entropy

ROOT = Path(__file__).resolve().parents[1]


def build():
    reports = []
    for seed in range(8):
        obj = fixture(seed=seed + 7, layers=2 if seed == 0 else 1 + seed % 3)
        if seed == 1:
            obj["chunks"] = [1] * 9
        if seed == 2:
            obj["chunks"] = [6, 3]
        if seed == 3:
            obj["config"]["block_size"] = 1
            obj["chunks"] = [1] * 5
            obj["tokens"] = [[0, 1, 2, 3, 4]]
            obj["weights"][1]["shape"] = [1, 4]
            obj["weights"][1]["values"] = obj["weights"][1]["values"][:4]
        if seed == 4:
            obj["tokens"].append([1, 2, 3, 4, 0, 1, 2, 3, 4])
        if seed == 5:
            obj["tokens"] = [[0, 1, 2, 3, 4, 0]]
            obj["chunks"] = [2, 2, 2]
        if seed in (6, 7):
            c = obj["config"]
            m = GPT(**c, rng=np.random.default_rng(seed + 7))
            opt = Adam(m.parameters(), lr=0.005)
            x = np.array([[0, 1, 2, 3, 4]])
            y = np.array([[1, 2, 3, 4, 0]])
            for _ in range(3):
                opt.zero_grad()
                cross_entropy(m(x), y).backward()
                opt.step()
            obj["weights"] = [
                dict(name=n, shape=list(p.shape), values=p.data.reshape(-1).tolist())
                for n, p in named_parameters(m)
            ]
        r = audit(obj)
        r["fixture"] = dict(
            seed=seed + 7, original_eager_training_steps=3 if seed in (6, 7) else 0
        )
        if (
            r["comparison"]["max_logit_delta"] > 2e-6
            or r["comparison"]["max_kv_delta"] > 2e-6
        ):
            raise AssertionError("native small-fixture parity")
        reports.append(r)
    return dict(
        schema="scratchgpt.cache.witnesses.v1",
        reports=reports,
        count=len(reports),
        steps=sum(len(r["steps"]) for r in reports),
        logit_coordinates=sum(
            np.asarray(s["cached_logits"]).size for r in reports for s in r["steps"]
        ),
        kv_coordinates=sum(
            np.asarray(c["key"]).size + np.asarray(c["value"]).size
            for r in reports
            for s in r["steps"]
            for c in s["cache"]
        ),
    )


def compare(a, b):
    if type(a) in (int, float) and type(b) in (int, float):
        if type(a) is int:
            assert type(b) is int and a == b
        else:
            assert type(b) is float and abs(a - b) <= 2e-6 * max(1, abs(a), abs(b))
    elif isinstance(a, dict):
        assert type(b) is dict and set(a) == set(b)
        for k in a:
            compare(a[k], b[k])
    elif isinstance(a, list):
        assert type(b) is list and len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    else:
        assert type(a) is type(b) and a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    obj = build()
    p = ROOT / "docs/cache-witnesses.json"
    if args.check:
        compare(obj, json.loads(p.read_text()))
    else:
        p.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in obj.items() if k != "reports"}))


if __name__ == "__main__":
    main()
