"""Teacher-forced three-arm cache timings. Run with BLAS threads fixed externally.

No speed threshold is a test. All repeats and the rollover regime are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np

from .checkpoint import config_validate, strict_json
from .inference import Inference
from .model import GPT


def benchmark_environment():
    # Do not publish compiler paths or unrelated machine configuration.
    build = getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {})
    blas = build.get("blas", {})
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        os=platform.system(),
        machine=platform.machine(),
        blas={k: blas.get(k) for k in ("name", "version")},
        cpu_model="not recorded",
        threads={
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    )


def load_legacy(path):
    """Read the old config/pN inference format, not a continuation checkpoint."""
    path = Path(path)
    if path.stat().st_size > 100_000_000:
        raise ValueError("checkpoint exceeds 100 MB")
    data = path.read_bytes()
    if len(data) > 100_000_000:
        raise ValueError("checkpoint exceeds 100 MB")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [x.filename for x in archive.infolist()]
        if len(names) > 100 or len(set(names)) != len(names):
            raise ValueError("invalid checkpoint members")
        if sum(x.file_size for x in archive.infolist()) > 100_000_000:
            raise ValueError("expanded checkpoint exceeds 100 MB")

        def header(archive, name):
            with archive.open(name) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError("unsupported NPY header")
                return shape, order, dtype, f.tell(), archive.getinfo(name).file_size

        shape, order, dtype, start, size = header(archive, "config.npy")
        if (
            shape != ()
            or dtype.kind != "U"
            or dtype.itemsize > 40_000
            or size != start + dtype.itemsize
        ):
            raise ValueError("invalid legacy config array header")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        config = z["config"]
        if config.shape != () or config.dtype.kind != "U" or len(str(config)) > 10_000:
            raise ValueError("invalid legacy config")
        c = strict_json(str(config))
        config_validate(c)
        m = GPT(**c, rng=np.random.default_rng(0))
        expected = ["config"] + [f"p{i}" for i in range(len(m.parameters()))]
        if set(z.files) != set(expected) or set(names) != {
            x + ".npy" for x in expected
        }:
            raise ValueError("unexpected legacy checkpoint fields")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for i, p in enumerate(m.parameters()):
                shape, order, dtype, start, size = header(archive, f"p{i}.npy")
                if (
                    shape != p.shape
                    or dtype != np.float32
                    or size != start + p.data.nbytes
                ):
                    raise ValueError("invalid legacy parameter header")
        for i, p in enumerate(m.parameters()):
            a = z[f"p{i}"]
            if a.dtype != np.float32 or a.shape != p.shape or not np.isfinite(a).all():
                raise ValueError("invalid legacy parameter shape, dtype or values")
            p.data[:] = a
    return m, hashlib.sha256(data).hexdigest()


def benchmark(
    model, *, prompt_lengths=(4, 16, 48, 64), steps=48, repeats=7, warmups=2, seed=19
):
    if (
        type(steps) is not int
        or not 1 <= steps <= 256
        or type(repeats) is not int
        or not 1 <= repeats <= 31
    ):
        raise ValueError("steps in [1,256], repeats in [1,31]")
    if type(warmups) is not int or not 0 <= warmups <= 10:
        raise ValueError("warmups in [0,10]")
    if not prompt_lengths or any(
        type(p) is not int or not 1 <= p <= model.block_size for p in prompt_lengths
    ):
        raise ValueError("prompt lengths must fit the model context")
    setup = time.perf_counter_ns()
    inf = Inference(model)
    setup = time.perf_counter_ns() - setup
    rng = np.random.default_rng(seed)
    reports = []
    for prompt in prompt_lengths:
        tokens = rng.integers(0, model.vocab_size, (1, prompt + steps - 1))

        def execute(arm):
            session = inf.session()
            out = []
            counts = {}
            for end in range(prompt, prompt + steps):
                context = tokens[:, max(0, end - model.block_size) : end]
                if arm == "original_autograd":
                    value = model(context).data
                elif arm == "numpy_full":
                    value, c = inf.forward_with_counts(context)
                    for k, v in c.items():
                        counts[k] = counts.get(k, 0) + v
                else:
                    value = (
                        session.prefill(context)
                        if end == prompt
                        else session.advance(tokens[:, end - 1 : end])
                    )
                out.append(value[:, -1].copy())
            return (
                np.stack(out),
                session.counts if arm == "numpy_cached" else counts,
                session.cache_nbytes,
            )

        arms = ("original_autograd", "numpy_full", "numpy_cached")
        for _ in range(warmups):
            for arm in arms:
                execute(arm)
        samples = {arm: [] for arm in arms}
        first = {}
        for repeat in range(repeats):
            # Rotate order; retain every timing, with no best-run selection.
            for arm in arms[repeat % 3 :] + arms[: repeat % 3]:
                start = time.perf_counter_ns()
                value, counts, nbytes = execute(arm)
                elapsed = time.perf_counter_ns() - start
                samples[arm].append(elapsed)
                if arm not in first:
                    first[arm] = (value, counts, nbytes)
                else:
                    np.testing.assert_array_equal(value, first[arm][0])
        ref = first["original_autograd"][0]
        rows = {}
        for arm in arms:
            values, counts, nbytes = first[arm]
            rows[arm] = dict(
                elapsed_ns=samples[arm],
                median_ns=statistics.median(samples[arm]),
                last_logits=values.tolist(),
                executed_counts=counts,
                final_kv_payload_bytes=nbytes,
                max_logit_delta=float(np.max(np.abs(values.astype(np.float64) - ref))),
            )
        reports.append(
            dict(
                prompt_length=prompt,
                steps=steps,
                tokens=tokens.tolist(),
                arms=rows,
                cache_only_median_factor=rows["numpy_full"]["median_ns"]
                / rows["numpy_cached"]["median_ns"],
                scope="Compare numpy_full with numpy_cached to isolate this cache. "
                "Original-autograd comparison also removes graph construction. All arms use the same teacher-forced tokens.",
            )
        )
    return dict(
        schema="scratchgpt.cache.benchmark.v1",
        config={
            k: getattr(model, k)
            for k in ("vocab_size", "block_size", "n_layer", "n_head", "d_model")
        },
        environment=benchmark_environment(),
        seed=seed,
        repeats=repeats,
        warmups=warmups,
        snapshot_setup_ns=setup,
        snapshot_weight_payload_bytes=inf.weight_nbytes,
        cases=reports,
        limits="Local CPU wall timings include Python/validation/output-copy overhead, not model loading or snapshot setup. "
        "Teacher forcing is not sampled generation. Timings are descriptive, never a speed gate. "
        "Every sliding-window step rebuilds; no shortened-context tradeoff. KV payload excludes weights, token IDs, "
        "temporaries, allocator/Python/BLAS memory and process RSS. Counters are matrix shapes, not FLOPs. "
        "Float32 logit closeness does not guarantee identical sampling at every RNG threshold.",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--prompts", type=int, nargs="+", default=[4, 16, 48, 64])
    args = ap.parse_args()
    try:
        m, sha = load_legacy(args.checkpoint)
        result = benchmark(
            m,
            prompt_lengths=args.prompts,
            steps=args.steps,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        result["checkpoint_sha256"] = sha
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        OverflowError,
        zipfile.BadZipFile,
    ) as exc:
        ap.exit(2, f"cache benchmark error: {exc}\n")


if __name__ == "__main__":
    main()
