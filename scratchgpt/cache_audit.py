"""Replay every cached logit and KV coordinate against original full forwards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import (
    config_validate,
    fields,
    finite,
    integer,
    named_parameters,
    strict_json,
)
from .inference import Inference
from .model import GPT
from .tensor import gather

SCHEMA = "scratchgpt.cache.input.v1"


def fixture(seed=7, layers=2):
    c = dict(vocab_size=5, block_size=6, n_layer=layers, n_head=2, d_model=4)
    m = GPT(**c, rng=np.random.default_rng(seed))
    return dict(
        schema=SCHEMA,
        config=c,
        chunks=[2, 1, 2, 1, 1, 2],
        tokens=[[0, 1, 2, 3, 4, 0, 1, 2, 3]],
        weights=[
            dict(name=n, shape=list(p.shape), values=p.data.reshape(-1).tolist())
            for n, p in named_parameters(m)
        ],
    )


def parse(obj):
    fields(obj, ("schema", "config", "chunks", "tokens", "weights"), "input")
    if obj["schema"] != SCHEMA:
        raise ValueError("unsupported input schema")
    c = obj["config"]
    config_validate(c)
    if (
        c["d_model"] > 12
        or c["n_layer"] > 3
        or c["vocab_size"] > 16
        or c["block_size"] > 8
    ):
        raise ValueError(
            "audit bounds: width<=12, layers<=3, vocabulary<=16, context<=8"
        )
    rows = obj["tokens"]
    if (
        type(rows) is not list
        or not 1 <= len(rows) <= 2
        or any(type(r) is not list for r in rows)
    ):
        raise ValueError("tokens: expected 1-2 rows")
    width = len(rows[0])
    if not 1 <= width <= 24 or any(len(r) != width for r in rows):
        raise ValueError("tokens: expected equal row lengths in [1,24]")
    for row in rows:
        for value in row:
            integer(value, 0, c["vocab_size"] - 1, "token")
    chunks = obj["chunks"]
    if type(chunks) is not list or not 1 <= len(chunks) <= 24:
        raise ValueError("chunks: expected a nonempty list")
    for value in chunks:
        integer(value, 1, c["block_size"], "chunk length")
    if sum(chunks) != width:
        raise ValueError("chunk lengths must sum to the token count")
    m = GPT(**c, rng=np.random.default_rng(0))
    if type(obj["weights"]) is not list or len(obj["weights"]) != len(m.parameters()):
        raise ValueError("weights must cover every original named parameter")
    for row, (name, p) in zip(obj["weights"], named_parameters(m)):
        fields(row, ("name", "shape", "values"), "weight")
        if (
            row["name"] != name
            or row["shape"] != list(p.shape)
            or any(type(x) is not int for x in row["shape"])
        ):
            raise ValueError("weight name/shape/order mismatch")
        if type(row["values"]) is not list or len(row["values"]) != p.data.size:
            raise ValueError("weight length mismatch")
        for value in row["values"]:
            finite(value, -10, 10, "weight")
        p.data[:] = np.asarray(row["values"], dtype=np.float32).reshape(p.shape)
    return m, np.asarray(rows, dtype=np.int64)


def original_trace(model, tokens):
    """Original Tensor/Block arithmetic; diagnostic duplicate projections not timed."""
    batch, length = tokens.shape
    x = gather(model.tok_emb.w, tokens) + gather(model.pos_emb.w, np.arange(length))
    kv = []
    for block in model.blocks:
        qkv = (
            block.attn.qkv(block.ln1(x))
            .data.reshape(batch, length, 3, model.n_head, model.d_model // model.n_head)
            .transpose(2, 0, 3, 1, 4)
        )
        kv.append((qkv[1].copy(), qkv[2].copy()))
        x = block(x, model.mask(length))
    return kv


def _max_delta(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def audit(obj):
    m, tokens = parse(obj)
    inf = Inference(m)
    session = inf.session()
    stale = inf.session()
    steps = []
    consumed = 0
    full_counts = dict(projected_layer_rows=0, attention_score_cells=0, head_rows=0)
    for chunk in obj["chunks"]:
        old = session.length
        consumed += chunk
        mode = (
            "prefill"
            if old == 0
            else ("rebuild" if old + chunk > m.block_size else "append")
        )
        values = session.advance(tokens[:, consumed - chunk : consumed])
        context = tokens[:, max(0, consumed - m.block_size) : consumed]
        reference = m(context).data
        # Deliberately wrong control: dropping old KV slots does not rebase
        # learned positions or remove their influence on later-layer states.
        if stale.length + chunk > m.block_size:
            remove = stale.length + chunk - m.block_size
            stale._past = tuple(
                (k[:, :, remove:].copy(), v[:, :, remove:].copy())
                for k, v in stale._past
            )
            stale._tokens = stale._tokens[:, remove:].copy()
        wrong = (
            stale.prefill(tokens[:, :chunk])
            if old == 0
            else stale.append(tokens[:, consumed - chunk : consumed])
        )
        native_full, counts = inf.forward_with_counts(context)
        for key, value in counts.items():
            full_counts[key] += value
        expected = reference[:, -values.shape[1] :]
        reference_kv = original_trace(m, context)
        kv_delta = max(
            _max_delta(a, b)
            for pair, ref in zip(session._past, reference_kv)
            for a, b in zip(pair, ref)
        )
        queries = list(range(context.shape[1] - values.shape[1], context.shape[1]))
        mask = [[int(k <= q) for k in range(context.shape[1])] for q in queries]
        steps.append(
            dict(
                mode=mode,
                consumed=consumed,
                window_start=consumed - context.shape[1],
                context=context.tolist(),
                positions=queries,
                causal_mask=mask,
                cached_logits=values.tolist(),
                original_logits=expected.tolist(),
                numpy_full_logits=native_full[:, -values.shape[1] :].tolist(),
                cache=[
                    dict(key=a.tolist(), value=b.tolist(), shape=list(a.shape))
                    for a, b in session._past
                ],
                original_cache=[
                    dict(key=a.tolist(), value=b.tolist(), shape=list(a.shape))
                    for a, b in reference_kv
                ],
                cache_nbytes=session.cache_nbytes,
                cumulative_counts=session.counts,
                max_logit_delta=_max_delta(values, expected),
                max_kv_delta=kv_delta,
                numpy_full_delta=_max_delta(native_full, reference),
                stale_shift_last_logits=wrong[:, -1].tolist(),
                stale_shift_last_delta=_max_delta(wrong[:, -1], reference[:, -1]),
            )
        )
    return dict(
        schema="scratchgpt.cache.report.v1",
        input=obj,
        steps=steps,
        comparison=dict(
            max_logit_delta=max(s["max_logit_delta"] for s in steps),
            max_kv_delta=max(s["max_kv_delta"] for s in steps),
            numpy_full_delta=max(s["numpy_full_delta"] for s in steps),
            stale_shift_max_delta=max(s["stale_shift_last_delta"] for s in steps),
        ),
        cached_counts=session.counts,
        numpy_full_counts=full_counts,
        weight_snapshot_nbytes=inf.weight_nbytes,
        scope="Float32 inference on copied original-GPT weights, no training or checkpoint changes. "
        "Every context crop rebuilds all layers at rebased absolute positions. "
        "KV bytes count resident ndarray payload, not peak memory/RSS, weights or temporaries. "
        "Matrix rows/score cells are not FLOPs or latency. Float rounding can change sampled outcomes.",
    )


def parse_json(data):
    if len(data) > 2_000_000:
        raise ValueError("input exceeds two million bytes")
    text = data.decode("utf-8")
    depth = 0
    quoted = False
    escaped = False
    for c in text:
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c in "[{":
            depth += 1
            if depth > 48:
                raise ValueError("input nesting exceeds 48 containers")
        elif c in "]}":
            depth -= 1
    return strict_json(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?")
    ap.add_argument("--fixture", action="store_true")
    args = ap.parse_args()
    try:
        if args.fixture:
            if args.input:
                raise ValueError("use --fixture without an input file")
            result = fixture()
        else:
            if not args.input:
                raise ValueError("provide input JSON or --fixture")
            p = Path(args.input)
            if p.stat().st_size > 2_000_000:
                raise ValueError("input exceeds two million bytes")
            result = audit(parse_json(p.read_bytes()))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        OverflowError,
        UnicodeError,
    ) as exc:
        ap.exit(2, f"cache audit error: {exc}\n")


if __name__ == "__main__":
    main()
