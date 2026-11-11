"""Inference-only float32 snapshots and append-only KV sessions.

Training and the original GPT forward stay unchanged. A snapshot owns copied
weights: later training does not update it. Context-window rebasing requires a
complete rebuild, not removal of one cached key/value pair.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .model import GPT
from .nn import Block, CausalSelfAttention, Embedding, LayerNorm, Linear, MLP


def _ln(x, w, b):
    xc = x - x.mean(axis=-1, keepdims=True)
    variance = (xc * xc).mean(axis=-1, keepdims=True)
    inv = (variance + np.float32(1e-5)) ** np.float32(-0.5)
    return (xc * inv) * w + b


def _gelu(x):
    inner = (x + np.float32(0.044715) * x ** np.float32(3)) * np.float32(
        np.sqrt(2 / np.pi)
    )
    return (x * np.float32(0.5)) * (np.tanh(inner) + np.float32(1))


def _copy(t, shape):
    a = np.array(t.data, copy=True)
    if a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(
            "snapshot requires finite float32 parameters with original shapes"
        )
    a.flags.writeable = False
    return a


class Inference:
    """Copied original-GPT weights, no autograd graph or mutable model reference."""

    def __init__(self, model):
        if (
            type(model) is not GPT
            or model.dropout
            or type(model.tok_emb) is not Embedding
            or type(model.pos_emb) is not Embedding
            or type(model.ln_f) is not LayerNorm
            or type(model.head) is not Linear
            or any(type(b) is not Block for b in model.blocks)
        ):
            raise ValueError("snapshot requires an original GPT with dropout=0")
        names = ("vocab_size", "block_size", "n_layer", "n_head", "d_model")
        for name in names:
            value = getattr(model, name)
            if type(value) is not int or value < 1:
                raise ValueError("model dimensions must be positive integers")
            setattr(self, name, value)
        v, t, d = self.vocab_size, self.block_size, self.d_model
        if d % self.n_head or len(model.blocks) != self.n_layer:
            raise ValueError("inconsistent attention or block dimensions")
        self._token = _copy(model.tok_emb.w, (v, d))
        self._position = _copy(model.pos_emb.w, (t, d))
        self._final = (_copy(model.ln_f.w, (d,)), _copy(model.ln_f.b, (d,)))
        self._head = _copy(model.head.w, (d, v))
        layers = []
        for b in model.blocks:
            if (
                type(b.attn) is not CausalSelfAttention
                or type(b.mlp) is not MLP
                or type(b.ln1) is not LayerNorm
                or type(b.ln2) is not LayerNorm
                or any(
                    type(a) is not Linear
                    for a in [b.attn.qkv, b.attn.proj, b.mlp.fc, b.mlp.proj]
                )
                or b.attn.n_head != self.n_head
                or b.attn.dropout
                or b.mlp.dropout
            ):
                raise ValueError("snapshot requires consistent heads and dropout=0")
            layers.append(
                tuple(
                    _copy(p, shape)
                    for p, shape in [
                        (b.ln1.w, (d,)),
                        (b.ln1.b, (d,)),
                        (b.attn.qkv.w, (d, 3 * d)),
                        (b.attn.proj.w, (d, d)),
                        (b.ln2.w, (d,)),
                        (b.ln2.b, (d,)),
                        (b.mlp.fc.w, (d, 4 * d)),
                        (b.mlp.proj.w, (4 * d, d)),
                    ]
                )
            )
        self._layers = tuple(layers)
        self.weight_nbytes = sum(
            a.nbytes
            for a in [
                self._token,
                self._position,
                *self._final,
                self._head,
                *(a for layer in layers for a in layer),
            ]
        )

    def _input(self, tokens):
        if isinstance(tokens, (list, tuple)) and any(
            isinstance(value, (bool, np.bool_))
            for row in tokens
            if isinstance(row, (list, tuple))
            for value in row
        ):
            raise ValueError("boolean token IDs are not supported")
        a = np.asarray(tokens)
        if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1 or a.dtype.kind not in "iu":
            raise ValueError("tokens must be a nonempty integer batch [batch,time]")
        if (
            a.shape[1] > self.block_size
            or (a >= self.vocab_size).any()
            or (a < 0).any()
        ):
            raise ValueError("token IDs or sequence length outside model bounds")
        return np.array(a, dtype=np.int64, copy=True)

    def _run(self, tokens, past=(), *, retain=False):
        batch, length = tokens.shape
        offset = past[0][0].shape[2] if past else 0
        total = offset + length
        if total > self.block_size:
            raise ValueError(
                "cache overflow: rebuild with rebased context instead of shifting KV"
            )
        if past and past[0][0].shape[0] != batch:
            raise ValueError("cache batch size cannot change")
        x = self._token[tokens] + self._position[np.arange(offset, total)]
        heads, width = self.n_head, self.d_model // self.n_head
        blocked = np.arange(total)[None, :] > np.arange(offset, total)[:, None]
        mask = np.where(blocked, np.float32(-1e9), np.float32(0))[None, None]
        fresh = []
        for i, (w1, b1, qkvw, proj, w2, b2, fc, out) in enumerate(self._layers):
            qkv = (
                (_ln(x, w1, b1) @ qkvw)
                .reshape(batch, length, 3, heads, width)
                .transpose(2, 0, 3, 1, 4)
            )
            q, k, v = qkv
            if past:
                k = np.concatenate((past[i][0], k), axis=2)
                v = np.concatenate((past[i][1], v), axis=2)
            elif retain:
                # A prefill view otherwise retains the Q portion of QKV storage.
                k, v = k.copy(), v.copy()
            scores = (q @ k.transpose(0, 1, 3, 2)) * np.float32(width**-0.5) + mask
            e = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attention = e / e.sum(axis=-1, keepdims=True)
            y = (
                (attention @ v)
                .transpose(0, 2, 1, 3)
                .reshape(batch, length, self.d_model)
            )
            x = x + y @ proj
            x = x + _gelu(_ln(x, w2, b2) @ fc) @ out
            if retain:
                fresh.append((k, v))
        logits = _ln(x, *self._final) @ self._head
        if not np.isfinite(logits).all() or any(
            not np.isfinite(a).all() for pair in fresh for a in pair
        ):
            raise ValueError("nonfinite inference result; session was not updated")
        counts = dict(
            projected_layer_rows=batch * length * self.n_layer,
            attention_score_cells=batch * heads * length * total * self.n_layer,
            head_rows=batch * length,
        )
        return logits, tuple(fresh), counts

    def forward(self, tokens):
        """Full causal NumPy reference with the same arithmetic as cached runs."""
        return self.forward_with_counts(tokens)[0]

    def forward_with_counts(self, tokens):
        """Full forward plus executed matrix-row/score-cell counts, not FLOPs."""
        logits, _, counts = self._run(self._input(tokens))
        return logits, counts

    def session(self):
        return Session(self)


class Session:
    """One independent sequence batch. Invalid calls leave prior state intact."""

    def __init__(self, snapshot):
        self._snapshot = snapshot
        self._past = ()
        self._tokens = None
        self._counts = Counter()

    @property
    def length(self):
        return 0 if self._tokens is None else self._tokens.shape[1]

    @property
    def cache_nbytes(self):
        """Resident KV ndarray payload only, not weights, peak memory or RSS."""
        return sum(a.nbytes for pair in self._past for a in pair)

    @property
    def counts(self):
        return dict(self._counts)

    def reset(self):
        self._past, self._tokens = (), None
        self._counts.clear()

    def prefill(self, tokens):
        a = self._snapshot._input(tokens)
        logits, fresh, counts = self._snapshot._run(a, retain=True)
        self._past, self._tokens = fresh, a
        self._counts.update(counts)
        self._counts["prefills"] += 1
        return logits

    def append(self, tokens):
        if not self._past:
            raise ValueError("prefill the session before appending")
        a = self._snapshot._input(tokens)
        logits, fresh, counts = self._snapshot._run(a, self._past, retain=True)
        joined = np.concatenate((self._tokens, a), axis=1)
        self._past, self._tokens = fresh, joined
        self._counts.update(counts)
        self._counts["appends"] += 1
        return logits

    def advance(self, tokens):
        """Append a chunk, or rebuild its final window when absolute positions shift.

        Returns logits for the entire final window on a rebuild, and only the
        newly supplied chunk otherwise. Use [:,-1] for next-token generation.
        """
        a = self._snapshot._input(tokens)
        if self._tokens is None:
            return self.prefill(a)
        if a.shape[0] != self._tokens.shape[0]:
            raise ValueError("cache batch size cannot change")
        if self.length + a.shape[1] <= self._snapshot.block_size:
            return self.append(a)
        joined = np.concatenate((self._tokens, a), axis=1)[
            :, -self._snapshot.block_size :
        ]
        result = self.prefill(joined)
        self._counts["window_rebuilds"] += 1
        return result
