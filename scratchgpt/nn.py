"""Minimal neural-net layers built on the autograd engine."""
import numpy as np

from .tensor import Tensor, dropout, gather, gelu, layer_norm, softmax


def _collect(v):
    if isinstance(v, Tensor):
        return [v]
    if isinstance(v, Module):
        return v.parameters()
    if isinstance(v, (list, tuple)):
        out = []
        for u in v:
            out.extend(_collect(u))
        return out
    return []


class Module:
    def parameters(self):
        ps = []
        for v in list(self.__dict__.values()):
            ps.extend(_collect(v))
        return ps


class Linear(Module):
    def __init__(self, fan_in, fan_out, rng, scale=1.0):
        self.w = Tensor(rng.normal(0.0, 0.02 * scale, (fan_in, fan_out)))

    def __call__(self, x):
        return x @ self.w


class Embedding(Module):
    def __init__(self, num, dim, rng):
        self.w = Tensor(rng.normal(0.0, 0.02, (num, dim)))

    def __call__(self, idx):
        return gather(self.w, idx)


class LayerNorm(Module):
    def __init__(self, dim, rng):
        self.w = Tensor(np.ones(dim))
        self.b = Tensor(np.zeros(dim))

    def __call__(self, x):
        return layer_norm(x, self.w, self.b)


class CausalSelfAttention(Module):
    def __init__(self, d_model, n_head, rng, dropout=0.0, scale=1.0):
        self.n_head = n_head
        self.dropout = dropout
        self.qkv = Linear(d_model, 3 * d_model, rng)
        self.proj = Linear(d_model, d_model, rng, scale=scale)

    def __call__(self, x, mask, rng=None):
        B, T, D = x.shape
        H = self.n_head
        dh = D // H
        qkv = self.qkv(x).view((B, T, 3, H, dh)).permute((2, 0, 3, 1, 4))
        q, k, v = qkv.select(0), qkv.select(1), qkv.select(2)
        att = (q @ k.permute((0, 1, 3, 2))) * (dh ** -0.5) + mask
        att = softmax(att, -1)
        if self.dropout and rng is not None:
            att = dropout(att, self.dropout, rng)
        y = (att @ v).permute((0, 2, 1, 3)).view((B, T, D))
        return self.proj(y)


class MLP(Module):
    def __init__(self, d_model, rng, dropout=0.0, scale=1.0):
        self.fc = Linear(d_model, 4 * d_model, rng)
        self.proj = Linear(4 * d_model, d_model, rng, scale=scale)
        self.dropout = dropout

    def __call__(self, x, rng=None):
        h = gelu(self.fc(x))
        if self.dropout and rng is not None:
            h = dropout(h, self.dropout, rng)
        return self.proj(h)


class Block(Module):
    def __init__(self, d_model, n_head, rng, dropout=0.0, scale=1.0):
        self.ln1 = LayerNorm(d_model, rng)
        self.attn = CausalSelfAttention(d_model, n_head, rng, dropout, scale)
        self.ln2 = LayerNorm(d_model, rng)
        self.mlp = MLP(d_model, rng, dropout, scale)

    def __call__(self, x, mask, rng=None):
        x = x + self.attn(self.ln1(x), mask, rng)
        x = x + self.mlp(self.ln2(x), rng)
        return x
