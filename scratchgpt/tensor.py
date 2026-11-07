"""Pure-NumPy reverse-mode autograd.

No PyTorch, no JAX, no micrograd trickery: every op below carries its own
hand-written backward rule, and the test suite checks each one against
finite differences.
"""
from __future__ import annotations

import numpy as np


def _noop(node):
    pass


def _unreduce(g, shape):
    g = np.asarray(g)
    if g.shape == tuple(shape):
        return g
    extra = g.ndim - len(shape)
    if extra > 0:
        g = g.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1)
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    return g


def _acc(t, g):
    g = _unreduce(np.asarray(g, dtype=np.float32), t.data.shape)
    if g.shape != t.data.shape:
        g = np.broadcast_to(g, t.data.shape)
    if t.grad is None:
        t.grad = g
    else:
        t.grad = t.grad + g


def const(x):
    return x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=np.float32))


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op")
    __array_ufunc__ = None  # keep NumPy ufuncs from silently coercing Tensors

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = None
        self._backward = _noop
        self._prev = tuple(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def item(self):
        return float(self.data)

    def numpy(self):
        return self.data

    def __repr__(self):
        return f"Tensor(op={self._op!r}, shape={self.data.shape})"

    def backward(self):
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for c in t._prev:
                build(c)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            if t.grad is not None:
                t._backward(t)

    # ---- arithmetic -------------------------------------------------
    def __add__(self, other):
        other = const(other)
        out = Tensor(self.data + other.data, (self, other), "add")

        def _bw(node):
            _acc(self, node.grad)
            _acc(other, node.grad)

        out._backward = _bw
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = const(other)
        out = Tensor(self.data * other.data, (self, other), "mul")

        def _bw(node):
            _acc(self, node.grad * other.data)
            _acc(other, node.grad * self.data)

        out._backward = _bw
        return out

    __rmul__ = __mul__

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _bw(node):
            _acc(self, -node.grad)

        out._backward = _bw
        return out

    def __sub__(self, other):
        return self + (-const(other))

    def __rsub__(self, other):
        return const(other) + (-self)

    def __truediv__(self, other):
        other = const(other)
        out = Tensor(self.data / other.data, (self, other), "div")

        def _bw(node):
            g = node.grad
            _acc(self, g / other.data)
            _acc(other, -g * self.data / (other.data * other.data))

        out._backward = _bw
        return out

    def __matmul__(self, other):
        other = const(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other), "matmul")

        def _bw(node):
            g = node.grad
            _acc(self, np.matmul(g, np.swapaxes(other.data, -1, -2)))
            _acc(other, np.matmul(np.swapaxes(self.data, -1, -2), g))

        out._backward = _bw
        return out

    def __rmatmul__(self, other):
        return const(other) @ self

    # ---- elementwise ------------------------------------------------
    def pow(self, n):
        out = Tensor(self.data ** n, (self,), "pow")

        def _bw(node):
            _acc(self, node.grad * n * self.data ** (n - 1))

        out._backward = _bw
        return out

    __pow__ = pow

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _bw(node):
            _acc(self, node.grad * node.data)

        out._backward = _bw
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _bw(node):
            _acc(self, node.grad / self.data)

        out._backward = _bw
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data), (self,), "sqrt")

        def _bw(node):
            _acc(self, node.grad / (2.0 * node.data))

        out._backward = _bw
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _bw(node):
            _acc(self, node.grad * (1.0 - t * t))

        out._backward = _bw
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _bw(node):
            _acc(self, node.grad * (self.data > 0))

        out._backward = _bw
        return out

    def sigmoid(self):
        x = self.data
        s = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
        out = Tensor(s, (self,), "sigmoid")

        def _bw(node):
            _acc(self, node.grad * s * (1.0 - s))

        out._backward = _bw
        return out

    # ---- reductions and shapes --------------------------------------
    def _reduced_axes(self, axis):
        if axis is None:
            return None
        ax = axis if isinstance(axis, tuple) else (axis,)
        return tuple(sorted(a % self.data.ndim for a in ax))

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _bw(node):
            g = node.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, self._reduced_axes(axis))
            _acc(self, np.broadcast_to(g, self.data.shape))

        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        ax = self._reduced_axes(axis)
        n = self.data.size if ax is None else int(np.prod([self.data.shape[a] for a in ax]))
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")

        def _bw(node):
            g = node.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, ax)
            _acc(self, np.broadcast_to(g, self.data.shape) / n)

        out._backward = _bw
        return out

    def view(self, shape):
        old = self.data.shape
        out = Tensor(self.data.reshape(shape), (self,), "view")

        def _bw(node):
            _acc(self, node.grad.reshape(old))

        out._backward = _bw
        return out

    def permute(self, axes):
        axes = tuple(axes)
        inv = tuple(int(i) for i in np.argsort(axes))
        out = Tensor(np.transpose(self.data, axes), (self,), "permute")

        def _bw(node):
            _acc(self, np.transpose(node.grad, inv))

        out._backward = _bw
        return out

    def select(self, i):
        out = Tensor(self.data[i], (self,), "select")

        def _bw(node):
            buf = np.zeros_like(self.data)
            buf[i] += node.grad
            _acc(self, buf)

        out._backward = _bw
        return out


# ---- composite ops ---------------------------------------------------
def softmax(x, axis=-1):
    m = x.data.max(axis=axis, keepdims=True)
    e = np.exp(x.data - m)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(s, (x,), "softmax")

    def _bw(node):
        g = node.grad
        _acc(x, s * (g - (g * s).sum(axis=axis, keepdims=True)))

    out._backward = _bw
    return out


def log_softmax(x, axis=-1):
    m = x.data.max(axis=axis, keepdims=True)
    z = x.data - m
    ls = z - np.log(np.exp(z).sum(axis=axis, keepdims=True))
    out = Tensor(ls, (x,), "log_softmax")

    def _bw(node):
        g = node.grad
        _acc(x, g - np.exp(node.data) * g.sum(axis=axis, keepdims=True))

    out._backward = _bw
    return out


def gelu(x):
    c = float(np.sqrt(2.0 / np.pi))
    inner = (x + 0.044715 * x.pow(3)) * c
    return (x * 0.5) * (inner.tanh() + 1.0)


def dropout(x, p, rng):
    if not p:
        return x
    keep = 1.0 - p
    mask = (rng.random(x.data.shape) < keep).astype(np.float32) / keep
    out = Tensor(x.data * mask, (x,), "dropout")

    def _bw(node):
        _acc(x, node.grad * mask)

    out._backward = _bw
    return out


def gather(t, idx):
    idx = np.asarray(idx)
    out = Tensor(t.data[idx], (t,), "gather")

    def _bw(node):
        buf = np.zeros_like(t.data)
        np.add.at(buf, idx, node.grad)
        _acc(t, buf)

    out._backward = _bw
    return out


def pick_last(t, idx):
    idx = np.asarray(idx)[..., None]
    out = Tensor(np.take_along_axis(t.data, idx, -1)[..., 0], (t,), "pick")

    def _bw(node):
        buf = np.zeros_like(t.data)
        np.put_along_axis(buf, idx, node.grad[..., None], -1)
        _acc(t, buf)

    out._backward = _bw
    return out


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = (var + eps).pow(-0.5)
    return (xc * inv) * w + b


def cross_entropy(logits, targets):
    return pick_last(log_softmax(logits, -1), targets).mean() * -1.0
