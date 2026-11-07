"""A GPT: token + learned positional embeddings, pre-LN blocks, LM head."""
import numpy as np

from .nn import Block, Embedding, LayerNorm, Linear
from .tensor import Tensor, gather, layer_norm


def causal_mask(T):
    m = np.zeros((1, 1, T, T), dtype=np.float32)
    tri = np.triu(np.ones((T, T), dtype=bool), 1)
    m[0, 0][tri] = -1e9
    return Tensor(m)


class GPT:
    def __init__(self, vocab_size, block_size, n_layer, n_head, d_model, rng,
                 dropout=0.0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        scale = 1.0 / np.sqrt(2.0 * n_layer)  # GPT-2 residual-path init
        self.tok_emb = Embedding(vocab_size, d_model, rng)
        self.pos_emb = Embedding(block_size, d_model, rng)
        self.blocks = [Block(d_model, n_head, rng, dropout, scale)
                       for _ in range(n_layer)]
        self.ln_f = LayerNorm(d_model, rng)
        self.head = Linear(d_model, vocab_size, rng)
        self._masks = {}

    def mask(self, T):
        if T not in self._masks:
            self._masks[T] = causal_mask(T)
        return self._masks[T]

    def __call__(self, idx):
        B, T = idx.shape
        x = gather(self.tok_emb.w, idx) + gather(self.pos_emb.w, np.arange(T))
        m = self.mask(T)
        for blk in self.blocks:
            x = blk(x, m)
        x = layer_norm(x, self.ln_f.w, self.ln_f.b)
        return x @ self.head.w

    def parameters(self):
        ps = []
        for mod in [self.tok_emb, self.pos_emb, self.ln_f, self.head] + self.blocks:
            ps.extend(mod.parameters())
        return ps

    @property
    def n_params(self):
        return sum(int(p.data.size) for p in self.parameters())
