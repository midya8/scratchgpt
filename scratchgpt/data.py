"""Character-level dataset utilities: vocab, split, batcher."""
import numpy as np


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharVocab:
    def __init__(self, text):
        self.itos = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int32)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def train_val_split(data, val_frac=0.1):
    n_val = int(len(data) * val_frac)
    return data[:-n_val], data[-n_val:]


def get_batch(data, block, batch, rng):
    ix = rng.integers(0, len(data) - block - 1, size=batch)
    x = np.stack([data[i:i + block] for i in ix])
    y = np.stack([data[i + 1:i + block + 1] for i in ix])
    return x, y
