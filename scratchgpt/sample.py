"""Sample text from a trained checkpoint."""
import argparse
import json

import numpy as np

from .data import CharVocab, load_text
from .model import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.ckpt, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    vocab = CharVocab(load_text(args.data))
    model = GPT(rng=np.random.default_rng(args.seed), dropout=0.0, **cfg)
    for i, p in enumerate(model.parameters()):
        p.data[...] = z[f"p{i}"]

    rng = np.random.default_rng(args.seed)
    idx = list(vocab.encode(args.prompt))
    for _ in range(args.n):
        ctx = np.array(idx[-cfg["block_size"]:])[None]
        logits = model(ctx).data[0, -1] / args.temperature
        if args.top_k:
            kth = np.sort(logits)[-args.top_k]
            logits = np.where(logits < kth, -np.inf, logits)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        idx.append(int(rng.choice(len(p), p=p)))
    print(vocab.decode(np.array(idx)))


if __name__ == "__main__":
    main()
