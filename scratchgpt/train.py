"""CPU training with validated, atomic, exact step-boundary continuation."""
import argparse
import json
import math
import os
import time
import zipfile

import numpy as np

from .checkpoint import Training
from .data import CharVocab, load_text


def clip_grad_norm(params, max_norm):
    """Original public helper retained."""
    total = math.sqrt(sum(float((p.grad*p.grad).sum()) for p in params if p.grad is not None))
    if max_norm and total > max_norm:
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * (max_norm/total)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='runs/base')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--block', type=int, default=64)
    ap.add_argument('--n-layer', type=int, default=2)
    ap.add_argument('--n-head', type=int, default=4)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=200)
    ap.add_argument('--log-every', type=int, default=200)
    ap.add_argument('--eval-batches', type=int, default=10)
    ap.add_argument('--time-budget', type=float, default=780.0)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()
    try:
        if args.steps < 0 or args.log_every < 1 or not np.isfinite(args.time_budget) or args.time_budget < 0:
            raise ValueError('steps/time-budget must be nonnegative; log-every positive')
        text = load_text(args.data)
        cfg = dict(vocab_size=len(CharVocab(text)),block_size=args.block,n_layer=args.n_layer,
                   n_head=args.n_head,d_model=args.d_model)
        opts = dict(batch=args.batch,lr=args.lr,warmup=args.warmup,clip=args.clip,
                    eval_every=args.eval_every,eval_batches=args.eval_batches,seed=args.seed)
        ckpt = os.path.join(args.out,'ckpt.npz')
        run = Training.load(ckpt,text,cfg,opts) if os.path.exists(ckpt) else Training(text,cfg,opts)
        os.makedirs(args.out,exist_ok=True)
        print(f'model: {run.model.n_params:,} parameters; starting at step {run.step}')
        end, start = run.step+args.steps, run.step
        t0 = time.monotonic()
        while run.step < end and time.monotonic()-t0 < args.time_budget:
            row = run.update()
            if run.step % args.log_every == 0:
                print(f'step {run.step:5d} train {row["loss"]:.4f} {(time.monotonic()-t0)/(run.step-start):.3f}s/step')
            if run.step % args.eval_every == 0:
                print(f'step {run.step:5d} val {run.history[-1][2]:.4f}')
        run.save(ckpt)
        # Backward-compatible inference-only export. Not a continuation checkpoint.
        np.savez(os.path.join(args.out,'final.npz'),config=json.dumps(cfg),
                 **{f'p{i}':p.data for i,p in enumerate(run.model.parameters())})
        print(json.dumps(dict(step=run.step,history=run.history[-3:],preview_val_loss=run.evaluate(),
                              state_sha256=run.digest(),schema='scratchgpt.continuation.v1')))
    except (ValueError,OSError,KeyError,TypeError,EOFError,zipfile.BadZipFile) as e:
        ap.exit(2, f'checkpoint/training error: {e}\n')


if __name__ == '__main__':
    main()
