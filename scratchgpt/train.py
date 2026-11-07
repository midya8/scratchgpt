"""Train a GPT from scratch on a text file, on CPU, in pure NumPy."""
import argparse
import json
import math
import os
import time

import numpy as np

from .data import CharVocab, get_batch, load_text, train_val_split
from .model import GPT
from .optim import Adam
from .tensor import cross_entropy


def clip_grad_norm(params, max_norm):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float((p.grad * p.grad).sum())
    total = math.sqrt(total)
    if max_norm and total > max_norm:
        s = max_norm / total
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * s
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--time-budget", type=float, default=780.0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    text = load_text(args.data)
    vocab = CharVocab(text)
    data = vocab.encode(text)
    train_data, val_data = train_val_split(data)

    cfg = dict(vocab_size=len(vocab), block_size=args.block,
               n_layer=args.n_layer, n_head=args.n_head,
               d_model=args.d_model)
    model = GPT(rng=rng, dropout=0.0, **cfg)
    opt = Adam(model.parameters(), lr=args.lr)
    print(f"model: {model.n_params:,} parameters, vocab {len(vocab)}")

    ckpt_path = os.path.join(args.out, "ckpt.npz")
    final_path = os.path.join(args.out, "final.npz")
    step, history = 0, []
    if os.path.exists(ckpt_path):
        z = np.load(ckpt_path, allow_pickle=False)
        saved = json.loads(str(z["config"]))
        if saved != cfg:
            raise SystemExit(f"checkpoint config {saved} != {cfg}")
        for i, p in enumerate(model.parameters()):
            p.data[...] = z[f"p{i}"]
        opt.t = int(z["opt_t"])
        for i, p in enumerate(model.parameters()):
            opt.m[i][...] = z[f"m{i}"]
            opt.v[i][...] = z[f"v{i}"]
        step = int(z["step"])
        history = json.loads(str(z["history"]))
        print(f"resumed at step {step}")

    end = step + args.steps
    deadline = time.time() + args.time_budget
    eval_rng = np.random.default_rng(args.seed + 1)
    t0 = time.time()

    def eval_val():
        losses = []
        for _ in range(args.eval_batches):
            x, y = get_batch(val_data, args.block, args.batch, eval_rng)
            losses.append(float(cross_entropy(model(x), y).data))
        return float(np.mean(losses))

    loss = None
    while step < end:
        if time.time() > deadline:
            print("time budget reached")
            break
        x, y = get_batch(train_data, args.block, args.batch, rng)
        loss = cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), args.clip)
        if step < args.warmup:
            opt.lr = args.lr * (step + 1) / args.warmup
        opt.step()
        step += 1
        if step % args.log_every == 0:
            rate = (time.time() - t0) / max(step - (end - args.steps), 1)
            print(f"step {step:5d}  train {float(loss.data):.4f}  {rate:.3f}s/step")
        if step % args.eval_every == 0:
            vl = eval_val()
            history.append([step, float(loss.data), vl])
            print(f"step {step:5d}  val   {vl:.4f}")

    if loss is not None:
        history.append([step, float(loss.data), eval_val()])
    np.savez(ckpt_path, step=step, opt_t=opt.t,
             config=json.dumps(cfg), history=json.dumps(history),
             **{f"p{i}": p.data for i, p in enumerate(model.parameters())},
             **{f"m{i}": opt.m[i] for i in range(len(opt.m))},
             **{f"v{i}": opt.v[i] for i in range(len(opt.v))})
    np.savez(final_path, config=json.dumps(cfg),
             **{f"p{i}": p.data for i, p in enumerate(model.parameters())})
    print(json.dumps({"step": step, "last": history[-3:]}))


if __name__ == "__main__":
    main()
