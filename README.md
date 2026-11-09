# scratchgpt

[![CI](https://github.com/midya8/scratchgpt/actions/workflows/ci.yml/badge.svg)](https://github.com/midya8/scratchgpt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

A GPT built and trained **from scratch in pure NumPy**. No PyTorch, no JAX,
no TensorFlow: one ~400-line reverse-mode autograd engine with hand-written
backward rules for every op, a pre-LN transformer on top of it, and a model
actually trained to convergence on TinyShakespeare with nothing but a CPU
core and `numpy`.

```text
419,328 parameters | 2 layers, 4 heads, d_model 128 | val loss 1.83 nats/char
3,000 steps at 88 ms/step | ~4.5 minutes on one CPU core | zero ML dependencies
```

## Exact-continuation workbench (v0.2)

[Open the offline browser replay](https://midya8.github.io/scratchgpt/) ·
[Checkpoint contract and reproduction](CONTINUATION.md)

The trainer now saves complete train/evaluation RNG state, Adam buffers and clock,
data identity, numerical options and runtime metadata, with a validated no-pickle
loader and atomic continuation saves. A separate synthetic audit checks all 13
restart boundaries, exposes RNG/moment-loss counterexamples, and independently
recomputes every Adam coordinate and transformer forward pass. The original
Shakespeare model/results below are preserved, not retrained or rebenchmarked.
Old inference checkpoints remain sampleable but are not exact-resume artifacts.

## Results

Loss curve from the shipped checkpoint (real training run, logged every 200 steps):

![loss curve](docs/loss.png)

Text sampled from the trained checkpoint (temperature 0.8, top-k 20):

```
JULIET:
Sto hid, if heary sweet and so had clauden, or more am buto some
Compen to too dess to kince of Richord, but now way.

PARINCE:
Bit is my lord, hold well, you ammile, or murder.
...
```

Not Shakespeare yet, and the README won't pretend otherwise: this is a
419k-parameter model trained for 4 CPU-minutes. The point is that every
gradient that produced it was computed by ~400 lines of hand-written NumPy
autograd, verified against finite differences.

## Quickstart

```bash
pip install -e ".[dev]"

# train from scratch (~5 CPU-minutes for the shipped config)
python -m scratchgpt.train --data data/tinyshakespeare.txt --out runs/base --steps 3000

# or sample directly from the committed checkpoint
python -m scratchgpt.sample --ckpt checkpoints/final.npz --data data/tinyshakespeare.txt \
    --prompt "JULIET:" --n 400 --temperature 0.8 --top-k 20
```

## What's inside

| File | What it is |
| --- | --- |
| `scratchgpt/tensor.py` | Reverse-mode autograd: broadcasting-aware `add/mul/div/matmul`, `pow/exp/log/sqrt/tanh/relu/sigmoid`, reductions, `view/permute/select`, `softmax`, `log_softmax`, `gather` (scatter-add backward), `pick_last`, layer norm, fused cross-entropy, dropout |
| `scratchgpt/nn.py` | Linear, Embedding, LayerNorm, causal multi-head self-attention, GELU MLP, pre-LN residual block |
| `scratchgpt/model.py` | GPT: token + learned positional embeddings, causal mask, LM head, GPT-2-style residual init |
| `scratchgpt/optim.py` | Adam with bias correction |
| `scratchgpt/train.py` | Training loop: warmup, gradient-norm clipping, periodic eval, validated exact-continuation checkpoints |
| `scratchgpt/sample.py` | Temperature + top-k sampling CLI |
| `tests/` | Every autograd op checked against central finite differences; model forward/backward; a training run that must reduce loss; a **causal-leak test** proving position *t* cannot see position *t+1* |

## Design notes

- **Hand-written backward, verified.** Each op implements its own vectorized
  backward rule with correct broadcasting reduction; the test suite
  differentiates every one of them numerically.
- **No reference cycles.** Backward closures receive their own output node as
  an argument instead of capturing it, so a computation graph frees the
  moment it goes out of scope. Training holds steady at ~350 MB of RSS.
- **GPT-2 details kept.** Pre-LN blocks, learned position embeddings,
  residual-path output scaling of 1/sqrt(2·n_layer), no biases.
- **Honest training setup.** Char-level vocab of 65, batch 16 × context 64,
  Adam at 3e-4 with 100-step warmup and global grad-norm clipping at 1.0.

## Limitations (on purpose)

Single CPU training means a small model and short context. The architecture
scales; the training script already handles checkpoints and resume, within the same numerical environment. See the continuation contract for data,
option and runtime constraints; this is not a cross-machine determinism promise.

## License

MIT. TinyShakespeare text is public domain.
