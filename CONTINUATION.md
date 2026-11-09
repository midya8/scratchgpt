# A checkpoint is a continuation contract

The previous trainer saved model parameters, Adam moments and the update clock,
but re-created its train/evaluation RNGs on every process invocation. A resumed
run therefore did not continue the same batch stream. It also accepted changed
text or training options without binding them to the saved state. This update
fixes that contract; it is not a new optimizer or transformer architecture.

## What is saved

`Training.save(path)` writes schema `scratchgpt.continuation.v1` as NPZ with
`allow_pickle=False` on load:

- All float32 parameters with explicit names, order and shapes.
- All Adam first/second moments, update clock, beta values, epsilon and current LR.
- Both complete PCG64 states, including cached uint32 information.
- Model dimensions, batch size, base LR, warmup, clipping, evaluation cadence,
  evaluation batches and seed.
- SHA-256 of the actual UTF-8 text used for tokenization, character count, sorted
  vocabulary and the existing contiguous last-floor-10-percent split rule.
- Scheduled evaluation history, last training loss, step and runtime fingerprint.

The boundary is **after the update and any scheduled evaluation, before the next
training batch**. `--steps` means additional updates; `--out`, time budget and
logging frequency do not affect the numerical state. Every update explicitly
sets its LR from the saved base LR and warmup schedule.

A terminal evaluation is now a **preview on a cloned evaluation RNG**. It neither
advances scheduled evaluation randomness nor inserts a terminal history row.
Otherwise simply stopping at a non-evaluation boundary would change later
history. Repeated previews leave the entire state digest unchanged.

The loader validates the complete artifact into a fresh object before returning
it. Unknown/missing fields or arrays, different data/options/environment,
parameter-order changes, wrong shapes/dtypes, nonfinite values, negative Adam
second moments, invalid RNGs, mismatched clocks/history/schedule, object arrays,
duplicate ZIP members and excessive archive size are refused. JSON duplicate
keys/nonfinite constants are refused. Bounds: 100 MB compressed/uncompressed,
250 members, one-million-character metadata, at most seven million parameters.
This is a bounded loader, not authentication of an arbitrary file's provenance.

A same-directory temporary file is flushed and fsynced, then `os.replace` replaces
the continuation file. Serialization/replace failures and nonfinite state do not
truncate the previous valid checkpoint. The temporary file is cleaned up. This
does not promise durability on every filesystem or against every power-loss
scenario. The legacy-compatible `final.npz` is a separate inference export, not
an atomic continuation file.

## Reproduce

```bash
pip install -e '.[dev]'
OPENBLAS_NUM_THREADS=1 pytest -q
OPENBLAS_NUM_THREADS=1 python scripts/build_replay.py --check
node --test tests/test_*.cjs  # Node 22+, Google Chrome installed
PYTHON=python OPENBLAS_NUM_THREADS=1 bash examples/verify_wheel.sh  # uv + build
```

A fresh run and a later invocation with the same numerical settings and output
directory resume automatically:

```bash
OPENBLAS_NUM_THREADS=1 python -m scratchgpt.train \
  --data data/tinyshakespeare.txt --out runs/continued --steps 100
OPENBLAS_NUM_THREADS=1 python -m scratchgpt.train \
  --data data/tinyshakespeare.txt --out runs/continued --steps 100
```

Keep every numerical option the same on both invocations. The existing shipped
`checkpoints/final.npz` has no continuation metadata and is deliberately refused
for resume. It still works with the unchanged sample CLI. Old `ckpt.npz` files
cannot reconstruct lost RNG state; no silent migration or exact-resume claim is
made. Start in a new directory to train anew. Manual copying of weights would be
a warm start, not continuation.

## The complete small-run audit

`docs/replay.json` contains all 12 updates of a distinct 1,056-parameter model:
one layer, two heads, dimension 8, context 6, batch 3, seed 7, LR 0.003,
warmup 5, clip 0.4, evaluation every 3 updates with 2 batches. The text is an
original four-line synthetic string repeated 60 times. This is not a language
benchmark and the tiny fit is not evidence of useful text quality.

For every stop boundary 0 through 12, a separate fit is saved, loaded and
continued to 12. All **13/13** final complete states match bit for bit within the
same environment. Every parameter/moment, RNG state and history is included, not
just the final loss. Stops before, at and after warmup and evaluation boundaries
are retained. The separate-process CLI test also compares 12 uninterrupted
updates with 4+8 updates in two processes, including terminal previews.

Counterexamples retain the same model, data, update algorithm and terminal step:

| Arm | What is changed on restart | Observed result |
| --- | --- | --- |
| Exact | Nothing omitted | 13/13 complete states identical |
| Legacy RNG reset | Re-create train and eval RNGs as at a new invocation | Final weights differ at all 11 interior boundaries |
| Missing Adam memory | Zero m/v only, retain the clock and RNGs | Final weights differ at all 11 interior boundaries |
| Eval RNG only | Reset only the evaluation generator | Training weights identical; later scheduled history changes |

At boundary 4, maximum final weight differences are about **0.01715618** for
legacy RNG reset and **0.02066660** for missing moments. Boundaries 0 and 12
are controls: at 0 there is no previous stream/memory to lose; at 12 there is no
future update. The legacy arm isolates the RNG omission rather than reproducing
the old trainer's extra terminal-history row. These are descriptive effects for
this one run, not rankings of which state category matters more in general.

Every one of 12,672 parameter-coordinate updates is independently checked using
a scalar float64 Adam equation: **38,016** p/m/v values, maximum absolute error
**5.933e-8** against native float32. An independent forward pass with no Tensor,
GPT, nn or Adam calls compares **2,592** logits, maximum error **3.031e-8**.
Additional finite differences use that independent forward objective for three
coordinates of every parameter tensor. Tests cover more random models, one/two
layers, several context lengths and causal non-leakage.

The browser independently recomputes all equations and forward inference from
full weights. It does **not** calculate gradients or train. Edited scalar
settings are one-coordinate arithmetic with a recorded, already-clipped
gradient, not a counterfactual training trajectory. All 12 updates, 13 parameter
tensors, 1,056 coordinates per update, batches and before/after logits remain
inspectable and downloadable. Actual Chrome tests inspect 101,376 coordinate
table cells, 468 selected-coordinate views, 72 inference views, both exports,
invalid-input clearing and 320/390/1280 layouts.

## Environment and numerical limits

Exactness here is between uninterrupted/resumed runs in the same environment.
The fingerprint records Python/NumPy versions, platform/machine, NumPy build
configuration and five common thread environment settings. It is necessary,
not a proof that every runtime setting/hardware detail is captured. Matching a
fingerprint does not create a universal determinism guarantee. Cross-machine,
BLAS-build, NumPy-version and floating-point reduction differences can change
results. Regeneration across supported Python versions compares values within
1e-6 rather than requiring saved runtime-specific digests. Browser float64/native
float32 comparisons use 2e-5 forward and 2e-7 Adam absolute tolerances.

The original model, autograd, optimizer, data utilities, ten original tests,
Shakespeare checkpoint/text, loss CSV/PNG and sample remain unchanged. The
shipped parameter count is independently verified as 419,328; checkpoint SHA-256
`3de6a1626e4489a2e0bc867fc58fe66d323ac6c527bb3af7be662b49510a817e`.
Historical 3,000-update loss/timing results are not a new training-speed run.

## Sources and attribution

Educational implementation and continuation/replay by Midya Rahmani. The
following algorithms and contracts are prior work, not claimed inventions:

- [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980): moment updates and bias correction.
- [NumPy compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html): strict stream-reproducibility conditions and limits.
- [PCG64 state](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.PCG64.state.html): complete generator state interface.
- [NumPy load](https://numpy.org/doc/stable/reference/generated/numpy.load.html): NPZ and the pickle boundary.
- [PyTorch checkpoint tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html): model/optimizer checkpoint versus inference weights. PyTorch is not used here.
- [Python os](https://docs.python.org/3/library/os.html): replace/fsync semantics.
- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762): transformer attention.
