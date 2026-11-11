# Cached inference and the context boundary

The new path copies the original GPT's weights into an inference-only float32 NumPy snapshot. Prefill runs the full causal context; append computes only new token rows and reuses each layer's earlier keys and values. Training, `GPT.__call__`, the existing sampler, rematerialization and continuation formats stay unchanged.

## Try it

```bash
pip install -e '.[dev]'
python -m scratchgpt.cache_audit --fixture > input.json
python -m scratchgpt.cache_audit input.json > report.json
python scripts/build_cache.py --check
python -m pytest -q
node --test tests/test_*.cjs
PYTHON=python bash examples/verify_cache_wheel.sh
```

`docs/cache.html` is a standalone workbench. Serve `docs/` over HTTP rather than opening it as a file. It executes an independent float64 cached model and the existing browser full-forward reference, with all tokens, chunk lengths and named weights editable. It does not read stored native logits into its computation. A native-witness label appears only for an exact complete-input match. Edits clear output and disable downloads until another successful run. Downloaded inputs work in the native audit CLI.

```python
import numpy as np
from scratchgpt.cache_benchmark import load_legacy
from scratchgpt.inference import Inference

model, checkpoint_sha = load_legacy('checkpoints/final.npz')
snapshot = Inference(model)
session = snapshot.session()
logits = session.prefill(np.array([[1, 2, 3]]))
logits = session.advance(np.array([[4]]))
next_logits = logits[:, -1]
print(session.counts, session.cache_nbytes)
```

The snapshot owns copied, read-only weight arrays. Later training does not update it. It accepts the original deterministic modules with dropout zero, not arbitrary replacements. Separate sessions own separate token and KV arrays. Returned output mutations do not alter session state. Validation failures leave an existing session unchanged.

`prefill` replaces the current sequence batch and adds to its counters. `append` requires an existing prefill and refuses context overflow or a changed batch size. `advance` handles prefill, append or a full-window rebuild. `reset` clears buffers and counters. On append, logits cover only the new chunk. On a rebuild, they cover the entire final window. Use `[:, -1]` for next-token use.

## Why cropping requires a rebuild

The original sampler keeps the final `block_size` tokens and assigns them learned positions starting at zero on every full forward. Once that window moves, a retained token's position embedding changes. Its deeper-layer hidden state also depended on tokens that may now be outside the window. Removing one old key/value slot does not reproduce this computation.

`advance` therefore rebuilds **all layers of the final full-length window on every crop**. It never shortens the window to make the timing look better. The audit includes a deliberately wrong `stale_shift` control that removes old slots without rebuilding. This is a control defined here, not a claim about a library implementation. No-overflow and context-length-one controls can agree even on this wrong path.

Prefill uses a square lower-triangular mask. Appending `q` tokens after `p` retained tokens uses a `q` by `p+q` mask: query row `i` may attend through key `p+i`, not merely through key `i`. Multi-token append remains causal.

The final context on a multi-token crop follows the final full-forward window. Intermediate cropped windows inside that chunk are not returned. Use one-token advances when every generation step matters.

## Inputs, reports and transport

The bounded CLI/browser schema is `scratchgpt.cache.input.v1`, with exactly `schema`, `config`, `tokens`, `chunks` and `weights`.

- Config uses the original five dimensions. Audit bounds are vocabulary 2-16, context 1-8, layers 1-3, width 2-12 and valid divisible head counts.
- Tokens are one or two equal-length rows, each containing 1-24 integer IDs in the vocabulary. Booleans and fractional values are not IDs.
- Chunks are positive integer lengths no greater than the context size; they must partition the stream exactly.
- Weights cover every named original parameter, in original order, with exact name, shape and flat value count. Finite values in `[-10, 10]` are converted to float32. The browser normalizes weights to float32 first, then computes in float64.
- JSON is limited to two million bytes and depth 48. Both transports reject duplicate keys, including escaped spellings of the same key, and nonfinite values. The browser strict-parser grammar reuses this portfolio's AlignLab implementation with this audit's limits.

The native report, `scratchgpt.cache.report.v1`, includes every cached, NumPy-full and original-GPT logit, all layer KV arrays and independently reconstructed original-Block KV arrays, context tokens, rebased query positions, masks, counters, payload bytes and wrong-control last logits at each step. Original diagnostic projections are not part of benchmark timing. The browser report uses its own `scratchgpt.cache.browser.v1` schema and includes its computed full-reference arrays, not fabricated native results.

## What the counts mean

`projected_layer_rows` sums batch times query length times layers. `attention_score_cells` sums batch times heads times query length times key length times layers. Dense score matrices include masked cells. `head_rows` counts rows projected by the language-model head. Prefill, append and window rebuild counts describe actual execution paths.

These counts are matrix shapes, not FLOPs, measured runtime or a hardware cost model. The browser reports the same logical dense shapes even though it uses scalar loops. Its `native_float32_kv_bytes` is the modeled native KV payload, not JavaScript memory.

Native `cache_nbytes` sums resident K/V ndarray payloads. It excludes copied weights, token IDs, temporaries, audit references/reports, Python objects, allocator overhead, BLAS storage and process RSS. Snapshot weight payload is reported separately. Neither is peak memory.

## Local timing record

`docs/cache-benchmark.json` retains all seven timed repeats for each of four cases. Two warmups precede them; execution order rotates between the three arms. Warmup timings are not stored. The checkpoint is the unchanged 419,328-parameter historical model:

`3de6a1626e4489a2e0bc867fc58fe66d323ac6c527bb3af7be662b49510a817e`

Python 3.10.12, NumPy 2.2.6, Linux x86_64, scipy-openblas 0.3.29, `OPENBLAS_NUM_THREADS=1`. The CPU model was not recorded. These numbers cannot identify hardware or support a hardware comparison.

Each case has 48 teacher-forced steps. Times below are medians in milliseconds, including validation, output copying and Python overhead, excluding model loading and snapshot setup.

| Prompt | Original autograd | NumPy full | NumPy cached | Full / cached | Window rebuilds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 125.327 | 111.085 | 13.404 | 8.287x | 0 |
| 16 | 168.836 | 156.078 | 14.556 | 10.722x | 0 |
| 48 | 255.114 | 240.073 | 170.580 | 1.407x | 31 |
| 64 | 267.836 | 251.783 | 253.230 | 0.994x | 47 |

Only **NumPy full versus NumPy cached** isolates caching within the same backend. Comparing against original autograd also removes graph construction. At full context the cached path rebuilds every step, performs the same counted matrix work and is slightly slower in this run. No speed threshold is a test. Teacher forcing is not sampled generation; no decoding throughput or latency guarantee follows.

Snapshot setup took 396,558 ns and copied 1,677,312 weight bytes. Final KV payload reaches 131,072 bytes in the full-window cases. These are separate measurements, not a process-memory claim.

Reproduce a new timing record, not these exact wall-clock values:

```bash
OPENBLAS_NUM_THREADS=1 python -m scratchgpt.cache_benchmark \
  --checkpoint checkpoints/final.npz --prompts 4 16 48 64 \
  --steps 48 --repeats 7 --warmups 2 > new-benchmark.json
```

The legacy checkpoint loader rejects pickle/object arrays, unexpected or duplicate ZIP members, oversized expanded input, malformed config and wrong parameter headers, shapes, dtypes or nonfinite values. It validates NPY headers before loading parameter arrays. It loads the old inference format, not an exact-continuation checkpoint.

## Evidence and numerical limits

The full suite has 158 Python tests, retaining all 111 prior tests. The 47 new tests cover 100 seeded small models/chunkings, every two-chunk boundary, causality, rolling contexts, context one, batch changes, ownership, snapshot isolation, transactional validation, executed counts, all 32 partitions of a short stream, malformed transport and checkpoint inputs, and the three-arm benchmark without a speed gate.

Eight complete frozen controls cover 43 steps, 685 cached-logit coordinates and 3,256 cached K/V coordinates. They include different batches, heads, layers and chunkings, no-crop and context-one controls, and tiny models after three original Adam updates. They are synthetic controls, not the Shakespeare weights.

Four new Node tests compare every frozen native logit/KV coordinate, mask, context and counter, plus partitions and invalid input. A real Chrome test checks all 43 steps, every displayed logit and all paged KV coordinates, 24 control/viewport combinations at 320, 390 and 1280 pixels, trusted-keystroke invalidation, errors, complete-input import, two real downloads and native replay of the downloaded input. The original continuation/rematerialization browser tests remain unchanged.

An isolated installed-wheel check imports all three new modules outside the checkout, rebuilds all frozen reports, exercises both audit CLI routes and the benchmark CLI, and checks 80 one-token steps of the original checkpoint, including 16 window rebuilds.

Float32 matrix-vector and matrix-matrix execution can round differently. Small-model comparisons use their separate `2e-6` absolute / `2e-5` relative bound. The new shipped-checkpoint cache test uses an explicit `1e-5` absolute bound with zero relative tolerance, plus a `3e-6` softmax total-variation bound. No old tolerance is relaxed. The archived benchmark's largest cached/original logit difference is below `6.5e-6`; the NumPy full arm matched original logits in that run. This is numerical closeness, not bitwise or universal sampled-token equivalence. An RNG threshold can separate two close distributions.

## Prior work and scope

KV caching and causal transformer attention are established techniques. This is a small implementation and audit, not a new method.

- [Hugging Face GPT-2 reference](https://huggingface.co/docs/transformers/model_doc/gpt2) documents learned position IDs and past-key/value use.
- [Transformers cache guide source](https://github.com/huggingface/transformers/blob/main/docs/source/en/kv_cache.md) discusses cache positions, masks and implementation tradeoffs. This project does not implement its cache APIs.
- [nanoGPT model source](https://github.com/karpathy/nanoGPT/blob/master/model.py) and [historical cache proposal #76](https://github.com/karpathy/nanoGPT/pull/76) provide the crop/recompute context. That proposal is closed, not a merged feature. Its timings are not ours.
- [Learned-position cache discussion](https://discuss.huggingface.co/t/how-does-attention-key-value-caching-work-with-models-that-have-learned-absolute-position-embeddings/56469) explains why position resets matter; community replies are not a library contract.
- [Independent nanoGPT cache write-up](https://czhou578.github.io/blog/2026/05/10/adding-kv-cache-to-nanogpt.html) was reviewed as a contrast. This implementation does not adopt its broad constant-time claim, unmasked prefill example or general claims about position schemes. The original local model and explicit full-forward comparisons decide correctness here.

No GPU kernels, quantization, paged allocation, batching scheduler, concurrent serving, gradients through the cache, changing weights inside a session, stochastic dropout replay or new training/generalization result is claimed.
