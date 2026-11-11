# Activation rematerialization: rebuild the interior, keep the gradient

[Live full-backward workbench](https://midya8.github.io/scratchgpt/remat.html)

This is an opt-in execution path for the original hand-written NumPy GPT, not a new optimizer or model. Contiguous groups of original transformer blocks execute once during forward, discard their interior Tensor graph, and execute again during backward. Their original backward rules then compute a local vector-Jacobian product (VJP) with the actual upstream derivative. The VJP adds contributions to the original boundary input and parameter leaves.

It is **activation checkpointing**, not the durable training checkpoint in [CONTINUATION.md](CONTINUATION.md). Original `GPT.__call__`, `Tensor.backward`, trainer, optimizer, continuation schema, Shakespeare checkpoint and historical results are unchanged. This module does not silently enable rematerialization in the trainer.

## Run it

```bash
pip install -e '.[dev]'
python -m scratchgpt.remat_audit --fixture > input.json
python -m scratchgpt.remat_audit input.json > report.json
python scripts/build_remat.py --check
python -m pytest -q
node --test tests/test_*.cjs
PYTHON=python bash examples/verify_remat_wheel.sh
```

Library use with the original GPT:

```python
from scratchgpt.remat import forward, graph_report
from scratchgpt.tensor import cross_entropy

# model is an existing original GPT with dropout=0; x/y are integer batches.
optimizer.zero_grad()
events = []
logits = forward(model, x, group_size=1, events=events)
loss = cross_entropy(logits, y)
retained = graph_report(loss, [*model.parameters(), model.mask(x.shape[1])])
loss.backward()
optimizer.step()
```

`group_size=0` delegates to the original eager model. Positive values, from 1 through `n_layer`, group adjacent blocks. A final shorter group is allowed. Bigger groups retain fewer boundary nodes but create a larger temporary graph during recomputation. No optimal schedule, peak-memory or speed claim follows from this grouping.

A fresh forward graph is required for each backward pass. Use first-order gradients only. Do not mutate weights, input arrays, mask, module structure or settings between forward and backward. Region guards bind Tensor shape/dtype/data digests and block settings/parameter identities, then verify recomputed output bytes before publishing that region's VJP. A failure can happen after other outer regions have already accumulated gradients. Discard all gradients after any failed backward pass and start again with a fresh graph.

The path accepts the original deterministic no-dropout Block modules, not an arbitrary callable. Tied parameters inside one region are rejected. Shared outer paths and separately checkpointed uses are tested for additive gradient contributions. Parameters are copied only inside each ephemeral region execution. Those copies and the region's temporary input/mask/graph are discarded after that execution. The wrapper retains original boundary/parameter references, output boundary values, small hashes and event counts, not a hidden original interior graph.

## Backward ownership

For a region `y = F(x, theta)` and upstream derivative `u`, recomputation builds fresh private leaves `x_copy`, `theta_copy` and reuses the original Block classes and arithmetic. Seed the local output gradient with `u`, run a local reverse postorder, then accumulate:

```
x.grad     += J_x(F)^T u
theta.grad += J_theta(F)^T u
```

The detached wrapper exposes the original `x` and parameter leaves as outer graph parents, but none of the original region interior. It never clears existing original parameter gradients. This matters with residual/fanout paths and pre-existing contributions. The standard engine's backward method is not rewritten to add this feature.

The original GPT does not pass a dropout RNG through its blocks. This extension does not quietly change that behavior. Its API refuses nonzero dropout settings instead of suggesting that stochastic recomputation or RNG replay has been verified.

## Measurements, not memory slogans

`graph_report` walks the actual graph reachable through Tensor `_prev` after the loss forward. Declared model parameter nodes and the causal mask are excluded. It reports Tensor-node count, operation counts and the sum of each remaining Tensor's `data.size`.

That sum is a **logical Tensor.data element count**, not unique storage, saved-backward bytes, allocator memory, process RSS, peak memory or measured speed. Views count once per Tensor even when their underlying allocations alias. It does not inspect extra ndarrays captured by backward closures. It also excludes gradients, audit copies/reports, Python objects, parameter/mask storage and BLAS buffers. The audit intentionally holds both full result arms and gradients, so its own process memory is not a training-memory benchmark. Event interior counts use the same policy with private parameter/mask nodes excluded.

For example, checked-in witness 7 has three layers, width 6, one attention head, one four-token sequence and 1,488 parameters. Its native loss graph has 5,361 logical elements eagerly and 324 with two blocks per region, with 182 versus 22 retained Tensor nodes. These are observed graph diagnostics for that fixture, not a claimed memory-reduction ratio. Both paths have identical native logits, gradients and one-step SGD arithmetic for this witness. Other environments are compared numerically, not promised bitwise identical.

The independent browser engine has scalar nodes rather than Tensor-array nodes. For this same input it retains 17,971 versus 837 scalar nodes. These counts are **not** expected to equal native logical-element counts. Each browser block, backward rule and checkpointed region actually executes; it does not substitute a stored native gradient or animate a planner. Browser scalar closure/value arrays and audit reports are not counted either.

## Full-coordinate input and report

`docs/remat-example.json` is a complete input for `python -m scratchgpt.remat_audit`. The browser accepts and downloads the same input schema, `scratchgpt.remat.input.v1`:

- Exact fields: `schema`, `config`, `group_size`, `learning_rate`, `tokens`, `targets`, `weights`.
- Config has the original five fields. Audit/browser limits are width 2-8, one to three layers, vocabulary 2-12, context 1-6, and valid divisible head counts. Library `forward` is not limited to these small audit models; the original 419,328-parameter checkpoint is tested separately.
- Tokens/targets are matching rectangular integer batches, one or two rows, within the vocabulary and context bounds. Targets enter the CE objective, not the transformer input.
- Learning rate is finite in `[0, .1]`. The reported update is a one-step SGD arithmetic comparison, not an optimizer-state checkpoint or a new training run.
- `weights` lists every original named parameter in `named_parameters` order, with exact name, shape and flat values. Values must be finite in `[-10, 10]`; native loads them as float32. Missing, reordered or malformed weights are rejected. CLI files are bounded to 500 KB. These are numerical inputs, not executable/pickle objects or authenticated checkpoints.

The native report schema is `scratchgpt.remat.report.v1`. Both `eager` and `rematerialized` include full logits, CE, graph diagnostics, execution events, every named parameter gradient and every SGD-updated weight. The comparison covers all coordinates, not just a norm. The original input is included.

The browser report is explicitly `scratchgpt.remat.browser.v1`: independent scalar float64 results and scalar graph counts, not the native Tensor counts. Float64 browser and float32 native are compared with numerical tolerances. A matching stored native witness is shown only when the complete input matches. Edited inputs run both real browser paths immediately when submitted, but no native result is invented for them. Download input and run the CLI to obtain a fresh native report. Editing clears stale results and disables downloads until a new successful run.

## Evidence

- Original 70 Python tests retained. Added 41 tests, for 111 total, cover all grouping choices, 30 seeded batch/length/head/width/layer shapes with nonuniform upstream derivatives, five-step original Adam trajectories, fanout and shared-parameter contributions, pre-existing parameter gradients, all-parameter float64 finite differences using the independent forward oracle, malformed schema/CLI inputs, mutation/replacement rejection, one-use graphs and actual GC-visible discard of forward interiors.
- Every one of the original trained checkpoint's 419,328 parameter gradients is compared to original eager backward for group sizes 1 and 2, together with exact logits. This is a short gradient audit of the shipped weights, not retraining or a new Shakespeare result.
- Eighteen complete native witnesses contain 15,184 parameter positions across bounded shapes and all grouping choices, including eager controls. Half of the six base fixtures use weights after three actual original eager Adam steps. Every witness is rebuilt independently of its stored output.
- An independent JavaScript scalar autograd engine implements embedding lookup, layer norm, QKV causal multi-head attention, residuals, tanh-approximation GELU, head, CE, reverse differentiation and region VJPs. Node tests compare 61,888 numeric fields against every full native witness, plus an all-88-parameter finite-difference scalar control and edited-input/grouping controls.
- Actual Chrome checks all 18 full reports, 106,288 parameter-table cells, 1,206 coordinate selections and 54 viewport layouts. Trusted keystrokes clear stale output; two actual downloaded files are read and the downloaded input is replayed natively. Original continuation browser tests are retained.
- The installed wheel is imported outside the checkout, rebuilds every native witness's gradients/updates, runs both new CLI paths and audits the original trained checkpoint's full gradients for both groupings. Original installed continuation/sample reproduction remains a separate unchanged check.

## Prior work and source notes

- [Chen et al. (2016), Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/html/1604.06174v2): saving selected boundaries and recomputing interiors; its asymptotic scheduling results are not claimed for this bounded grouping implementation.
- [PyTorch checkpoint reference](https://docs.pytorch.org/docs/main/utils/checkpoint.html): changed forward/recompute state can silently invalidate gradients; stochastic RNG handling and reentrant variants have specific constraints. PyTorch is not used here.
- [PyTorch activation-checkpointing engineering guide](https://pytorch.org/blog/activation-checkpointing-techniques/): storage/compute tradeoffs and the distinction between region retention and peak memory. Its performance figures do not describe this NumPy engine.
- [JAX saved-residual/checkpoint guide](https://docs.jax.dev/en/latest/gradient-checkpointing.html) and [API reference](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html): checkpointed inputs/boundaries remain while selected residuals are rebuilt. JAX is not used here.
- [PyTorch historical RNG-preservation discussion](https://github.com/pytorch/pytorch/pull/14253): a reason not to infer correct dropout replay from deterministic examples.
- [Independent gradient-checkpointing guide](https://residentmario.github.io/pytorch-training-performance-guide/gradient-checkpoints.html): an additional engineering account of the storage/compute tradeoff, not a benchmark of this project.

Activation checkpointing, transformer models, reverse-mode differentiation and SGD/Adam are established techniques. No new method, GPU/FlashAttention kernel, optimal schedule, higher-order differentiation, training speedup, process-memory reduction, held-out gain or production-scale result is claimed.
