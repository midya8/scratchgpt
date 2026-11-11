"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const fmt = (value) => Number(value).toPrecision(8);
  let data,
    base,
    report,
    page = 0;

  function rows(element, values) {
    element.replaceChildren(
      ...values.map((values) => {
        const tr = document.createElement("tr");
        for (const value of values) {
          const td = document.createElement("td");
          td.textContent = String(value);
          tr.append(td);
        }
        return tr;
      }),
    );
  }
  function options(id, values) {
    $(id).replaceChildren(
      ...values.map(([value, label]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        return option;
      }),
    );
  }
  function clear() {
    report = null;
    window.cacheReport = null;
    $("results").hidden = true;
    $("download-input").disabled = true;
    $("download-report").disabled = true;
    $("error").textContent = "";
    $("status").textContent = "Input changed. Run again to replace the result.";
  }
  function setInput(input) {
    base = structuredClone(input);
    $("tokens").value = JSON.stringify(input.tokens);
    $("chunks").value = JSON.stringify(input.chunks);
    $("input-json").value = JSON.stringify(input, null, 2);
    const c = input.config;
    $("model-info").textContent =
      `${c.n_layer} layers / ${c.n_head} heads / width ${c.d_model} / context ${c.block_size} / vocabulary ${c.vocab_size}`;
    clear();
  }
  function cacheRows() {
    if (!report) return;
    const s = report.steps[Number($("step").value)];
    const layer = Number($("layer").value),
      a = s.cache[layer],
      b = s.full_engine_cache[layer];
    const flatten = (values, prefix = []) =>
      values.flatMap((v, i) =>
        Array.isArray(v)
          ? flatten(v, [...prefix, i])
          : [[[...prefix, i].join("/"), v]],
      );
    const keys = flatten(a.key),
      values = flatten(a.value),
      refKeys = flatten(b.key),
      refValues = flatten(b.value);
    const start = page * 32,
      end = Math.min(start + 32, keys.length);
    rows(
      $("kv").tBodies[0],
      keys
        .slice(start, end)
        .map(([index, value], j) => [
          index,
          fmt(value),
          fmt(refKeys[start + j][1]),
          fmt(values[start + j][1]),
          fmt(refValues[start + j][1]),
        ]),
    );
    $("page-label").textContent = `${start}-${end - 1} of ${keys.length}`;
    $("previous").disabled = page === 0;
    $("next").disabled = end >= keys.length;
  }
  function step() {
    if (!report) return;
    const s = report.steps[Number($("step").value)];
    $("step-info").textContent =
      `${s.mode}: consumed ${s.consumed} tokens. Window starts at stream index ${s.window_start}. Context ${JSON.stringify(s.context)}. Rebased positions ${JSON.stringify(s.positions)}.`;
    rows($("mask").tHead, [
      ["Query / key", ...s.causal_mask[0].map((_, i) => i)],
    ]);
    rows(
      $("mask").tBodies[0],
      s.causal_mask.map((row, i) => [i, ...row]),
    );
    rows(
      $("logits").tBodies[0],
      s.cached_logits.flatMap((batch, b) =>
        batch.flatMap((logits, p) =>
          logits.map((value, token) => [
            b,
            p,
            token,
            fmt(value),
            fmt(s.reference_logits[b][p][token]),
            fmt(value - s.reference_logits[b][p][token]),
          ]),
        ),
      ),
    );
    page = 0;
    cacheRows();
  }
  function run(input) {
    clear();
    try {
      if (!input) {
        input = structuredClone(base);
        input.tokens = CacheLab.parseJSON($("tokens").value);
        input.chunks = CacheLab.parseJSON($("chunks").value);
      }
      const r = CacheLab.audit(input);
      report = r;
      window.cacheReport = r;
      $("results").hidden = false;
      $("download-input").disabled = false;
      $("download-report").disabled = false;
      $("input-json").value = JSON.stringify(r.input, null, 2);
      $("logit-delta").textContent = fmt(r.comparison.max_logit_delta);
      $("stale-delta").textContent = fmt(r.comparison.stale_shift_max_delta);
      const canonical = (value) =>
        Array.isArray(value)
          ? value.map(canonical)
          : value && typeof value === "object"
            ? Object.fromEntries(
                Object.keys(value)
                  .sort()
                  .map((k) => [k, canonical(value[k])]),
              )
            : value;
      const native = data.reports.find(
        (n) =>
          JSON.stringify(canonical(n.input)) ===
          JSON.stringify(canonical(r.input)),
      );
      $("native-status").textContent = native
        ? `Matching full native witness. Original/native cached max logit delta ${fmt(native.comparison.max_logit_delta)}; KV delta ${fmt(native.comparison.max_kv_delta)}. Numerical tolerance, not bitwise equality.`
        : "Edited input. No matching stored native witness. Download the input and run python -m scratchgpt.cache_audit input.json to produce a fresh original-GPT comparison.";
      options(
        "step",
        r.steps.map((s, i) => [
          i,
          `Chunk ${i + 1} / ${s.mode} / ${s.consumed} consumed`,
        ]),
      );
      options(
        "layer",
        r.steps[0].cache.map((_, i) => [i, `Layer ${i}`]),
      );
      rows(
        $("steps").tBodies[0],
        r.steps.map((s) => [
          s.consumed,
          s.mode,
          s.window_start,
          s.cumulative_counts.projected_layer_rows,
          s.cumulative_counts.attention_score_cells,
          s.native_float32_kv_bytes,
        ]),
      );
      step();
      $("status").textContent =
        `Executed ${r.steps.length} chunks. All weights, logits and KV arrays are included in the downloadable report. No native result is used by the browser computation.`;
    } catch (e) {
      clear();
      $("error").textContent = e.message;
      $("status").textContent =
        "No current result. Fix the input and run again.";
    }
  }
  function reset() {
    setInput(data.reports[Number($("preset").value)].input);
    run();
  }
  $("audit-form").addEventListener("submit", (e) => {
    e.preventDefault();
    run();
  });
  $("preset").addEventListener("change", reset);
  $("reset").addEventListener("click", reset);
  for (const id of ["tokens", "chunks", "input-json"])
    $(id).addEventListener("input", clear);
  $("import").addEventListener("click", () => {
    const text = $("input-json").value;
    clear();
    try {
      const input = CacheLab.parseJSON(text);
      CacheLab.validate(input);
      setInput(input);
      run(input);
    } catch (e) {
      $("error").textContent = e.message;
    }
  });
  $("step").addEventListener("change", step);
  $("layer").addEventListener("change", () => {
    page = 0;
    cacheRows();
  });
  $("previous").addEventListener("click", () => {
    page--;
    cacheRows();
  });
  $("next").addEventListener("click", () => {
    page++;
    cacheRows();
  });
  function download(name, value) {
    if (!report) return;
    const a = document.createElement("a"),
      url = URL.createObjectURL(
        new Blob([JSON.stringify(value, null, 2) + "\n"], {
          type: "application/json",
        }),
      );
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 20000);
  }
  $("download-input").addEventListener("click", () =>
    download("scratchgpt-cache-input.json", report?.input),
  );
  $("download-report").addEventListener("click", () =>
    download("scratchgpt-cache-browser-report.json", report),
  );
  fetch("cache-witnesses.json")
    .then((r) => {
      if (!r.ok) throw Error("Native controls failed to load");
      return r.json();
    })
    .then((d) => {
      data = d;
      options(
        "preset",
        d.reports.map((r, i) => [
          i,
          `Control ${i + 1} / context ${r.input.config.block_size} / batch ${r.input.tokens.length} / chunks ${r.input.chunks.join(",")}`,
        ]),
      );
      reset();
      window.cacheReady = true;
    })
    .catch((e) => {
      clear();
      $("error").textContent = e.message;
    });
  fetch("cache-benchmark.json")
    .then((r) => {
      if (!r.ok) throw Error("Benchmark unavailable");
      return r.json();
    })
    .then((b) => {
      rows(
        $("benchmark").tBodies[0],
        b.cases.map((c) => [
          c.prompt_length,
          ...["original_autograd", "numpy_full", "numpy_cached"].map((k) =>
            (c.arms[k].median_ns / 1e6).toFixed(3),
          ),
          c.cache_only_median_factor.toFixed(3) + "x",
          c.arms.numpy_cached.executed_counts.window_rebuilds || 0,
        ]),
      );
      $("benchmark-cards").replaceChildren(
        ...b.cases.map((c) => {
          const card = document.createElement("article"),
            title = document.createElement("h3"),
            list = document.createElement("dl");
          title.textContent = `Prompt ${c.prompt_length} / 48 steps`;
          const values = [
            [
              "Original autograd",
              (c.arms.original_autograd.median_ns / 1e6).toFixed(3) + " ms",
            ],
            [
              "NumPy full",
              (c.arms.numpy_full.median_ns / 1e6).toFixed(3) + " ms",
            ],
            [
              "NumPy cached",
              (c.arms.numpy_cached.median_ns / 1e6).toFixed(3) + " ms",
            ],
            ["Full / cached", c.cache_only_median_factor.toFixed(3) + "x"],
            [
              "Window rebuilds",
              c.arms.numpy_cached.executed_counts.window_rebuilds || 0,
            ],
          ];
          for (const [label, value] of values) {
            const term = document.createElement("dt"),
              text = document.createElement("dd");
            term.textContent = label;
            text.textContent = value;
            list.append(term, text);
          }
          card.append(title, list);
          return card;
        }),
      );
      $("benchmark-info").textContent =
        `Python ${b.environment.python}; NumPy ${b.environment.numpy}; ${b.environment.os} ${b.environment.machine}. Checkpoint SHA-256 ${b.checkpoint_sha256}. These are archived local timings, not live browser timings.`;
    })
    .catch((e) => {
      $("benchmark-info").textContent = e.message + ". No timings displayed.";
    });
})();
