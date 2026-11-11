/* Independent scalar float64 cached inference. Native implementation is float32.
   Reuses the older independent full-forward reference, never native witness logits. */
(function (root) {
  "use strict";
  const Reference =
    typeof module !== "undefined" ? require("./engine.js") : root.Replay;
  const keys = (o, names) =>
    o &&
    typeof o === "object" &&
    !Array.isArray(o) &&
    Object.keys(o).sort().join("|") === names.slice().sort().join("|");
  const integer = (x, a, b) => Number.isInteger(x) && x >= a && x <= b;
  const finite = (x) => typeof x === "number" && Number.isFinite(x);
  function shapeOf(x) {
    return Array.isArray(x) ? [x.length, ...shapeOf(x[0])] : [];
  }
  function validate(o) {
    if (
      !keys(o, ["schema", "config", "chunks", "tokens", "weights"]) ||
      o.schema !== "scratchgpt.cache.input.v1"
    )
      throw Error("Unsupported input schema or fields");
    const c = o.config;
    if (
      !keys(c, ["vocab_size", "block_size", "n_layer", "n_head", "d_model"]) ||
      !integer(c.vocab_size, 2, 16) ||
      !integer(c.block_size, 1, 8) ||
      !integer(c.n_layer, 1, 3) ||
      !integer(c.n_head, 1, 16) ||
      !integer(c.d_model, 2, 12) ||
      c.d_model % c.n_head
    )
      throw Error("Invalid bounded model config");
    if (
      !Array.isArray(o.tokens) ||
      !integer(o.tokens.length, 1, 2) ||
      !Array.isArray(o.tokens[0]) ||
      !integer(o.tokens[0].length, 1, 24) ||
      o.tokens.some(
        (r) =>
          !Array.isArray(r) ||
          r.length !== o.tokens[0].length ||
          r.some((t) => !integer(t, 0, c.vocab_size - 1)),
      )
    )
      throw Error("Invalid token batch");
    if (
      !Array.isArray(o.chunks) ||
      !integer(o.chunks.length, 1, 24) ||
      o.chunks.some((n) => !integer(n, 1, c.block_size)) ||
      o.chunks.reduce((a, b) => a + b, 0) !== o.tokens[0].length
    )
      throw Error("Invalid chunk partition");
    const d = c.d_model,
      expected = [
        ["tok_emb.w", [c.vocab_size, d]],
        ["pos_emb.w", [c.block_size, d]],
        ["ln_f.w", [d]],
        ["ln_f.b", [d]],
        ["head.w", [d, c.vocab_size]],
      ];
    for (let l = 0; l < c.n_layer; l++)
      for (const [n, s] of [
        ["ln1.w", [d]],
        ["ln1.b", [d]],
        ["attn.qkv.w", [d, 3 * d]],
        ["attn.proj.w", [d, d]],
        ["ln2.w", [d]],
        ["ln2.b", [d]],
        ["mlp.fc.w", [d, 4 * d]],
        ["mlp.proj.w", [4 * d, d]],
      ])
        expected.push(["blocks." + l + "." + n, s]);
    if (!Array.isArray(o.weights) || o.weights.length !== expected.length)
      throw Error("Complete named weights required");
    o.weights.forEach((r, i) => {
      let [n, s] = expected[i];
      if (
        !keys(r, ["name", "shape", "values"]) ||
        r.name !== n ||
        JSON.stringify(r.shape) !== JSON.stringify(s) ||
        !Array.isArray(r.values) ||
        r.values.length !== s.reduce((a, b) => a * b, 1) ||
        r.values.some((x) => !finite(x) || Math.abs(x) > 10)
      )
        throw Error("Invalid weight name, shape, order or value");
    });
    return c;
  }
  const reshape = (xs, shape) =>
    shape.length === 1
      ? xs.slice()
      : Array.from({ length: shape[0] }, (_, i) =>
          reshape(
            xs.slice(
              i * shape.slice(1).reduce((a, b) => a * b, 1),
              (i + 1) * shape.slice(1).reduce((a, b) => a * b, 1),
            ),
            shape.slice(1),
          ),
        );
  const mm = (row, w) =>
    w[0].map((_, j) => row.reduce((s, v, k) => s + v * w[k][j], 0));
  function norm(x, w, b) {
    const mean = x.reduce((a, b) => a + b, 0) / x.length,
      xc = x.map((v) => v - mean),
      inv = (xc.reduce((s, v) => s + v * v, 0) / x.length + 1e-5) ** -0.5;
    return xc.map((v, i) => v * inv * w[i] + b[i]);
  }
  const softmax = (xs) => {
    const mx = Math.max(...xs),
      e = xs.map((v) => Math.exp(v - mx)),
      sum = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / sum);
  };
  const delta = (a, b) => {
    let aa = a.flat(Infinity),
      bb = b.flat(Infinity);
    if (aa.length !== bb.length) throw Error("comparison shape mismatch");
    return aa.reduce((mx, v, i) => Math.max(mx, Math.abs(v - bb[i])), 0);
  };
  function run(w, c, tokens, past = null) {
    const B = tokens.length,
      T = tokens[0].length,
      D = c.d_model,
      H = c.n_head,
      dh = D / H,
      offset = past ? past[0].key[0][0].length : 0,
      total = offset + T;
    if (total > c.block_size)
      throw Error("Cache overflow; rebuild the context");
    let x = tokens.map((row) =>
        row.map((t, i) =>
          w["tok_emb.w"][t].map((v, j) => v + w["pos_emb.w"][offset + i][j]),
        ),
      ),
      cache = [];
    for (let l = 0; l < c.n_layer; l++) {
      let p = "blocks." + l + ".",
        qkv = x.map((row) =>
          row.map((v) =>
            mm(norm(v, w[p + "ln1.w"], w[p + "ln1.b"]), w[p + "attn.qkv.w"]),
          ),
        );
      let key = Array.from({ length: B }, (_, b) =>
        Array.from({ length: H }, (_, h) => {
          const now = qkv[b].map((v) => v.slice(D + h * dh, D + (h + 1) * dh));
          return past
            ? past[l].key[b][h].map((v) => v.slice()).concat(now)
            : now;
        }),
      );
      let value = Array.from({ length: B }, (_, b) =>
        Array.from({ length: H }, (_, h) => {
          const now = qkv[b].map((v) =>
            v.slice(2 * D + h * dh, 2 * D + (h + 1) * dh),
          );
          return past
            ? past[l].value[b][h].map((v) => v.slice()).concat(now)
            : now;
        }),
      );
      let y = x.map((row, b) =>
        row.map((v, i) => {
          let heads = [];
          for (let h = 0; h < H; h++) {
            let q = qkv[b][i].slice(h * dh, (h + 1) * dh),
              scores = key[b][h].map(
                (k, j) =>
                  q.reduce((s, z, a) => s + z * k[a], 0) * dh ** -0.5 +
                  (j > offset + i ? -1e9 : 0),
              ),
              a = softmax(scores);
            for (let k = 0; k < dh; k++)
              heads.push(a.reduce((s, z, j) => s + z * value[b][h][j][k], 0));
          }
          let out = mm(heads, w[p + "attn.proj.w"]);
          return v.map((z, j) => z + out[j]);
        }),
      );
      x = y.map((row) =>
        row.map((v) => {
          let ff = mm(
            norm(v, w[p + "ln2.w"], w[p + "ln2.b"]),
            w[p + "mlp.fc.w"],
          ).map(
            (z) =>
              0.5 *
              z *
              (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z ** 3))),
          );
          let out = mm(ff, w[p + "mlp.proj.w"]);
          return v.map((z, j) => z + out[j]);
        }),
      );
      cache.push({ key, value, shape: [B, H, total, dh] });
    }
    const logits = x.map((row) =>
      row.map((v) => mm(norm(v, w["ln_f.w"], w["ln_f.b"]), w["head.w"])),
    );
    if (logits.flat(Infinity).some((v) => !Number.isFinite(v)))
      throw Error("Nonfinite inference");
    return {
      logits,
      cache,
      counts: {
        projected_layer_rows: B * T * c.n_layer,
        attention_score_cells: B * H * T * total * c.n_layer,
        head_rows: B * T,
      },
    };
  }
  function audit(input) {
    const c = validate(input),
      o = JSON.parse(JSON.stringify(input)),
      w = Object.fromEntries(
        o.weights.map((r) => [
          r.name,
          reshape(r.values.map(Math.fround), r.shape),
        ]),
      );
    let past = null,
      stale = null,
      consumed = 0,
      steps = [],
      cachedCounts = {},
      fullCounts = {};
    const charge = (dest, src) => {
      for (const [k, v] of Object.entries(src)) dest[k] = (dest[k] || 0) + v;
    };
    for (const chunk of o.chunks) {
      let old = past ? past[0].key[0][0].length : 0;
      consumed += chunk;
      const mode =
          old === 0
            ? "prefill"
            : old + chunk > c.block_size
              ? "rebuild"
              : "append",
        context = o.tokens.map((r) =>
          r.slice(Math.max(0, consumed - c.block_size), consumed),
        ),
        next = o.tokens.map((r) => r.slice(consumed - chunk, consumed));
      const cached = run(
          w,
          c,
          mode === "append" ? next : context,
          mode === "append" ? past : null,
        ),
        full = run(w, c, context);
      past = cached.cache;
      charge(cachedCounts, cached.counts);
      charge(fullCounts, full.counts);
      cachedCounts[mode === "append" ? "appends" : "prefills"] =
        (cachedCounts[mode === "append" ? "appends" : "prefills"] || 0) + 1;
      if (mode === "rebuild")
        cachedCounts.window_rebuilds = (cachedCounts.window_rebuilds || 0) + 1;
      if (stale && old + chunk > c.block_size) {
        const n = old + chunk - c.block_size;
        stale = stale.map((layer) => ({
          key: layer.key.map((bs) => bs.map((h) => h.slice(n))),
          value: layer.value.map((bs) => bs.map((h) => h.slice(n))),
        }));
      }
      const wrong = run(w, c, next, stale);
      stale = wrong.cache;
      const reference = context.map((row) => Reference.forward(w, c, row)),
        length = cached.logits[0].length,
        expected = reference.map((row) => row.slice(-length));
      let kvDelta = 0;
      for (let i = 0; i < past.length; i++)
        for (const k of ["key", "value"])
          kvDelta = Math.max(kvDelta, delta(past[i][k], full.cache[i][k]));
      const positions = Array.from(
        { length },
        (_, i) => context[0].length - length + i,
      );
      steps.push({
        mode,
        consumed,
        window_start: consumed - context[0].length,
        context,
        positions,
        causal_mask: positions.map((q) =>
          Array.from({ length: context[0].length }, (_, k) => +(k <= q)),
        ),
        cached_logits: cached.logits,
        reference_logits: expected,
        full_engine_logits: full.logits.map((row) => row.slice(-length)),
        cache: past,
        full_engine_cache: full.cache,
        cumulative_counts: { ...cachedCounts },
        native_float32_kv_bytes:
          2 * o.tokens.length * context[0].length * c.d_model * c.n_layer * 4,
        max_logit_delta: delta(cached.logits, expected),
        max_kv_delta: kvDelta,
        full_reference_delta: delta(full.logits, reference),
        stale_shift_last_logits: wrong.logits.map((row) => row.at(-1)),
        stale_shift_last_delta: delta(
          wrong.logits.map((row) => row.at(-1)),
          reference.map((row) => row.at(-1)),
        ),
      });
    }
    return {
      schema: "scratchgpt.cache.browser.v1",
      input: o,
      steps,
      cached_counts: cachedCounts,
      full_counts: fullCounts,
      comparison: {
        max_logit_delta: Math.max(...steps.map((s) => s.max_logit_delta)),
        max_kv_delta: Math.max(...steps.map((s) => s.max_kv_delta)),
        stale_shift_max_delta: Math.max(
          ...steps.map((s) => s.stale_shift_last_delta),
        ),
      },
      scope:
        "Independent scalar float64 cache and older independent full-forward reference. Native engine uses float32. native_float32_kv_bytes models only native KV payload, not browser memory. No training, speed, RSS or universal same-sample claim.",
    };
  }
  // Same strict transport grammar as AlignLab, with this audit's size/depth limits.
  function parseJSON(text) {
    if (new TextEncoder().encode(text).length > 2000000)
      throw Error("JSON input exceeds two million bytes");
    let depth = 0,
      quoted = false,
      escaped = false;
    for (const c of text) {
      if (quoted) {
        if (escaped) escaped = false;
        else if (c === "\\") escaped = true;
        else if (c === '"') quoted = false;
      } else if (c === '"') quoted = true;
      else if (c === "{" || c === "[") {
        if (++depth > 48) throw Error("JSON nesting exceeds 48");
      } else if (c === "}" || c === "]") depth--;
    }
    let i = 0;
    const ws = () => {
      while (/[ \t\r\n]/.test(text[i] ?? "") && i < text.length) i++;
    };
    function value() {
      ws();
      const c = text[i];
      if (c === "{") {
        i++;
        const o = {};
        ws();
        if (text[i] === "}") {
          i++;
          return o;
        }
        while (true) {
          ws();
          if (text[i] !== '"') throw Error("expected key");
          const k = string();
          if (Object.hasOwn(o, k)) throw Error("duplicate JSON key");
          ws();
          if (text[i++] !== ":") throw Error("expected colon");
          Object.defineProperty(o, k, {
            value: value(),
            enumerable: true,
            writable: true,
            configurable: true,
          });
          ws();
          const d = text[i++];
          if (d === "}") return o;
          if (d !== ",") throw Error("expected comma");
        }
      }
      if (c === "[") {
        i++;
        const a = [];
        ws();
        if (text[i] === "]") {
          i++;
          return a;
        }
        while (true) {
          a.push(value());
          ws();
          const d = text[i++];
          if (d === "]") return a;
          if (d !== ",") throw Error("expected comma");
        }
      }
      if (c === '"') return string();
      for (const [word, v] of [
        ["true", true],
        ["false", false],
        ["null", null],
      ])
        if (text.startsWith(word, i)) {
          i += word.length;
          return v;
        }
      const m = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(
        text.slice(i),
      );
      if (!m) throw Error("invalid JSON");
      i += m[0].length;
      const n = Number(m[0]);
      if (!Number.isFinite(n)) throw Error("nonfinite JSON number");
      return n;
    }
    function string() {
      const start = i++;
      while (i < text.length) {
        const c = text[i++];
        if (c === "\\") i++;
        else if (c === '"') return JSON.parse(text.slice(start, i));
      }
      throw Error("unterminated string");
    }
    const result = value();
    ws();
    if (i !== text.length) throw Error("trailing JSON");
    return result;
  }
  const api = { audit, validate, parseJSON };
  if (typeof module !== "undefined") module.exports = api;
  else root.CacheLab = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
