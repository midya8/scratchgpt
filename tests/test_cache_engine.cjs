"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const Cache = require("../docs/cache-engine.js");
const data = require("../docs/cache-witnesses.json");

function numeric(a, b, path = "root") {
  if (Array.isArray(a)) {
    assert.ok(Array.isArray(b), path);
    assert.equal(a.length, b.length, path);
    a.forEach((v, i) => numeric(v, b[i], `${path}/${i}`));
  } else {
    assert.equal(typeof a, "number", path);
    assert.equal(typeof b, "number", path);
    assert.ok(Number.isFinite(a) && Number.isFinite(b), path);
    assert.ok(
      Math.abs(a - b) <= 2e-6 * Math.max(1, Math.abs(a), Math.abs(b)),
      `${path}: ${a} / ${b}`,
    );
  }
}

test("all native cache, logit, mask and executed-count fields on every witness", () => {
  let coordinates = 0;
  for (const native of data.reports) {
    const browser = Cache.audit(native.input);
    assert.deepEqual(browser.input, native.input);
    assert.deepEqual(browser.cached_counts, native.cached_counts);
    assert.deepEqual(browser.full_counts, native.numpy_full_counts);
    assert.equal(browser.steps.length, native.steps.length);
    browser.steps.forEach((b, i) => {
      const n = native.steps[i];
      for (const key of [
        "mode",
        "consumed",
        "window_start",
        "context",
        "positions",
        "causal_mask",
        "cumulative_counts",
      ]) {
        assert.deepEqual(b[key], n[key], key);
      }
      assert.equal(b.native_float32_kv_bytes, n.cache_nbytes);
      numeric(b.cached_logits, n.cached_logits, "cached logits");
      numeric(b.reference_logits, n.original_logits, "original logits");
      numeric(b.full_engine_logits, n.numpy_full_logits, "full logits");
      numeric(
        b.stale_shift_last_logits,
        n.stale_shift_last_logits,
        "wrong shift",
      );
      assert.equal(b.cache.length, n.cache.length);
      b.cache.forEach((layer, l) => {
        assert.deepEqual(layer.shape, n.cache[l].shape);
        assert.deepEqual(
          b.full_engine_cache[l].shape,
          n.original_cache[l].shape,
        );
        for (const key of ["key", "value"]) {
          numeric(layer[key], n.cache[l][key], "KV");
          numeric(
            b.full_engine_cache[l][key],
            n.original_cache[l][key],
            "original KV",
          );
          coordinates += layer[key].flat(Infinity).length;
        }
      });
      numeric(b.max_logit_delta, n.max_logit_delta);
      numeric(b.max_kv_delta, n.max_kv_delta);
      numeric(b.full_reference_delta, n.numpy_full_delta);
      numeric(b.stale_shift_last_delta, n.stale_shift_last_delta);
    });
    numeric(
      browser.comparison.max_logit_delta,
      native.comparison.max_logit_delta,
    );
    numeric(browser.comparison.max_kv_delta, native.comparison.max_kv_delta);
    numeric(
      browser.comparison.stale_shift_max_delta,
      native.comparison.stale_shift_max_delta,
    );
  }
  assert.equal(coordinates, data.kv_coordinates);
  assert.ok(
    Cache.audit(data.reports[0].input).comparison.stale_shift_max_delta > 1e-5,
  );
});

test("all 32 stream partitions and no mutation or hidden native-result dependency", () => {
  const original = structuredClone(data.reports[0].input);
  for (let mask = 0; mask < 32; mask++) {
    const input = structuredClone(original);
    input.tokens = [[0, 1, 2, 3, 4, 0]];
    input.chunks = [];
    let last = 0;
    for (let cut = 1; cut < 6; cut++) {
      if (mask & (1 << (cut - 1))) {
        input.chunks.push(cut - last);
        last = cut;
      }
    }
    input.chunks.push(6 - last);
    const before = JSON.stringify(input),
      report = Cache.audit(input);
    assert.equal(JSON.stringify(input), before);
    assert.ok(report.comparison.max_logit_delta < 1e-12);
    assert.ok(report.comparison.max_kv_delta < 1e-12);
  }
  assert.deepEqual(data.reports[0].input, original);
  const text = fs.readFileSync(
    require.resolve("../docs/cache-engine.js"),
    "utf8",
  );
  assert.ok(!text.includes("cache-witnesses.json"));
});

test("strict duplicate-key, escaped-key, number, nesting and byte limits", () => {
  const valid = JSON.stringify(data.reports[0].input);
  assert.deepEqual(Cache.parseJSON(valid), data.reports[0].input);
  for (const bad of [
    '{"a":1,"a":2}',
    '{"a":1,"\\u0061":2}',
    '{"a":NaN}',
    '{"a":1e999}',
    '{"a":1}x',
    "[".repeat(49) + "0" + "]".repeat(49),
    " ".repeat(2000001),
  ]) {
    assert.throws(() => Cache.parseJSON(bad));
  }
  assert.deepEqual(Cache.parseJSON('{"a":"[{}]\\\""}'), { a: '[{}]"' });
  const p = Cache.parseJSON('{"__proto__":{"x":1}}');
  assert.equal(Object.hasOwn(p, "__proto__"), true);
  assert.equal({}.x, undefined);
});

test("invalid schemas and integer/weight contracts are rejected before cloning", () => {
  for (const [key, value] of [
    ["chunks", [true, 8]],
    ["chunks", [1]],
    ["tokens", [[1.5]]],
    ["tokens", [[true]]],
    ["tokens", [[99]]],
    ["weights", []],
  ]) {
    const input = structuredClone(data.reports[0].input);
    input[key] = value;
    assert.throws(() => Cache.audit(input));
  }
  const input = structuredClone(data.reports[0].input);
  input.extra = () => null;
  assert.throws(() => Cache.audit(input));
  delete input.extra;
  input.weights[0].values[0] = NaN;
  assert.throws(() => Cache.audit(input));
});
