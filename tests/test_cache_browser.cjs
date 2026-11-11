"use strict";
const test = require("node:test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  os = require("node:os"),
  path = require("node:path"),
  { spawn, execFileSync } = require("node:child_process"),
  { waitForDebugger } = require("./chrome_support.js");
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
function same(a, b, p = "root") {
  if (typeof a === "number") {
    assert.ok(
      Number.isFinite(a) &&
        Number.isFinite(b) &&
        Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b)),
      p,
    );
  } else if (a && typeof a === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), p);
    for (const k of Object.keys(a)) same(a[k], b[k], p + "." + k);
  } else assert.equal(a, b, p);
}
test(
  "actual Chrome cache masks, all logits and KV pages, trusted input and real downloads",
  { timeout: 180000 },
  async () => {
    const server = require("node:http").createServer((req, res) => {
      try {
        const file = path.join(
          __dirname,
          "../docs",
          req.url === "/" ? "cache.html" : decodeURIComponent(req.url),
        );
        res.setHeader(
          "Content-Type",
          {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
          }[path.extname(file)] || "text/plain",
        );
        res.end(fs.readFileSync(file));
      } catch (e) {
        res.statusCode = 404;
        res.end("not found");
      }
    });
    await new Promise((r) => server.listen(0, "127.0.0.1", r));
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), "scratch-cache-")),
      downloads = path.join(profile, "downloads");
    fs.mkdirSync(downloads);
    const browser = spawn(
      process.env.CHROME_BIN || "google-chrome",
      [
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-dev-shm-usage",
        "--remote-debugging-port=0",
        `--user-data-dir=${profile}`,
        "about:blank",
      ],
      { stdio: ["ignore", "ignore", "pipe"] },
    );
    let ws,
      diagnostics = "",
      startupError;
    browser.on("error", (e) => (startupError = e));
    browser.stderr.on(
      "data",
      (d) => (diagnostics = (diagnostics + d).slice(-4000)),
    );
    try {
      for (
        let i = 0;
        i < 1200 &&
        !startupError &&
        browser.exitCode === null &&
        !fs.existsSync(path.join(profile, "DevToolsActivePort"));
        i++
      )
        await delay(25);
      assert.ok(
        fs.existsSync(path.join(profile, "DevToolsActivePort")),
        startupError?.message || diagnostics,
      );
      const port = fs
          .readFileSync(path.join(profile, "DevToolsActivePort"), "utf8")
          .split("\n")[0],
        tab = await waitForDebugger(`http://127.0.0.1:${port}/json/list`, {
          alive: () => browser.exitCode === null && !startupError,
        });
      ws = new WebSocket(tab.webSocketDebuggerUrl);
      await new Promise((r, j) => {
        ws.onopen = r;
        ws.onerror = j;
      });
      let id = 0;
      const pending = new Map();
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data),
          p = pending.get(m.id);
        if (p) {
          pending.delete(m.id);
          m.error
            ? p.reject(Error(JSON.stringify(m.error)))
            : p.resolve(m.result);
        }
      };
      const call = (method, params = {}) =>
        new Promise((resolve, reject) => {
          const key = ++id;
          pending.set(key, { resolve, reject });
          ws.send(JSON.stringify({ id: key, method, params }));
        });
      const js = async (expression) => {
        const r = await call("Runtime.evaluate", {
          expression,
          returnByValue: true,
          awaitPromise: true,
        });
        if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
        return r.result.value;
      };
      await call("Page.enable");
      await call("Page.navigate", {
        url:
          process.env.CACHE_URL ||
          `http://127.0.0.1:${server.address().port}/cache.html`,
      });
      for (let i = 0; i < 500; i++) {
        if (await js("Boolean(window.cacheReady)")) break;
        await delay(20);
      }
      assert.ok(await js("window.cacheReady"));
      const data = require("../docs/cache-witnesses.json");
      const Cache = require("../docs/cache-engine.js");
      const fmt = (x) => Number(x).toPrecision(8);
      const set = async (id, value, event = "change") =>
        js(
          `document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(String(value))};document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event(${JSON.stringify(event)}))`,
        );
      const table = async (id) =>
        js(
          `Array.from(document.querySelectorAll('#'+${JSON.stringify(id)}+' tbody tr'),tr=>Array.from(tr.cells,c=>c.textContent))`,
        );
      const flat = (values, prefix = []) =>
        values.flatMap((v, i) =>
          Array.isArray(v)
            ? flat(v, [...prefix, i])
            : [[[...prefix, i].join("/"), v]],
        );
      let coordinates = 0,
        logits = 0,
        steps = 0;
      for (let ri = 0; ri < data.reports.length; ri++) {
        await set("preset", ri);
        const expected = Cache.audit(data.reports[ri].input),
          actual = await js("window.cacheReport");
        same(actual, expected);
        assert.match(
          await js('document.getElementById("native-status").textContent'),
          /Matching full native witness/,
        );
        for (let si = 0; si < expected.steps.length; si++) {
          await set("step", si);
          const s = actual.steps[si];
          steps++;
          assert.deepEqual(
            await table("mask"),
            s.causal_mask.map((row, i) => [String(i), ...row.map(String)]),
          );
          const expectedLogits = s.cached_logits.flatMap((batch, b) =>
            batch.flatMap((row, p) =>
              row.map((value, t) => [
                String(b),
                String(p),
                String(t),
                fmt(value),
                fmt(s.reference_logits[b][p][t]),
                fmt(value - s.reference_logits[b][p][t]),
              ]),
            ),
          );
          assert.deepEqual(await table("logits"), expectedLogits);
          logits += expectedLogits.length;
          for (let li = 0; li < s.cache.length; li++) {
            await set("layer", li);
            const a = s.cache[li],
              b = s.full_engine_cache[li],
              keys = flat(a.key),
              values = flat(a.value),
              refKeys = flat(b.key),
              refValues = flat(b.value);
            for (let start = 0; start < keys.length; start += 32) {
              const end = Math.min(start + 32, keys.length);
              assert.deepEqual(
                await table("kv"),
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
              coordinates += (end - start) * 2;
              if (end < keys.length)
                await js('document.getElementById("next").click()');
            }
          }
        }
        for (const width of [320, 390, 1280]) {
          await call("Emulation.setDeviceMetricsOverride", {
            width,
            height: 1000,
            deviceScaleFactor: 1,
            mobile: width < 700,
          });
          assert.ok(
            await js("document.documentElement.scrollWidth<=innerWidth"),
            "overflow " + ri + " " + width,
          );
        }
      }
      assert.equal(coordinates, data.kv_coordinates);
      assert.equal(logits, data.logit_coordinates);
      assert.equal(steps, data.steps);
      assert.equal(
        await js('document.querySelectorAll("#benchmark tbody tr").length'),
        4,
      );
      assert.match(
        await js('document.getElementById("benchmark-info").textContent'),
        /Python 3.10.12/,
      );
      await set("preset", 0);
      await js(
        'document.getElementById("tokens").focus();document.getElementById("tokens").select()',
      );
      await call("Input.insertText", { text: "[[0,1,2,3,4,0]]" });
      assert.equal(await js('document.getElementById("results").hidden'), true);
      assert.equal(
        await js('document.getElementById("download-input").disabled'),
        true,
      );
      assert.equal(await js("window.cacheReport"), null);
      await js(
        'document.querySelector("#audit-form button[type=submit]").click()',
      );
      assert.equal(await js('document.getElementById("results").hidden'), true);
      assert.ok(
        await js('document.getElementById("error").textContent.length>0'),
      );
      await js(
        'document.getElementById("chunks").focus();document.getElementById("chunks").select()',
      );
      await call("Input.insertText", { text: "[2,1,3]" });
      await js(
        'document.querySelector("#audit-form button[type=submit]").click()',
      );
      assert.equal(
        await js('document.getElementById("results").hidden'),
        false,
      );
      assert.match(
        await js('document.getElementById("native-status").textContent'),
        /Edited input/,
      );
      let edited = await js("window.cacheReport");
      same(edited, Cache.audit(edited.input));
      const imported = structuredClone(edited.input);
      imported.weights[0].values[0] += 0.001;
      await js(
        'document.getElementById("input-json").closest("details").open=true',
      );
      await set("input-json", JSON.stringify(imported), "input");
      assert.equal(await js("window.cacheReport"), null);
      await js('document.getElementById("import").click()');
      assert.equal(
        await js('document.getElementById("results").hidden'),
        false,
      );
      assert.match(
        await js('document.getElementById("native-status").textContent'),
        /Edited input/,
      );
      edited = await js("window.cacheReport");
      assert.deepEqual(edited.input, imported);
      same(edited, Cache.audit(imported));

      await call("Browser.setDownloadBehavior", {
        behavior: "allow",
        downloadPath: downloads,
      });
      for (const [id, name] of [
        ["download-input", "scratchgpt-cache-input.json"],
        ["download-report", "scratchgpt-cache-browser-report.json"],
      ]) {
        const box = await js(
          `(()=>{const e=document.getElementById(${JSON.stringify(id)});e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`,
        );
        await call("Input.dispatchMouseEvent", {
          type: "mousePressed",
          button: "left",
          clickCount: 1,
          ...box,
        });
        await call("Input.dispatchMouseEvent", {
          type: "mouseReleased",
          button: "left",
          clickCount: 1,
          ...box,
        });
        const file = path.join(downloads, name);
        for (let i = 0; i < 200 && !fs.existsSync(file); i++) await delay(25);
        assert.ok(fs.existsSync(file), "real download " + name);
        assert.deepEqual(
          JSON.parse(fs.readFileSync(file)),
          id === "download-input" ? edited.input : edited,
        );
      }
      const native = JSON.parse(
        execFileSync(
          process.env.CACHE_PYTHON || "python3",
          [
            "-m",
            "scratchgpt.cache_audit",
            path.join(downloads, "scratchgpt-cache-input.json"),
          ],
          {
            cwd: path.join(__dirname, ".."),
            env: { ...process.env, OPENBLAS_NUM_THREADS: "1" },
          },
        ),
      );
      for (let i = 0; i < edited.steps.length; i++) {
        const a = edited.steps[i].cached_logits.flat(Infinity),
          b = native.steps[i].cached_logits.flat(Infinity);
        assert.equal(a.length, b.length);
        a.forEach((v, j) => assert.ok(Math.abs(v - b[j]) < 2e-6));
      }
      for (const [id, value] of [
        ["tokens", "[]"],
        ["chunks", "[true]"],
        ["tokens", "[[99]]"],
      ]) {
        await set("preset", 0);
        await set(id, value, "input");
        assert.equal(await js("window.cacheReport"), null);
        await js(
          'document.querySelector("#audit-form button[type=submit]").click()',
        );
        assert.equal(
          await js('document.getElementById("results").hidden'),
          true,
        );
        assert.ok(
          await js('document.getElementById("error").textContent.length>0'),
        );
      }
      await set("preset", 0);
      await js(
        'document.getElementById("input-json").closest("details").open=true',
      );
      await set("input-json", '{"schema":1,"\\u0073chema":2}', "input");
      await js('document.getElementById("import").click()');
      assert.equal(await js("window.cacheReport"), null);
      assert.match(
        await js('document.getElementById("error").textContent'),
        /duplicate/i,
      );
      await set("preset", 0);
      if (process.env.CACHE_SCREENSHOT) {
        const width = Number(process.env.CACHE_WIDTH || 1280);
        await call("Emulation.setDeviceMetricsOverride", {
          width,
          height: 1100,
          deviceScaleFactor: 1,
          mobile: width < 700,
        });
        await set("step", 4);
        await js(
          'document.querySelectorAll("details").forEach(e=>e.open=false);document.activeElement.blur();window.scrollTo(0,0)',
        );
        const m = await call("Page.getLayoutMetrics"),
          image = await call("Page.captureScreenshot", {
            format: "png",
            captureBeyondViewport: true,
            clip: {
              x: 0,
              y: 0,
              width,
              height: m.cssContentSize.height,
              scale: 1,
            },
          });
        fs.writeFileSync(
          process.env.CACHE_SCREENSHOT,
          Buffer.from(image.data, "base64"),
        );
      }
      console.log(
        JSON.stringify({
          native_reports: data.reports.length,
          steps,
          logits,
          kv_coordinates: coordinates,
          layouts: data.reports.length * 3,
          real_downloads: 2,
          downloaded_input_native_replay: true,
        }),
      );
    } finally {
      server.close();
      if (ws) ws.close();
      if (browser.pid && browser.exitCode === null) {
        browser.kill("SIGTERM");
        await new Promise((r) => browser.once("exit", r));
      }
      await fs.promises.rm(profile, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 200,
      });
    }
  },
);
