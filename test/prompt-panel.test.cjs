const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const { JSDOM } = require("jsdom");
const { webviewHtml } = require("../dist/webview");

function panel(t) {
  const dom = new JSDOM(
    webviewHtml((name) => name, "viewer:", "nonce", "test"),
    { runScripts: "outside-only" },
  );
  t.after(() => dom.window.close());
  const messages = [];
  dom.window.eval(
    readFileSync(require.resolve("../media/prompt-panel.js"), "utf8"),
  );
  const byId = (id) => dom.window.document.getElementById(id);
  const api = dom.window.createPromptPanel({
    byId,
    postMessage: (message) => messages.push(message),
  });
  api.load();
  api.handleMessage({
    type: "prompt-info",
    requestId: messages.at(-1).requestId,
    info: { available: true, engines: ["onnx-light"] },
  });
  return { dom, byId, api, messages };
}

test("prompt defaults, engine applicability and aligned validation results", (t) => {
  const { dom, byId, api, messages } = panel(t);
  assert.equal(byId("prompt-fields").disabled, false);
  assert.equal(byId("prompt-max-tokens").value, "32");
  assert.equal(byId("prompt-engine").options[1].disabled, true);
  byId("prompt-text").value = "<script>not executable</script>";
  byId("prompt-form").dispatchEvent(
    new dom.window.Event("submit", { cancelable: true }),
  );
  const run = messages.at(-1);
  assert.equal(run.options.maxTokens, 32);
  assert.equal(run.options.prompt, byId("prompt-text").value);
  assert.equal(byId("prompt-fields").disabled, true);
  const result = {
    prompt: run.options.prompt,
    engine: "onnx-light",
    tokens: [
      { text: "<img src=x>", id: 8, score: 2, timeMs: 120 },
      { text: " Paris", id: 9, score: 3, timeMs: 4 },
    ],
  };
  api.handleMessage({
    type: "prompt-result",
    requestId: run.requestId,
    result,
  });
  assert.equal(byId("prompt-validate").hidden, false);
  assert.equal(byId("prompt-results").querySelector("img"), null);
  assert.equal(byId("prompt-results").querySelector("script"), null);
  byId("prompt-text").value = "Changed but not run";
  byId("prompt-validate").click();
  const validation = messages.at(-1);
  assert.equal(validation.options.validate, true);
  assert.equal(validation.options.prompt, run.options.prompt);
  api.handleMessage({
    type: "prompt-result",
    requestId: validation.requestId,
    result: {
      ...result,
      reference: { tokens: result.tokens },
      discrepancies: [1, 2].map(() => ({
        maxAbsolute: 0.02,
        relative: 0.01,
        l2: 0.03,
        below001: 2,
        below01: 3,
        below1: 4,
        total: 4,
      })),
    },
  });
  const rows = byId("prompt-results").querySelectorAll("tr");
  assert.equal(rows.length, 17);
  for (const row of rows) assert.equal(row.cells.length, 3);
  assert.match(byId("prompt-results").textContent, /Coefficients Δ < 0.01/);
  assert.match(
    byId("prompt-results").textContent,
    /prefill 120.000 ms; decode mean 4.000 ms/,
  );
});

test("appends tokens and updates decode mean before completion without replacing the table", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  const requestId = messages.at(-1).requestId;
  const table = byId("prompt-results").querySelector("table");
  for (const [index, timeMs] of [100, 2, 6].entries()) {
    api.handleMessage({
      type: "prompt-token",
      requestId,
      index,
      token: {
        text: index === 0 ? "<script>" : " token",
        id: index,
        score: 1,
        timeMs,
      },
    });
    assert.equal(byId("prompt-results").querySelector("table"), table);
    for (const row of table.rows) assert.equal(row.cells.length, index + 2);
    assert.equal(byId("prompt-validate").hidden, true);
  }
  assert.match(
    byId("prompt-results").textContent,
    /prefill 100.000 ms; decode mean 4.000 ms \(2 decode tokens\)/,
  );
  assert.match(table.textContent, /PrefillDecodeDecode/);
  assert.equal(byId("prompt-results").querySelector("script"), null);
  assert.match(byId("prompt-status").textContent, /Generated 3\/32 tokens/);
  byId("prompt-cancel").click();
  api.handleMessage({
    type: "prompt-token",
    requestId,
    index: 3,
    token: { text: "stale", id: 3, score: 1, timeMs: 1 },
  });
  assert.doesNotMatch(table.textContent, /stale/);
});

test("engine settings pages retain independent controls, including parallel loading", (t) => {
  const { dom, byId, api, messages } = panel(t);
  const field = (page, name) =>
    byId(page).querySelector(`[data-setting="${name}"]`);
  byId("prompt-settings-open").click();
  assert.equal(byId("prompt-session-page").hidden, false);
  assert.equal(byId("prompt-settings-light").hidden, false);
  field("prompt-settings-light", "threads").value = "2";
  field("prompt-settings-light", "sdkLoader").value = "true";
  field("prompt-settings-light", "sdkLoader").dispatchEvent(
    new dom.window.Event("change", { bubbles: true }),
  );
  field("prompt-settings-light", "loadOverlap").value = "false";
  field("prompt-settings-light", "loadMemoryMiB").value = "512";
  byId("prompt-keep-loaded").checked = true;
  byId("prompt-settings-back").click();
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  const first = messages.at(-1);
  assert.equal(first.options.keepLoaded, true);
  assert.deepEqual(JSON.parse(JSON.stringify(first.options.sessionSettings)), {
    threads: 2,
    sdkLoader: true,
    loadOverlap: false,
    loadMemoryMiB: 512,
    profile: false,
  });
  assert.equal(byId("prompt-session-fields").disabled, true);
  byId("prompt-cancel").click();
  api.handleMessage({
    type: "prompt-info",
    requestId: first.requestId + 1,
    info: { available: true, engines: ["onnx-light", "onnxruntime-genai"] },
  });
  byId("prompt-engine").value = "onnxruntime-genai";
  byId("prompt-settings-open").click();
  assert.equal(byId("prompt-settings-genai").hidden, false);
  const provider = field("prompt-settings-genai", "provider");
  provider.value = "cuda";
  provider.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  assert.equal(field("prompt-settings-genai", "deviceId").disabled, false);
  field("prompt-settings-genai", "deviceId").value = "1";
  field("prompt-settings-genai", "gpuMemoryMiB").value = "256";
  field("prompt-settings-genai", "prepacking").value = "false";
  byId("prompt-settings-back").click();
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  const second = messages.at(-1);
  assert.equal(second.options.engine, "onnxruntime-genai");
  assert.equal(second.options.sessionSettings.provider, "cuda");
  assert.equal(second.options.sessionSettings.deviceId, 1);
  assert.equal(second.options.sessionSettings.prepacking, false);
  assert.equal(second.options.sessionSettings.threads, undefined);
  assert.equal(field("prompt-settings-light", "threads").value, "2");
});

test("loading and warmup appear before tokens, and unloading updates retained state", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  const requestId = messages.at(-1).requestId;
  api.handleMessage({
    type: "prompt-lifecycle",
    requestId,
    lifecycle: { loadMs: 12, reused: false, retained: true },
  });
  assert.match(byId("prompt-lifecycle").textContent, /loading: 12.000 ms/);
  assert.match(byId("prompt-lifecycle").textContent, /warmup.*running/);
  api.handleMessage({
    type: "prompt-lifecycle",
    requestId,
    lifecycle: { loadMs: 12, warmupMs: 3, reused: true, retained: true },
  });
  assert.match(byId("prompt-lifecycle").textContent, /Reused model; initial/);
  assert.match(byId("prompt-lifecycle").textContent, /warmup.*3.000 ms/);
  byId("prompt-cancel").click();
  assert.match(byId("prompt-lifecycle").textContent, /Unloaded/);
  api.handleMessage({
    type: "prompt-lifecycle",
    requestId,
    lifecycle: { loadMs: 999, reused: false, retained: true },
  });
  assert.doesNotMatch(byId("prompt-lifecycle").textContent, /999/);
});

test("invalid settings stop generation and pipeline-only loading limitations are explicit", (t) => {
  const { dom, byId, api, messages } = panel(t);
  const threads = byId("prompt-settings-light").querySelector(
    '[data-setting="threads"]',
  );
  threads.value = "-1";
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  assert.equal(
    messages.some((message) => message.type === "run-prompt"),
    false,
  );
  assert.equal(byId("prompt-session-page").hidden, false);
  threads.value = "1";
  api.handleMessage({
    type: "prompt-info",
    requestId: messages[0].requestId,
    info: { available: true, engines: ["onnx-light"], pipeline: true },
  });
  const overlap = byId("prompt-settings-light").querySelector(
    '[data-setting="loadOverlap"]',
  );
  assert.equal(overlap.disabled, true);
  assert.equal(byId("prompt-pipeline-loading").hidden, false);
  overlap.value = "false";
  byId("prompt-settings-back").click();
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  assert.equal(messages.at(-1).options.sessionSettings.loadOverlap, undefined);
});

test("a single prefill token has no decode mean and missing SDK timings are not fabricated", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  const requestId = messages.at(-1).requestId;
  api.handleMessage({
    type: "prompt-token",
    requestId,
    index: 0,
    token: { text: "first", id: 1, score: 1, timeMs: 12 },
  });
  assert.match(
    byId("prompt-results").textContent,
    /decode mean N\/A \(no decode tokens\)/,
  );
  api.handleMessage({
    type: "prompt-token",
    requestId,
    index: 1,
    token: { text: "second", id: 2, score: null, timeMs: null },
  });
  assert.match(
    byId("prompt-results").textContent,
    /decode mean Unavailable \(SDK\)/,
  );
});

test("prompt cancellation and model replacement reject stale results and errors", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  const requestId = messages.at(-1).requestId;
  byId("prompt-cancel").click();
  assert.equal(messages.at(-1).type, "prompt-cancel");
  api.handleMessage({
    type: "prompt-error",
    requestId,
    error: "stale failure",
  });
  assert.equal(byId("prompt-status").textContent, "Cancelled.");
  api.reset();
  api.handleMessage({ type: "prompt-result", requestId, result: {} });
  assert.equal(byId("prompt-results").childElementCount, 0);
  assert.equal(byId("prompt-fields").disabled, true);
});

test("SDK score limitations are explicit rather than fabricated", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  api.handleMessage({
    type: "prompt-result",
    requestId: messages.at(-1).requestId,
    result: {
      prompt: "Hello",
      tokens: [{ text: " world", id: 1, score: null }],
      limitations: ["The SDK does not expose generation logits."],
    },
  });
  assert.match(byId("prompt-results").textContent, /Unavailable \(SDK\)/);
  assert.match(byId("prompt-results").textContent, /does not expose/);
});
