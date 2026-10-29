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
      { text: "<img src=x>", id: 8, score: 2 },
      { text: " Paris", id: 9, score: 3 },
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
  assert.equal(rows.length, 14);
  for (const row of rows) assert.equal(row.cells.length, 3);
  assert.match(byId("prompt-results").textContent, /Coefficients Δ < 0.01/);
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
