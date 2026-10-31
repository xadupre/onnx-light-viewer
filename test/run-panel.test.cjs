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
  const byId = (id) => dom.window.document.getElementById(id);
  byId("prompt-view").hidden = false;
  const messages = [];
  let promptActivations = 0;
  dom.window.eval(
    readFileSync(require.resolve("../media/run-panel.js"), "utf8"),
  );
  const api = dom.window.createRunPanel({
    byId,
    postMessage: (message) => messages.push(message),
    activatePrompt: () => ++promptActivations,
  });
  return { dom, byId, api, messages, activations: () => promptActivations };
}

test("Run window preserves Prompt and has accessible independent execution tabs", (t) => {
  const { dom, byId, messages, activations } = panel(t);
  assert.equal(byId("prompt-tab").textContent, "Run");
  assert.equal(byId("execution-prompt-view").hidden, false);
  assert.equal(byId("execution-run-view").hidden, true);
  byId("execution-run-tab").click();
  assert.equal(byId("execution-prompt-view").hidden, true);
  assert.equal(byId("execution-run-view").hidden, false);
  assert.equal(messages.at(-1).type, "run-inputs");
  byId("execution-run-tab").dispatchEvent(
    new dom.window.KeyboardEvent("keydown", {
      key: "ArrowLeft",
      bubbles: true,
    }),
  );
  assert.equal(byId("execution-prompt-view").hidden, false);
  assert.equal(byId("execution-run-tab").tabIndex, -1);
  assert.equal(dom.window.document.activeElement.id, "execution-prompt-tab");
  assert.equal(activations(), 1);
});

test("regular run sends exact text inputs, handles errors and ignores stale results", (t) => {
  const { byId, api, messages } = panel(t);
  api.activate();
  const requestId = messages.at(-1).requestId;
  api.handleMessage({
    type: "run-inputs",
    requestId,
    stages: [],
    path: [],
    inputs: [{ name: "<input>", text: 'int64[1] "<input>" = {}' }],
  });
  byId("run-input-0").value = 'int64[1] "<input>" = {9007199254740993}';
  byId("run-form").querySelector("button").click();
  const run = messages.at(-1);
  assert.equal(run.type, "run-model");
  assert.equal(run.inputs[0], byId("run-input-0").value);
  assert.equal(byId("run-fields").disabled, true);
  byId("run-form").querySelector("button").click();
  assert.equal(messages.at(-1), run);
  api.handleMessage({
    type: "run-error",
    requestId: run.requestId,
    error: "SDK error",
  });
  assert.equal(byId("run-fields").disabled, false);
  assert.equal(byId("run-status").textContent, "SDK error");
  byId("run-form").querySelector("button").click();
  const result = {
    timeMs: 1.25,
    outputs: [{ name: "<output>", text: "<script>alert(1)</script>" }],
  };
  api.handleMessage({
    type: "run-result",
    requestId: messages.at(-1).requestId,
    result,
  });
  assert.match(byId("run-status").textContent, /1.250 ms/);
  assert.equal(byId("run-results").querySelector("script"), null);
  assert.match(byId("run-results").textContent, /<script>/);
  api.reset();
  api.handleMessage({ type: "run-result", requestId: run.requestId, result });
  assert.equal(byId("run-results").textContent, "");
  assert.equal(byId("run-fields").disabled, true);
});

test("pipeline stage changes reload input forms without changing execution mode", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("execution-run-tab").click();
  const stages = ["embedding", "prefill"].map((name) => ({
    name,
    path: [{ pipeline: name }],
  }));
  api.handleMessage({
    type: "run-inputs",
    requestId: messages.at(-1).requestId,
    stages,
    path: stages[0].path,
    inputs: [],
  });
  byId("run-stage").value = JSON.stringify(stages[1].path);
  byId("run-stage").dispatchEvent(new dom.window.Event("change"));
  assert.equal(
    JSON.stringify(messages.at(-1).path),
    JSON.stringify(stages[1].path),
  );
  assert.equal(byId("execution-run-view").hidden, false);
});
