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
  byId("run-form").querySelector('button[type="submit"]').click();
  const run = messages.at(-1);
  assert.equal(run.type, "run-model");
  assert.equal(run.inputs[0], byId("run-input-0").value);
  assert.equal(byId("run-fields").disabled, true);
  byId("run-form").querySelector('button[type="submit"]').click();
  assert.equal(messages.at(-1), run);
  api.handleMessage({
    type: "run-error",
    requestId: run.requestId,
    error: "SDK error",
  });
  assert.equal(byId("run-fields").disabled, false);
  assert.equal(byId("run-status").textContent, "SDK error");
  byId("run-form").querySelector('button[type="submit"]').click();
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
  assert.match(byId("run-results").textContent, /<script>/i);
  api.reset();
  api.handleMessage({ type: "run-result", requestId: run.requestId, result });
  assert.equal(byId("run-results").textContent, "");
  assert.equal(byId("run-fields").disabled, true);
});

test("Run shows one-shot session lifecycle, actual inputs and save/reload controls", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("run-engine").value = "onnxruntime";
  byId("run-engine").dispatchEvent(new dom.window.Event("change"));
  assert.match(
    byId("run-lifecycle").textContent,
    /ONNX Runtime.*no retained session/,
  );
  api.activate();
  api.handleMessage({
    type: "run-inputs",
    requestId: messages.at(-1).requestId,
    stages: [],
    path: [],
    inputs: [{ name: "X", text: "int64[1] X = {}", shape: ["1"] }],
  });

  assert.equal(byId("run-metadata-0").textContent, "int64[1] X = {}");
  const submit = byId("run-form").querySelector('button[type="submit"]');
  submit.click();
  const requestId = messages.at(-1).requestId;
  assert.match(byId("run-lifecycle").textContent, /run in progress/);
  const text = "int64[1] X\nValues: [9007199254740993]";
  api.handleMessage({
    type: "run-result",
    requestId,
    result: {
      timeMs: 1,
      lifecycle: { loadMs: 2, reused: false, retained: false },
      inputs: [{ name: "X", text }],
      outputs: [{ name: "Y", text, canSave: true }],
    },
  });
  assert.equal(byId("run-used-inputs").hidden, false);
  assert.equal(
    byId("run-input-previews").querySelector("pre").textContent,
    text,
  );
  assert.match(
    byId("run-lifecycle").textContent,
    /created.*released.*2\.000.*No session is kept alive/,
  );
  byId("run-input-previews").querySelector("button").click();
  assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1))), {
    type: "run-save-input",
    requestId,
    index: 0,
  });
  api.handleMessage({
    type: "run-input-saved",
    requestId,
    name: "X",
    path: "/tmp/exact.pb",
  });
  assert.equal(byId("run-file-0").value, "/tmp/exact.pb");
  assert.equal(byId("run-load-0").hidden, false);
  byId("run-results").querySelector("button").click();
  assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1))), {
    type: "run-save-output",
    requestId,
    index: 0,
  });
  api.handleMessage({
    type: "run-output-saved",
    requestId,
    name: "Y",
    path: "/tmp/output.pb",
  });
  assert.match(byId("run-status").textContent, /Saved output Y/);
  byId("run-load-0").click();
  assert.equal(messages.at(-1).type, "run-file");
  api.handleMessage({ ...messages.at(-1), path: "/tmp/exact.pb" });
  assert.equal(byId("run-source-0").value, "protobuf");
  assert.equal(byId("run-file-0").disabled, false);
  submit.click();
  assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).inputs)), [
    { name: "X", source: "protobuf", path: "/tmp/exact.pb" },
  ]);
  assert.equal(byId("run-used-inputs").hidden, true);
  api.handleMessage({
    type: "run-input-saved",
    requestId,
    name: "X",
    path: "/tmp/stale.pb",
  });
  assert.equal(byId("run-file-0").value, "/tmp/exact.pb");
  api.reset();
  assert.equal(byId("run-input-previews").textContent, "");
  assert.match(byId("run-lifecycle").textContent, /no retained session/);
});

test("Run offers opt-in reuse, truthful session status and correlated unloading", (t) => {
  const { dom, byId, api, messages } = panel(t);
  api.activate();
  api.handleMessage({
    type: "run-inputs",
    requestId: messages.at(-1).requestId,
    stages: [],
    path: [],
    inputs: [{ name: "X", text: "float[1] X = {1}" }],
  });
  assert.equal(byId("run-keep-loaded").checked, false);
  byId("run-keep-loaded").checked = true;
  byId("run-keep-loaded").dispatchEvent(new dom.window.Event("change"));
  const submit = byId("run-form").querySelector('button[type="submit"]');
  for (const reused of [false, true]) {
    submit.click();
    assert.equal(messages.at(-1).keepLoaded, true);
    api.handleMessage({
      type: "run-result",
      requestId: messages.at(-1).requestId,
      result: {
        timeMs: 1,
        inputs: [],
        outputs: [],
        lifecycle: { loadMs: 2, reused, retained: true },
      },
    });
    assert.match(byId("run-lifecycle").textContent, /still loaded.*next run/);
    assert.match(
      byId("run-lifecycle").textContent,
      reused ? /reused from the previous run/ : /created for this run/,
    );
    assert.equal(byId("run-unload").disabled, false);
  }
  byId("run-unload").click();
  const release = messages.at(-1);
  assert.equal(release.type, "run-release");
  api.handleMessage({ ...release, type: "run-session-released" });
  assert.match(byId("run-lifecycle").textContent, /no retained session/);
  assert.equal(byId("run-unload").disabled, true);
  submit.click();
  api.handleMessage({
    ...release,
    type: "run-session-error",
    error: "stale release",
  });
  assert.doesNotMatch(byId("run-status").textContent, /stale release/);
});

for (const control of [
  "run-engine",
  "run-verbose",
  "run-log-severity",
  "run-log-verbosity",
  "run-keep-loaded",
])
  test(`Run releases the retained session when ${control} changes`, (t) => {
    const { dom, byId, api, messages } = panel(t);
    api.activate();
    api.handleMessage({
      type: "run-inputs",
      requestId: messages.at(-1).requestId,
      stages: [],
      path: [],
      inputs: [],
    });
    api.handleMessage({
      type: "run-result",
      requestId: messages.at(-1).requestId,
      result: {
        timeMs: 1,
        outputs: [],
        lifecycle: { loadMs: 1, reused: false, retained: true },
      },
    });
    byId(control).dispatchEvent(new dom.window.Event("change"));
    assert.equal(messages.at(-1).type, "run-release");
    assert.equal(byId("run-unload").disabled, true);
  });

test("Run updates the retained session status after an idle worker stops", (t) => {
  const { byId, api, messages } = panel(t);
  api.activate();
  api.handleMessage({
    type: "run-inputs",
    requestId: messages.at(-1).requestId,
    stages: [],
    path: [],
    inputs: [],
  });
  api.handleMessage({
    type: "run-result",
    requestId: messages.at(-1).requestId,
    result: {
      timeMs: 1,
      outputs: [],
      lifecycle: { loadMs: 1, reused: true, retained: true },
    },
  });
  api.handleMessage({ type: "run-session-lost", error: "Worker exited." });
  assert.match(byId("run-lifecycle").textContent, /no retained session/);
  assert.equal(byId("run-unload").disabled, true);
  assert.match(byId("run-status").textContent, /Worker exited/);
});

for (const type of ["metadata-error", "shape-error"])
  test(`Run recovers from ${type} without stale messages or lost input drafts`, (t) => {
    const { byId, api, messages } = panel(t);
    const failure = { type, error: "Model analysis failed." };
    api.activate();
    const pendingInputs = messages.at(-1);
    assert.equal(api.handleMessage(failure), false);
    api.activate();
    const freshInputs = messages.at(-1);
    assert.ok(freshInputs.requestId > pendingInputs.requestId);
    api.handleMessage({
      ...pendingInputs,
      stages: [],
      path: [],
      inputs: [],
    });
    assert.equal(byId("run-fields").disabled, true);
    api.handleMessage({
      ...freshInputs,
      stages: [],
      path: [],
      inputs: [{ name: "X", text: "float[1] X = {}" }],
    });
    const draft = "float[1] X = {-1}";
    byId("run-input-0").value = draft;
    const submit = byId("run-form").querySelector('button[type="submit"]');
    submit.click();
    const run = messages.at(-1);
    assert.equal(run.type, "run-model");
    assert.equal(byId("run-fields").disabled, true);
    assert.equal(api.handleMessage(failure), false);
    assert.equal(byId("run-fields").disabled, false);
    assert.equal(byId("run-input-0").value, draft);
    const status = byId("run-status").textContent;
    assert.match(status, /cancelled.*Model analysis failed/);
    for (const type of ["run-result", "run-error", "run-log"])
      api.handleMessage({
        type,
        requestId: run.requestId,
        result: { timeMs: 1, outputs: [{ name: "old", text: "old" }] },
        error: "old",
        output: "old",
      });
    assert.equal(byId("run-status").textContent, status);
    assert.equal(byId("run-results").textContent, "");
    assert.equal(byId("run-log").textContent, "");
    submit.click();
    assert.equal(messages.at(-1).type, "run-model");
    assert.ok(messages.at(-1).requestId > run.requestId);
    assert.equal(messages.at(-1).inputs[0], draft);
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

test("Run requires dimensions, supports mixed input sources and retains engine logging", (t) => {
  const { dom, byId, api, messages } = panel(t);
  api.activate();
  api.handleMessage({
    type: "run-inputs",
    requestId: messages.at(-1).requestId,
    stages: [],
    path: [],
    inputs: [
      { name: "X", text: "float[batch,?] X = {}", shape: ["batch", null] },
      { name: "Y", text: 'float["3"] Y = {}', shape: ["3"], dynamicAxes: [0] },
      { name: "Z", text: "float[batch] Z = {}", shape: ["batch"] },
    ],
  });
  assert.equal(byId("run-dimensions").querySelectorAll("input").length, 3);
  assert.equal(byId("run-source-0").value, "random");
  const submit = byId("run-form").querySelector('button[type="submit"]');
  const before = messages.length;
  submit.click();
  assert.equal(messages.length, before);
  byId("run-dimension-0").value = "2";
  byId("run-dimension-1").value = "3";
  byId("run-dimension-2").value = "2";
  byId("run-verbose").value = "4";
  byId("run-engine").value = "onnxruntime";
  byId("run-engine").dispatchEvent(new dom.window.Event("change"));
  assert.equal(byId("run-logging-light").hidden, true);
  assert.equal(byId("run-verbose").disabled, true);
  byId("run-log-severity").value = "0";
  byId("run-log-verbosity").value = "2";
  byId("run-source-1").value = "protobuf";
  byId("run-source-1").dispatchEvent(new dom.window.Event("change"));
  byId("run-file-1").parentElement.querySelector("button").click();
  assert.equal(messages.at(-1).type, "run-file");
  api.handleMessage({ ...messages.at(-1), path: "/tmp/input.pb" });
  submit.click();
  const run = messages.at(-1);
  assert.equal(run.engine, "onnxruntime");
  assert.equal(
    JSON.stringify(run.dimensions),
    JSON.stringify({
      "symbol:batch": 2,
      "axis:X:1": 3,
      "symbol:3": 2,
    }),
  );
  assert.equal(
    JSON.stringify(run.settings),
    JSON.stringify({ logSeverity: 0, logVerbosity: 2 }),
  );
  assert.equal(
    JSON.stringify(run.inputs),
    JSON.stringify([
      { name: "X", source: "random" },
      { name: "Y", source: "protobuf", path: "/tmp/input.pb" },
      { name: "Z", source: "random" },
    ]),
  );
  api.handleMessage({
    type: "run-log",
    requestId: run.requestId,
    output: "<engine log>",
  });
  assert.equal(byId("run-fields").disabled, true);
  assert.equal(byId("run-log").textContent, "<engine log>");
  api.handleMessage({
    type: "run-error",
    requestId: run.requestId,
    error: "error",
  });
  byId("run-engine").value = "onnx-light";
  byId("run-engine").dispatchEvent(new dom.window.Event("change"));
  assert.equal(byId("run-verbose").value, "4");
  assert.equal(byId("run-log-verbosity").value, "2");
});

test("symbolic and unnamed dynamic dimensions with colliding labels remain independent", (t) => {
  const { byId, api, messages } = panel(t);
  api.activate();
  api.handleMessage({
    type: "run-inputs",
    requestId: messages.at(-1).requestId,
    stages: [],
    path: [],
    inputs: [
      {
        name: "X",
        text: "float[X:1,?] X = {}",
        shape: ["X:1", null],
        dynamicAxes: [0, 1],
      },
    ],
  });
  const fields = [...byId("run-dimensions").querySelectorAll("input")];
  assert.equal(fields.length, 2);
  assert.deepEqual(
    fields.map((field) => field.dataset.dimension),
    ["symbol:X:1", "axis:X:1"],
  );
  fields[0].value = "2";
  fields[1].value = "3";
  byId("run-form").querySelector('button[type="submit"]').click();
  assert.equal(
    JSON.stringify(messages.at(-1).dimensions),
    JSON.stringify({ "symbol:X:1": 2, "axis:X:1": 3 }),
  );
});
