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
  const { byId, api, messages } = panel(t);
  assert.equal(byId("prompt-tab").textContent, "Prompt");
  assert.ok(byId("prompt-tab").querySelector("svg"));
  assert.equal(byId("prompt-view").querySelector("h1").textContent, "Prompt");
  assert.equal(byId("prompt-fields").disabled, false);
  assert.equal(byId("prompt-max-tokens").value, "32");
  assert.equal(byId("prompt-engine").options[1].disabled, true);
  byId("prompt-text").value = "<script>not executable</script>";
  const runButton = byId("prompt-form").querySelector('button[type="submit"]');
  assert.ok(runButton.querySelector("svg"));
  assert.equal(runButton.textContent, "");
  assert.equal(runButton.getAttribute("aria-label"), "Run prompt");
  runButton.click();
  const run = messages.at(-1);
  assert.equal(run.options.engine, "onnx-light");
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
  assert.equal(validation.reuseResult, true);
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
  assert.equal(rows.length, 20);
  assert.equal(
    byId("prompt-results").querySelector(".prompt-token-mismatch"),
    null,
  );
  for (const row of rows)
    assert.equal(
      row.cells.length,
      row.classList.contains("prompt-token-section") ? 1 : 3,
    );
  assert.match(byId("prompt-results").textContent, /Coefficients Δ < 0.01/);
  assert.match(
    byId("prompt-results").textContent,
    /prefill 120.000 ms; decode mean 4.000 ms/,
  );
});

test("separates ONNX and Transformers and marks whole columns by token ID, not text or scores", (t) => {
  const { byId, api, messages } = panel(t);
  byId("prompt-form").querySelector('button[type="submit"]').click();
  const requestId = messages.at(-1).requestId;
  const tokens = [
    { text: " same", id: 1, score: 2, timeMs: 3 },
    { text: " same", id: 2, score: 2, timeMs: 3 },
    { text: " last", id: 3, score: 2, timeMs: 3 },
  ];
  api.handleMessage({
    type: "prompt-result",
    requestId,
    result: { prompt: "Hello", engine: "onnx-light", tokens },
  });
  assert.equal(byId("prompt-results").querySelector("caption"), null);
  assert.equal(
    byId("prompt-results").querySelector(".prompt-token-mismatch"),
    null,
  );
  for (const referenceTokens of [
    [],
    [
      { ...tokens[0], text: "different decoding", score: 9 },
      { ...tokens[1], id: 99 },
    ],
    [
      { ...tokens[0], text: "different decoding", score: 9 },
      { ...tokens[1], id: 99 },
      tokens[2],
      { ...tokens[2], id: 4 },
    ],
  ]) {
    byId("prompt-validate").click();
    api.handleMessage({
      type: "prompt-result",
      requestId: messages.at(-1).requestId,
      result: {
        prompt: "Hello",
        engine: "onnx-light",
        tokens,
        reference: { tokens: referenceTokens },
        discrepancies: tokens.map(() => ({ maxAbsolute: 0 })),
      },
    });
    const table = byId("prompt-results").querySelector("table");
    assert.match(table.caption.textContent, /Red columns.*token ID/);
    assert.deepEqual(
      [...table.querySelectorAll('[scope="rowgroup"]')].map(
        (cell) => cell.textContent,
      ),
      [
        "ONNX · onnx-light",
        "Hugging Face · Transformers",
        "ONNX / Transformers logit comparison",
      ],
    );
    const columnCount = Math.max(tokens.length, referenceTokens.length);
    for (const row of table.querySelectorAll("tr:not(.prompt-token-section)")) {
      assert.equal(row.cells.length, columnCount + 1);
      assert.equal(
        row.cells[0].classList.contains("prompt-token-mismatch"),
        false,
      );
      for (let index = 0; index < columnCount; index++) {
        const cell = row.cells[index + 1];
        const mismatch = tokens[index]?.id !== referenceTokens[index]?.id;
        assert.equal(
          cell.classList.contains("prompt-token-mismatch"),
          mismatch,
        );
        assert.equal(cell.hasAttribute("title"), mismatch);
      }
    }
    const missingGroup =
      table.tBodies[referenceTokens.length < tokens.length ? 1 : 0];
    assert.equal(missingGroup.rows[1].cells[columnCount].textContent, "—");
  }
});

test("rounds Prompt scores and validation metrics to three decimals without changing IDs, counts or raw values", (t) => {
  const { byId, api, messages } = panel(t);
  byId("prompt-form").querySelector('button[type="submit"]').click();
  const requestId = messages.at(-1).requestId;
  const tokens = [
    { text: "1.234567", id: 12345, score: 1.234567, timeMs: 12.345678 },
    { text: " second", id: 7, score: -2.345678, timeMs: 0.00049 },
    { text: " missing", id: 8, score: null, timeMs: null },
  ];
  const values = (label) =>
    [...byId("prompt-results").querySelectorAll("tr")].find(
      (row) => row.cells[0].textContent === label,
    ).cells;
  for (const [index, token] of tokens.entries())
    api.handleMessage({ type: "prompt-token", requestId, index, token });
  const expectedScores = ["1.235", "-2.346", "Unavailable (SDK)"];
  assert.deepEqual(
    [...values("Score (logit)")].slice(1).map((cell) => cell.textContent),
    expectedScores,
  );
  const result = {
    prompt: "1.234567",
    engine: "onnx-light",
    tokens,
    reference: {
      tokens: tokens.map((token, index) => ({
        ...token,
        score: [0.123456, -12.00049, 2][index],
      })),
    },
    discrepancies: tokens.map(() => ({
      maxAbsolute: 0.1234567,
      relative: 0.012345,
      l2: 1.234567,
      below001: 2,
      below01: 3,
      below1: 4,
      total: 12345,
    })),
  };
  const original = structuredClone(result);
  api.handleMessage({ type: "prompt-result", requestId, result });
  assert.deepEqual(
    [...values("Score (logit)")].slice(1).map((cell) => cell.textContent),
    expectedScores,
  );
  assert.deepEqual(
    [...values("Hugging Face score (logit)")]
      .slice(1)
      .map((cell) => cell.textContent),
    ["0.123", "-12.000", "2.000"],
  );
  assert.deepEqual(
    [...values("Time (ms)")].slice(1).map((cell) => cell.textContent),
    ["12.346", "0.000", "Unavailable (SDK)"],
  );
  for (const [label, expected] of [
    ["Maximum absolute distance", "0.123"],
    ["Relative error", "0.012"],
    ["L2 distance", "1.235"],
    ["Coefficients Δ < 0.01", "2"],
    ["Coefficients Δ < 0.1", "3"],
    ["Coefficients Δ < 1", "4"],
    ["Coefficients compared", "12345"],
  ]) {
    assert.equal(values(label)[1].textContent, expected);
  }
  assert.equal(values("Token ID")[1].textContent, "12345");
  assert.equal(values("Output")[1].textContent, "1.234567");
  assert.deepEqual(result, original);
});

test("validation accepts an explicit HF source after native generation without changing the generated prompt", (t) => {
  const { dom, byId, api, messages } = panel(t);
  api.handleMessage({
    type: "prompt-info",
    requestId: messages.at(-1).requestId,
    info: {
      available: true,
      engines: ["onnx-light"],
      validationRequiresReference: true,
    },
  });
  assert.match(
    byId("prompt-validation-status").textContent,
    /validation is unavailable/,
  );
  byId("prompt-form").querySelector('button[type="submit"]').click();
  const run = messages.at(-1);
  assert.equal(run.options.validate, false);
  api.handleMessage({
    type: "prompt-result",
    requestId: run.requestId,
    result: { prompt: run.options.prompt, tokens: [] },
  });
  assert.equal(byId("prompt-validate").disabled, true);
  byId("prompt-reference-source").value = " Qwen/Qwen3-0.6B ";
  byId("prompt-reference-source").dispatchEvent(new dom.window.Event("input"));
  assert.equal(byId("prompt-validate").disabled, false);
  byId("prompt-text").value = "Edited after generation";
  byId("prompt-validate").click();
  assert.equal(messages.at(-1).options.referenceSource, "Qwen/Qwen3-0.6B");
  assert.equal(messages.at(-1).options.prompt, run.options.prompt);
  assert.equal(byId("prompt-reference-source").disabled, true);
  api.reset();
  assert.equal(byId("prompt-reference-source").value, "");
});

test("warns before generation and requires a reference for validation on the first run", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("prompt-view").hidden = false;
  api.handleMessage({
    type: "prompt-info",
    requestId: messages.at(-1).requestId,
    info: {
      available: true,
      engines: ["onnxruntime-genai"],
      validationRequiresReference: true,
    },
  });
  assert.match(
    byId("prompt-validation-status").textContent,
    /Enter a model ID/,
  );
  byId("prompt-validate-run").checked = true;
  byId("prompt-validate-run").dispatchEvent(new dom.window.Event("change"));
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  assert.equal(
    messages.some(({ type }) => type === "run-prompt"),
    false,
  );
  assert.equal(byId("prompt-results").childElementCount, 0);
  assert.equal(byId("prompt-reference-source").required, true);
  assert.equal(dom.window.document.activeElement.id, "prompt-reference-source");
  byId("prompt-reference-source").value = "Qwen/Qwen3-0.6B";
  byId("prompt-reference-source").dispatchEvent(new dom.window.Event("input"));
  byId("prompt-form").querySelector('button[type="submit"]').click();
  const run = messages.at(-1);
  assert.equal(run.type, "run-prompt");
  assert.equal(run.options.engine, "onnxruntime-genai");
  assert.equal(run.options.validate, true);
  assert.equal(run.reuseResult, false);
  assert.equal(run.options.referenceSource, "Qwen/Qwen3-0.6B");
  api.handleMessage({
    type: "prompt-token",
    requestId: run.requestId,
    index: 0,
    token: { text: " Paris", id: 1, score: 4, timeMs: 1 },
  });
  assert.match(byId("prompt-results").textContent, /Paris/);
});

test("keeps stdout and validation tracebacks after the ONNX results with a concise status", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("prompt-form").querySelector('button[type="submit"]').click();
  api.handleMessage({
    type: "prompt-result",
    requestId: messages.at(-1).requestId,
    result: {
      prompt: "Hello",
      tokens: [{ text: " Paris", id: 1, score: 4, timeMs: 1 }],
    },
  });
  const table = byId("prompt-results").querySelector("table");
  byId("prompt-validate").click();
  const requestId = messages.at(-1).requestId;
  api.handleMessage({
    type: "prompt-progress",
    requestId,
    output: "Validating generated prefixes with the local HF reference",
  });
  assert.equal(byId("prompt-output").hidden, false);
  const error =
    "Hugging Face validation exited with code 1.\nTraceback:\nNo local Hugging Face source is available.";
  api.handleMessage({ type: "prompt-error", requestId, error });
  assert.equal(byId("prompt-status").textContent, error.split("\n")[0]);
  assert.equal(byId("prompt-progress").textContent, error);
  assert.equal(byId("prompt-results").querySelector("table"), table);
  assert.ok(
    byId("prompt-results").compareDocumentPosition(byId("prompt-output")) &
      dom.window.Node.DOCUMENT_POSITION_FOLLOWING,
  );
  api.reset();
  assert.equal(byId("prompt-output").hidden, true);
  assert.equal(byId("prompt-progress").textContent, "");
});

test("unavailable prompt explains the missing package instead of enabling unsupported execution", (t) => {
  const { byId, api, messages } = panel(t);
  const reason =
    "Prompt is disabled because generation metadata is missing. For Mobius, export with --runtime ort-genai.";
  api.handleMessage({
    type: "prompt-info",
    requestId: messages[0].requestId,
    info: { available: false, engines: [], reason },
  });
  assert.equal(byId("prompt-fields").disabled, true);
  assert.equal(byId("prompt-status").textContent, reason);
  assert.ok(
    [...byId("prompt-engine").options].every((option) => option.disabled),
  );
  const count = messages.length;
  byId("prompt-form").querySelector('button[type="submit"]').click();
  assert.equal(messages.length, count);
});

test("defaults to ONNX Runtime GenAI when available, regardless of engine list order", (t) => {
  const { dom, byId, api, messages } = panel(t);
  for (const engines of [
    ["onnx-light", "onnxruntime-genai"],
    ["onnxruntime-genai", "onnx-light"],
    ["onnxruntime-genai"],
  ]) {
    api.handleMessage({
      type: "prompt-info",
      requestId: messages[0].requestId,
      info: { available: true, engines },
    });
    assert.equal(byId("prompt-engine").value, "onnxruntime-genai");
    byId("prompt-settings-open").click();
    assert.equal(byId("prompt-settings-genai").hidden, false);
    assert.equal(byId("prompt-settings-light").hidden, true);
    byId("prompt-settings-back").click();
  }
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  assert.equal(messages.at(-1).options.engine, "onnxruntime-genai");
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
    assert.equal(table.rows[0].cells[0].colSpan, index + 2);
    for (const row of [...table.rows].slice(1))
      assert.equal(row.cells.length, index + 2);
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
  assert.equal(byId("prompt-settings-genai").hidden, true);
  assert.equal(byId("prompt-settings-engine"), null);
  assert.match(byId("prompt-settings-title").textContent, /onnx-light/);
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
  byId("prompt-engine").dispatchEvent(new dom.window.Event("change"));
  assert.equal(messages.at(-1).type, "prompt-release");
  byId("prompt-settings-open").click();
  assert.equal(byId("prompt-settings-genai").hidden, false);
  assert.equal(byId("prompt-settings-light").hidden, true);
  assert.match(byId("prompt-settings-title").textContent, /GenAI/);
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

test("model replacement updates an open settings page to the available engine", (t) => {
  const { dom, byId, api, messages } = panel(t);
  api.handleMessage({
    type: "prompt-info",
    requestId: messages[0].requestId,
    info: { available: true, engines: ["onnx-light", "onnxruntime-genai"] },
  });
  byId("prompt-engine").value = "onnxruntime-genai";
  byId("prompt-engine").dispatchEvent(new dom.window.Event("change"));
  byId("prompt-settings-open").click();
  assert.equal(byId("prompt-settings-genai").hidden, false);
  api.load();
  api.handleMessage({
    type: "prompt-info",
    requestId: messages.at(-1).requestId,
    info: { available: true, engines: ["onnx-light"], pipeline: true },
  });
  assert.equal(byId("prompt-session-page").hidden, false);
  assert.equal(byId("prompt-settings-genai").hidden, true);
  assert.equal(byId("prompt-settings-light").hidden, false);
  assert.match(byId("prompt-settings-title").textContent, /onnx-light/);
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

test("invalid settings stop generation and pipelines expose SDK owning-loader controls", (t) => {
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
  const light = byId("prompt-settings-light");
  const overlap = light.querySelector('[data-setting="loadOverlap"]');
  const loader = light.querySelector('[data-setting="sdkLoader"]');
  assert.equal(overlap.disabled, true);
  assert.equal(loader.disabled, false);
  assert.equal(byId("prompt-settings-light").hidden, false);
  assert.equal(byId("prompt-settings-genai").hidden, true);
  loader.value = "true";
  loader.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  for (const field of light.querySelectorAll("[data-setting]"))
    assert.equal(field.matches(":disabled"), false, field.dataset.setting);
  overlap.value = "false";
  light.querySelector('[data-setting="loadMemoryMiB"]').value = "512";
  byId("prompt-settings-back").click();
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  assert.deepEqual(
    JSON.parse(JSON.stringify(messages.at(-1).options.sessionSettings)),
    {
      threads: 1,
      sdkLoader: true,
      loadOverlap: false,
      loadMemoryMiB: 512,
      profile: false,
    },
  );
  loader.value = "false";
  loader.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  assert.equal(overlap.disabled, true);
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

test("SDK capture limits preserve timings without fabricating raw scores", (t) => {
  const { dom, byId, api, messages } = panel(t);
  byId("prompt-form").dispatchEvent(new dom.window.Event("submit"));
  api.handleMessage({
    type: "prompt-result",
    requestId: messages.at(-1).requestId,
    result: {
      prompt: "Hello",
      tokens: [{ text: " world", id: 1, score: null, timeMs: 0.1 }],
      limitations: ["The SDK logit snapshot budget was exhausted."],
    },
  });
  assert.match(byId("prompt-results").textContent, /Unavailable \(SDK\)/);
  assert.match(byId("prompt-results").textContent, /snapshot budget/);
});
