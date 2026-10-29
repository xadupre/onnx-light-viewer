const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const { JSDOM } = require("jsdom");
const { huggingFaceTasks } = require("../dist/huggingFaceTasks");
const { webviewHtml } = require("../dist/webview");

for (const initialTab of ["model", "logs"]) {
  test(`groups viewer features by theme when opening ${initialTab}`, (t) => {
    const dom = new JSDOM(
      webviewHtml(
        (name) => name,
        "test-source",
        "test-nonce",
        "test",
        initialTab,
        require("../package.json").version,
      ),
    );
    t.after(() => dom.window.close());
    const { document } = dom.window;
    assert.equal(
      document.getElementById("app-version").textContent,
      `ONNX Light Viewer ${require("../package.json").version}`,
    );
    if (initialTab === "model")
      assert.equal(
        document.getElementById("app-version").closest("[hidden]"),
        null,
      );
    const groups = (selector) =>
      [...document.querySelectorAll(selector)].map((group) => [
        group.querySelector(".feature-group-title").textContent,
        [...group.querySelectorAll("button")].map((button) => button.id),
      ]);
    assert.deepEqual(groups(".viewer-tabs > .feature-group"), [
      [
        "Navigation",
        ["model-tab", "statistics-tab", "memory-tab", "tensors-tab"],
      ],
      [
        "Application",
        [
          "logs-tab",
          "status-tab",
          "controls-tab",
          "repositories-tab",
          "settings-tab",
        ],
      ],
    ]);
    assert.deepEqual(groups("#sidebar-controls > .feature-group"), [
      [
        "Navigation",
        [
          "open-model",
          "save-model",
          "show-find",
          "show-operator",
          "show-metadata",
        ],
      ],
      ["Convert", ["export-model"]],
      ["Optimization", ["optimize-model", "fuse-pattern", "undo-fusion"]],
    ]);
    for (const group of document.querySelectorAll(
      "#sidebar-controls > .feature-group",
    )) {
      assert.equal(group.getAttribute("role"), "group");
      assert.equal(
        document.getElementById(group.getAttribute("aria-labelledby")),
        group.querySelector(".feature-group-title"),
      );
    }
    const tabs = [...document.querySelectorAll('[role="tab"]')];
    assert.deepEqual(
      tabs
        .filter((tab) => tab.getAttribute("aria-selected") === "true")
        .map((tab) => tab.id),
      [`${initialTab}-tab`],
    );
    for (const tab of tabs) {
      assert.equal(
        tab.closest('[role="tablist"]').getAttribute("aria-label"),
        "Viewer",
      );
      const panel = document.getElementById(tab.getAttribute("aria-controls"));
      assert.equal(panel.hidden, tab.id !== `${initialTab}-tab`);
    }
  });
}

test("optimization panel exposes backend buttons without native introductory text", (t) => {
  const dom = new JSDOM(
    webviewHtml((name) => name, "test-source", "test-nonce", "test"),
  );
  t.after(() => dom.window.close());
  const panel = dom.window.document.getElementById("optimization-panel");
  assert.equal(panel.querySelector(":scope > p:not([hidden])"), null);
  assert.equal(
    panel
      .querySelector("#optimization-title")
      .nextElementSibling.getAttribute("aria-label"),
    "Optimizer backend",
  );
  assert.equal(
    panel.querySelector("#optimizer-onnx-light").getAttribute("aria-pressed"),
    "true",
  );
  assert.equal(
    panel.querySelector("#optimizer-onnxscript").getAttribute("aria-pressed"),
    "false",
  );
  assert.equal(
    panel.querySelector("#onnxscript-optimization-note").hidden,
    true,
  );
  assert.ok(panel.querySelector("#optimization-patterns"));
  assert.equal(
    panel.querySelector("#run-optimization").textContent,
    "Run global optimization",
  );
});

test("tensor statistics shell exposes a labelled dialog and loads its controller before the viewer", (t) => {
  const dom = new JSDOM(
    webviewHtml((name) => name, "test-source", "test-nonce", "test"),
  );
  t.after(() => dom.window.close());
  const { document } = dom.window;
  const dialog = document.getElementById("tensor-statistics-dialog");
  assert.equal(dialog.tagName, "DIALOG");
  assert.equal(
    document.getElementById(dialog.getAttribute("aria-labelledby")).textContent,
    "Tensor statistics",
  );
  const tab = document.getElementById("tensors-tab");
  assert.ok(tab.firstElementChild.matches('svg[aria-hidden="true"]'));
  assert.equal(tab.textContent, "Tensors");
  assert.equal(document.getElementById("show-tensor-statistics"), null);
  assert.equal(document.getElementById("all-tensor-statistics"), null);
  assert.equal(
    document.querySelector('label[for="tensor-statistics-name"]').textContent,
    "Tensor name",
  );
  assert.equal(
    document.getElementById("tensor-statistics-status").getAttribute("role"),
    "status",
  );
  const scripts = [...document.querySelectorAll("script[src]")].map((script) =>
    script.getAttribute("src"),
  );
  assert.ok(
    scripts.indexOf("tensor-statistics-panel.js") <
      scripts.indexOf("viewer.js"),
  );
});

test("action buttons use rounded neutral surfaces and distinct typography", (t) => {
  const dom = new JSDOM(
    webviewHtml((name) => name, "test-source", "test-nonce", "test"),
  );
  t.after(() => dom.window.close());
  const { document } = dom.window;
  const stylesheet = document.createElement("style");
  stylesheet.textContent = readFileSync(
    require.resolve("../media/viewer.css"),
    "utf8",
  );
  document.head.append(stylesheet);
  const rules = [...stylesheet.sheet.cssRules];
  const rule = (selector) =>
    rules.find((entry) => entry.selectorText === selector).style;
  for (const id of [
    "rewriting-next",
    "rewriting-all",
    "run-optimization",
    "preview-fusion",
    "apply-fusion",
    "open-submit",
    "save-submit",
  ]) {
    const button = document.getElementById(id);
    const style = dom.window.getComputedStyle(button);
    assert.equal(style.borderRadius, "10px", id);
    assert.equal(style.fontFamily, "system-ui, sans-serif", id);
    assert.ok(Number(style.fontWeight) >= 600, id);
  }
  assert.equal(rule("button").border, "1px solid var(--muted)");
  assert.equal(rule("button").background, "var(--panel)");
  assert.equal(rule("button").color, "var(--text)");
  assert.equal(rule(".primary").background, "");
  assert.equal(rule(".primary").color, "");
  assert.equal(rule("button:hover:not(:disabled)").background, "var(--border)");
  assert.equal(
    rule("button:focus-visible,\n[tabindex]:focus-visible").outline,
    "2px solid var(--mint)",
  );
  assert.equal(rule("button:disabled").cursor, "default");
  assert.equal(rule("button:disabled").opacity, "0.5");
  assert.equal(
    dom.window.getComputedStyle(document.getElementById("model-tab"))
      .backgroundColor,
    "rgba(0, 0, 0, 0)",
  );
});

test("keeps Hugging Face task choices and ordering unchanged in the shell", () => {
  const expected = [
    ["", "All tasks"],
    ["feature-extraction", "Feature extraction / embeddings"],
    ["sentence-similarity", "Sentence similarity"],
    ["text-ranking", "Text ranking"],
    ["text-generation", "Text generation"],
    ["text-classification", "Text classification"],
    ["token-classification", "Token classification"],
    ["question-answering", "Question answering"],
    ["summarization", "Summarization"],
    ["translation", "Translation"],
    ["image-classification", "Image classification"],
    ["object-detection", "Object detection"],
    ["image-segmentation", "Image segmentation"],
    ["automatic-speech-recognition", "Speech recognition"],
    ["text-to-speech", "Text to speech"],
  ];
  assert.deepEqual(
    huggingFaceTasks.map(({ value, label }) => [value, label]),
    expected,
  );
  const html = webviewHtml((name) => name, "test-source", "test-nonce", "test");
  const select = html.match(/<select id="hf-task"[^>]*>(.*?)<\/select>/)[1];
  assert.deepEqual(
    [...select.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)].map(
      ([, value, label]) => [value, label],
    ),
    expected,
  );
});

test("loads the static shell with only dependency-free task constants", () => {
  const shell = require.cache[require.resolve("../dist/webview")];
  const tasks = require.cache[require.resolve("../dist/huggingFaceTasks")];
  assert.deepEqual(shell.children, [tasks]);
  assert.deepEqual(tasks.children, []);
  assert.equal(
    require.cache[require.resolve("../dist/huggingface")],
    undefined,
  );
});
