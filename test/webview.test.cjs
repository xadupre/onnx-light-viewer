const assert = require("node:assert/strict");
const { test } = require("node:test");
const { huggingFaceTasks } = require("../dist/huggingFaceTasks");
const { webviewHtml } = require("../dist/webview");

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
