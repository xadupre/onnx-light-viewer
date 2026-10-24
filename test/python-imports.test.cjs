const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

for (const file of [
  "examples/generate_tiny_llm.py",
  "examples/generate_quantized.py",
  "examples/generate_pipeline.py",
  "test/models/generate_models.py",
]) {
  test(`${file} imports the ONNX helper module as oh`, () => {
    const source = readFileSync(path.resolve(__dirname, "..", file), "utf8");
    assert.match(source, /^import onnx_light\.onnx\.helper as oh$/m);
    assert.doesNotMatch(source, /^from onnx_light\.onnx import .*\bhelper\b/m);
    assert.doesNotMatch(source, /\bhelper\./);
    assert.match(source, /\boh\.make_model\(/);
  });
}
