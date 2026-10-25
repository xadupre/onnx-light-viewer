const assert = require("node:assert/strict");
const { readFile, writeFile } = require("node:fs/promises");
const path = require("node:path");
const { loadModel } = require("../dist/graph");

async function main() {
  const original = loadModel(
    await readFile(path.join(__dirname, "../test/models/fusion.onnx")),
    "Saved rewriting example",
  );
  const first = await original.applyFusion("TransposeTranspose", [0]);
  const full = await first.applyFusion("TransposeTranspose", [1]);
  const bytes = await full.serialize(undefined, undefined, true);
  const saved = loadModel(bytes, "Saved rewriting example");
  saved.verify();
  const [scenario] = saved.rewritingScenarios();
  assert.equal(scenario.steps.length, 2);
  const replayed = await saved.applyRewritingScenario(scenario.name, true);
  replayed.verify();
  assert.deepEqual(replayed.graph(), full.graph());
  const destination = path.join(__dirname, "rewriting-scenarios.onnx");
  await writeFile(destination, bytes);
  console.log(destination);
}

main();
