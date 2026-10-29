const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const {
  mkdtemp,
  mkdir,
  readFile,
  writeFile,
  rm,
  readdir,
  cp,
  open,
} = require("node:fs/promises");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const {
  promptModel,
  nativeEnvironment,
} = require("./helpers/prompt-model.cjs");
const {
  promptPipelineInfo,
  loadPromptPipeline,
} = require("../dist/promptPipeline");
const { loadModelFile, preparePromptPipeline } = require("../dist/graph");
const { runPrompt } = require("../dist/prompt");
const { provenancePath } = require("../dist/export");

async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), "viewer-mobius-pipeline-"));
  t.after(() => rm(root, { force: true, recursive: true }));
  const model = await promptModel(root, true);
  const provenance = {
    version: 1,
    model: "owner/model",
    source: root,
    exporter: "mobius",
  };
  await writeFile(provenancePath(model), JSON.stringify(provenance));
  for (const [name, value] of Object.entries(nativeEnvironment())) {
    const previous = process.env[name];
    process.env[name] = value;
    t.after(() => {
      if (previous === undefined) delete process.env[name];
      else process.env[name] = previous;
    });
  }
  return { root, model, provenance };
}

async function assertRelocatedPipeline(root, model) {
  assert.equal((await promptPipelineInfo(model)).available, true);
  const pipeline = await loadPromptPipeline(model);
  assert.equal(path.dirname(pipeline), root);
  assert.equal(
    JSON.parse(await readFile(provenancePath(pipeline), "utf8"))
      .packageDirectory,
    root,
  );
  const result = await runPrompt(
    pipeline,
    { engine: "onnx-light", prompt: "HH", maxTokens: 3 },
    "/no/python",
    new AbortController().signal,
    () => {},
  );
  assert.deepEqual(
    result.tokens.map((token) => token.id),
    [1, 1, 1],
  );
}

test("relocated flat Mobius packages ignore the old root whether it exists or not", async (t) => {
  const { root, model, provenance } = await fixture(t);
  await writeFile(
    provenancePath(model),
    JSON.stringify({ ...provenance, packageDirectory: root }),
  );
  const relocated = `${root}-relocated`;
  t.after(() => rm(relocated, { force: true, recursive: true }));
  await cp(root, relocated, { recursive: true });
  const selected = path.join(relocated, path.basename(model));
  await assertRelocatedPipeline(relocated, selected);
  await rm(root, { recursive: true });
  await assertRelocatedPipeline(relocated, selected);
});

test("native pipeline output uses a reserved path rather than a Node file descriptor", async (t) => {
  const { root, model } = await fixture(t);
  const output = path.join(root, "pipeline-\u00e9.pipe-onnx");
  await assert.rejects(
    preparePromptPipeline(model, "", root, output),
    /Could not write the generation pipeline/,
  );
  const handle = await open(output, "wx", 0o600);
  try {
    assert.equal(
      (await preparePromptPipeline(model, "", root, output)).available,
      true,
    );
    assert.deepEqual(
      (await loadModelFile(output, output))
        .graph()
        .nodes.map((node) => node.name),
      ["prefill"],
    );
  } finally {
    await handle.close();
  }
});

test("legacy Mobius history artifacts assemble a reusable native pipeline without modifying sources", async (t) => {
  const { root, model } = await fixture(t);
  const original = await readFile(model);
  const weights = await readFile(path.join(root, "weights", "decoder.bin"));
  const before = await readdir(root);
  const info = await promptPipelineInfo(model);
  assert.equal(info.available, true);
  assert.match(info.label, /Mobius/);
  assert.deepEqual(await readdir(root), before);
  const pipeline = await loadPromptPipeline(model);
  assert.equal(await loadPromptPipeline(model), pipeline);
  assert.deepEqual(
    (await loadModelFile(pipeline, pipeline))
      .graph()
      .nodes.map((node) => node.name),
    ["prefill"],
  );
  assert.deepEqual(await readFile(model), original);
  assert.deepEqual(
    await readFile(path.join(root, "weights", "decoder.bin")),
    weights,
  );
  const result = await runPrompt(
    pipeline,
    { engine: "onnx-light", prompt: "HH", maxTokens: 3 },
    "/no/python",
    new AbortController().signal,
    () => {},
  );
  assert.deepEqual(
    result.tokens.map((token) => token.id),
    [1, 1, 1],
  );
  const legacy = path.join(root, "model");
  await mkdir(legacy);
  const legacyModel = await promptModel(legacy);
  await writeFile(
    provenancePath(legacyModel),
    JSON.stringify({
      version: 1,
      exporter: "mobius",
      model: "owner/model",
      source: legacy,
    }),
  );
  assert.equal(path.dirname(await loadPromptPipeline(legacyModel)), legacy);
});

test("Mobius split embedding/decoder exports wire declared caches through native SDK feedback", async (t) => {
  const { root, model, provenance } = await fixture(t);
  await mkdir(path.join(root, "embedding"));
  await mkdir(path.join(root, "decoder"));
  execFileSync(
    process.platform === "win32" ? "python" : "python3",
    [
      "-c",
      `
import pathlib, struct, sys
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto as T
root = pathlib.Path(sys.argv[1])
port = oh.make_tensor_value_info
data = struct.pack("<16f", *([0, 4, 0, 0] * 4))
(root / "embedding" / "weights.bin").write_bytes(data)
weights = T(name="embeddings", data_type=T.FLOAT, dims=[4, 4])
weights.data_location = T.EXTERNAL
location = weights.external_data.add()
location.key, location.value = "location", "weights.bin"
embedding = oh.make_model(oh.make_graph(
    [oh.make_node("Gather", ["embeddings", "input_ids"], ["inputs_embeds"])],
    "embedding", [port("input_ids", T.INT64, [1, "sequence"])],
    [port("inputs_embeds", T.FLOAT, [1, "sequence", 4])], [weights]),
    opset_imports=[oh.make_opsetid("", 18)], ir_version=9)
onnx.save(embedding, root / "embedding" / "model.onnx")
decoder = oh.make_model(oh.make_graph([
    oh.make_node("Identity", ["inputs_embeds"], ["logits"]),
    oh.make_node("Unsqueeze", ["inputs_embeds", "axes"], ["new_key"]),
    oh.make_node("Concat", ["past_key_values.0.key", "new_key"], ["present.0.key"], axis=2)],
    "decoder", [port("inputs_embeds", T.FLOAT, [1, "sequence", 4]),
                port("past_key_values.0.key", T.FLOAT, [1, 1, "past_sequence", 4])],
    [port("logits", T.FLOAT, [1, "sequence", 4]),
     port("present.0.key", T.FLOAT, [1, 1, "total_sequence", 4])],
    [oh.make_tensor("axes", T.INT64, [1], [1])]),
    opset_imports=[oh.make_opsetid("", 18)], ir_version=9)
onnx.save(decoder, root / "decoder" / "model.onnx")
`,
      root,
    ],
    { stdio: "pipe" },
  );
  const decoder = path.join(root, "decoder", "model.onnx");
  await writeFile(
    provenancePath(decoder),
    JSON.stringify({
      ...provenance,
      source: path.join(root, "missing-checkpoint"),
      packageDirectory: root,
    }),
  );
  const configPath = path.join(root, "genai_config.json");
  const config = JSON.parse(await readFile(configPath, "utf8"));
  config.model.decoder.filename = "decoder/model.onnx";
  await writeFile(configPath, JSON.stringify(config));
  const relocated = `${root}-relocated`;
  t.after(() => rm(relocated, { force: true, recursive: true }));
  await cp(root, relocated, { recursive: true });
  const relocatedDecoder = path.join(relocated, "decoder", "model.onnx");
  await assertRelocatedPipeline(relocated, relocatedDecoder);
  const checks = new Map();
  assert.equal((await promptPipelineInfo(decoder, checks)).available, true);
  assert.equal((await promptPipelineInfo(model, checks)).available, true);
  assert.equal(checks.size, 1);
  const pipeline = await loadPromptPipeline(decoder);
  const native = await loadModelFile(pipeline, pipeline);
  native.verify();
  assert.equal(native.graph().nodes.length, 2);
  const result = await runPrompt(
    pipeline,
    { engine: "onnx-light", prompt: "HH", maxTokens: 3 },
    "/no/python",
    new AbortController().signal,
    () => {},
  );
  assert.deepEqual(
    result.tokens.map((token) => token.id),
    [1, 1, 1],
  );
  execFileSync(
    process.platform === "win32" ? "python" : "python3",
    [
      "-c",
      `
import sys
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto as T
model = onnx.load(sys.argv[1], load_external_data=False)
model.graph.input.append(oh.make_tensor_value_info("image_features", T.FLOAT, [1, 4]))
onnx.save(model, sys.argv[1])
`,
      decoder,
    ],
    { stdio: "pipe" },
  );
  assert.match(
    (await promptPipelineInfo(decoder)).label,
    /additional input: image_features/,
  );
  await rm(path.join(root, "embedding", "model.onnx"));
  const unavailable = await promptPipelineInfo(decoder);
  assert.equal(unavailable.available, false);
  assert.match(unavailable.label, /embedding.*missing/);
  await rm(root, { recursive: true });
  await assertRelocatedPipeline(relocated, relocatedDecoder);
});

test("history inspection distinguishes missing and unrelated artifacts and surfaces absent weights", async (t) => {
  const { root, model } = await fixture(t);
  await rm(provenancePath(model));
  assert.equal((await promptPipelineInfo(model)).available, false);
  await assert.rejects(loadPromptPipeline(model), /not a Mobius export/);
  assert.match(
    (await promptPipelineInfo(path.join(root, "absent.onnx"))).label,
    /no longer exists/,
  );
  await writeFile(
    provenancePath(model),
    JSON.stringify({
      version: 1,
      model: "owner/model",
      source: root,
      exporter: "mobius",
    }),
  );
  await rm(path.join(root, "weights", "decoder.bin"));
  await assert.rejects(promptPipelineInfo(model), /Missing external weights/);
});
