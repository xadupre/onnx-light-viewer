const assert = require("node:assert/strict");
const { execFile } = require("node:child_process");
const { readdirSync, existsSync } = require("node:fs");
const {
  mkdtemp,
  readFile,
  writeFile,
  readdir,
  rm,
  mkdir,
  rename,
  symlink,
} = require("node:fs/promises");
const path = require("node:path");
const { tmpdir } = require("node:os");
const { promisify } = require("node:util");
const { test } = require("node:test");
const { promptInfo, runPrompt } = require("../dist/prompt");
const { PromptSession } = require("../dist/promptSession");
const { provenancePath } = require("../dist/export");
const {
  promptModel,
  nativeEnvironment,
} = require("./helpers/prompt-model.cjs");

const execute = promisify(execFile);
const python = process.platform === "win32" ? "python" : "python3";
const options = { engine: "onnxruntime-genai", prompt: "H", maxTokens: 3 };
const signal = () => new AbortController().signal;

async function fixture(
  t,
  environment = true,
  external = false,
  externalLength = false,
) {
  const root = await mkdtemp(path.join(tmpdir(), "onnx-viewer-prompt-test-"));
  t.after(() => rm(root, { force: true, recursive: true }));
  const model = await promptModel(root, external, externalLength);
  if (environment) {
    for (const [name, value] of Object.entries(nativeEnvironment())) {
      const previous = process.env[name];
      process.env[name] = value;
      t.after(() => {
        if (previous === undefined) delete process.env[name];
        else process.env[name] = previous;
      });
    }
  }
  return { root, model };
}

test("native onnx-light Generate owns autoregression, EOS and error propagation", async (t) => {
  const { root, model } = await fixture(t, true, true);
  const run = () =>
    runPrompt(
      model,
      { ...options, engine: "onnx-light" },
      "/no/python",
      signal(),
      () => {},
    );
  assert.deepEqual(
    (await run()).tokens.map((token) => token.id),
    [1, 1, 1],
  );
  const pipelinePath = path.join(root, "model.PIPE-ONNX");
  await rename(path.join(root, "model.pipe-onnx"), pipelinePath);
  const pipeline = await runPrompt(
    pipelinePath,
    { ...options, engine: "onnx-light" },
    "/no/python",
    signal(),
    () => {},
  );
  assert.deepEqual(
    pipeline.tokens.map((token) => token.id),
    [1, 1, 1],
  );
  await writeFile(
    path.join(root, "generation_config.json"),
    JSON.stringify({ eos_token_id: 1 }),
  );
  assert.deepEqual(
    (await run()).tokens.map((token) => token.id),
    [1],
  );
  await writeFile(
    path.join(root, "generation_config.json"),
    JSON.stringify({ eos_token_id: [2, 1] }),
  );
  for (const file of [model, pipelinePath])
    assert.deepEqual(
      (
        await runPrompt(
          file,
          { ...options, engine: "onnx-light" },
          "/no/python",
          signal(),
          () => {},
        )
      ).tokens.map((token) => token.id),
      [1],
    );
  await writeFile(
    path.join(root, "generation_config.json"),
    JSON.stringify({ eos_token_id: [] }),
  );
  assert.deepEqual(
    (await run()).tokens.map((token) => token.id),
    [1, 1, 1],
  );
  await writeFile(
    path.join(root, "generation_config.json"),
    JSON.stringify({ eos_token_id: [1, "2"] }),
  );
  await assert.rejects(run(), /eos_token_id must be/);
  await rm(path.join(root, "generation_config.json"));
  await writeFile(model, "invalid ONNX");
  await assert.rejects(run(), /decode|parse|wire|field/i);
});

test("prompt availability follows local causal metadata and supports native pipeline discovery", async (t) => {
  const { root, model } = await fixture(t, false);
  assert.deepEqual(await promptInfo(model), {
    engines: ["onnx-light", "onnxruntime-genai"],
    available: true,
    validationRequiresReference: true,
  });
  await writeFile(path.join(root, "model.safetensors"), "checkpoint fixture");
  assert.equal((await promptInfo(model)).validationRequiresReference, false);
  await rm(path.join(root, "model.safetensors"));
  const genaiConfig = await readFile(path.join(root, "genai_config.json"));
  await rm(path.join(root, "genai_config.json"));
  assert.equal((await promptInfo(model)).available, false);
  const pipeline = path.join(root, "model.pipe-onnx");
  await writeFile(pipeline, "");
  assert.equal((await promptInfo(pipeline)).available, false);
  await writeFile(path.join(root, "genai_config.json"), genaiConfig);
  assert.deepEqual((await promptInfo(pipeline)).engines, ["onnx-light"]);
  await rm(path.join(root, "genai_config.json"));
  await writeFile(
    path.join(root, "config.json"),
    JSON.stringify({ architectures: ["BertForSequenceClassification"] }),
  );
  assert.equal((await promptInfo(model)).available, false);
  assert.equal((await promptInfo("relative.onnx")).available, false);
});

test("source discovery requires the GenAI configuration in the tokenizer directory before native loading", async (t) => {
  const { root, model } = await fixture(t);
  const source = path.join(root, "checkpoint");
  await mkdir(source);
  for (const name of ["config.json", "tokenizer.json", "tokenizer_config.json"])
    await rename(path.join(root, name), path.join(source, name));
  await writeFile(
    provenancePath(model),
    JSON.stringify({
      version: 1,
      exporter: "mobius",
      model: "owner/model",
      source,
    }),
  );
  const phases = [];
  const run = () =>
    runPrompt(
      model,
      { ...options, engine: "onnx-light" },
      "/no/python",
      signal(),
      () => {},
      undefined,
      undefined,
      undefined,
      (phase) => phases.push(phase),
    );
  const info = await promptInfo(model);
  assert.equal(info.available, false);
  assert.deepEqual(info.engines, []);
  assert.match(info.reason, /config\.json, genai_config\.json and tokenizer/);
  await assert.rejects(run(), /genai_config\.json/);
  assert.deepEqual(phases, []);
  assert.equal(
    (await readdir(root)).some((name) => name.startsWith(".prompt-")),
    false,
  );
  await writeFile(
    path.join(source, "genai_config.json"),
    await readFile(path.join(root, "genai_config.json")),
  );
  assert.deepEqual((await promptInfo(model)).engines, ["onnx-light"]);
  assert.deepEqual(
    (await run()).tokens.map((token) => token.id),
    [1, 1, 1],
  );
});

test("source discovery skips incomplete or malformed checkpoints and prefers complete source tokenizer packages", async (t) => {
  const { root, model } = await fixture(t);
  const source = path.join(root, "checkpoint");
  await mkdir(source);
  for (const name of ["config.json", "tokenizer.json", "tokenizer_config.json"])
    await writeFile(
      path.join(source, name),
      await readFile(path.join(root, name)),
    );
  const file = path.join(source, "tokenizer.json");
  const tokenizer = JSON.parse(await readFile(file, "utf8"));
  tokenizer.model.vocab.a = tokenizer.model.vocab.o;
  delete tokenizer.model.vocab.o;
  await writeFile(file, JSON.stringify(tokenizer));
  await writeFile(
    provenancePath(model),
    JSON.stringify({
      version: 1,
      exporter: "mobius",
      model: "owner/model",
      source,
    }),
  );
  const run = () =>
    runPrompt(
      model,
      { ...options, engine: "onnx-light" },
      "/no/python",
      signal(),
      () => {},
    );
  assert.deepEqual((await promptInfo(model)).engines, [
    "onnx-light",
    "onnxruntime-genai",
  ]);
  assert.equal((await run()).tokens[0].text, "o");
  await writeFile(
    path.join(source, "genai_config.json"),
    await readFile(path.join(root, "genai_config.json")),
  );
  assert.equal((await run()).tokens[0].text, "a");
  for (const config of ['{"architectures":', "null", "[]"]) {
    await writeFile(path.join(source, "config.json"), config);
    assert.deepEqual((await promptInfo(model)).engines, [
      "onnx-light",
      "onnxruntime-genai",
    ]);
    assert.equal((await run()).tokens[0].text, "o");
  }
  await rm(path.join(root, "config.json"));
  assert.deepEqual((await promptInfo(model)).engines, ["onnxruntime-genai"]);
  const genai = await runPrompt(
    model,
    options,
    "/no/python",
    signal(),
    () => {},
  );
  assert.deepEqual(
    genai.tokens.map((token) => token.id),
    [1, 1, 1],
  );
});

test("GenAI configuration alone does not make a tokenizer package available", async (t) => {
  const { root, model } = await fixture(t, false);
  await rm(path.join(root, "tokenizer.json"));
  assert.equal((await promptInfo(model)).available, true);
  await rm(path.join(root, "tokenizer_config.json"));
  assert.deepEqual((await promptInfo(model)).engines, []);
  await assert.rejects(
    runPrompt(model, options, "/no/python", signal(), () => {}),
    /genai_config\.json and tokenizer assets/,
  );
});

test("graph-only Mobius exports explain how to supply missing generation assets", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "onnx-viewer-graph-only-"));
  t.after(() => rm(root, { force: true, recursive: true }));
  const model = path.join(root, "model.onnx");
  await writeFile(model, "");
  const info = await promptInfo(model);
  assert.equal(info.available, false);
  assert.deepEqual(info.engines, []);
  assert.match(info.reason, /genai_config\.json/);
  assert.match(info.reason, /tokenizer/);
  assert.match(info.reason, /--runtime ort-genai/);
  assert.match(info.reason, /Integrated embeddings are supported/);
});

test("snapshot-style metadata symlinks remain available", async (t) => {
  const { root, model } = await fixture(t, false);
  const blobs = path.join(root, "blobs");
  await mkdir(blobs);
  for (const name of [
    "config.json",
    "genai_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
  ]) {
    const file = path.join(root, name);
    const blob = path.join(blobs, name);
    await writeFile(blob, await readFile(file));
    await rm(file);
    try {
      await symlink(blob, file, "file");
    } catch (error) {
      if (process.platform === "win32" && error.code === "EPERM") {
        t.skip("File symlinks require Windows developer privileges.");
        return;
      }
      throw error;
    }
  }
  assert.deepEqual((await promptInfo(model)).engines, [
    "onnx-light",
    "onnxruntime-genai",
  ]);
});

for (const engine of ["onnx-light", "onnxruntime-genai"]) {
  test(`native ${engine} selects the first-token argmax despite model sampling defaults`, async (t) => {
    const { root, model } = await fixture(t);
    const configPath = path.join(root, "genai_config.json");
    const config = JSON.parse(await readFile(configPath, "utf8"));
    config.search = {
      ...config.search,
      do_sample: true,
      temperature: 100,
      top_k: 4,
      top_p: 1,
    };
    await writeFile(configPath, JSON.stringify(config));
    const result = await runPrompt(
      model,
      { ...options, engine, maxTokens: 1 },
      "/no/python",
      signal(),
      () => {},
    );
    assert.equal(result.tokens.length, 1);
    assert.equal(result.tokens[0].id, 1);
    assert.equal(result.tokens[0].score, 4);
  });

  test(`native ${engine} streams real tokens and measures prefill and each decode step without Python`, async (t) => {
    const { root, model } = await fixture(t);
    const streamed = [];
    const result = await runPrompt(
      model,
      { ...options, engine, prompt: "HH" },
      "/this/python/must/not/be/executed",
      signal(),
      () => {},
      undefined,
      (token, index) => streamed.push({ token, index }),
    );
    assert.deepEqual(
      result.tokens.map((token) => token.id),
      [1, 1, 1],
    );
    assert.deepEqual(
      result.tokens.map((token) => token.score),
      [4, 4, 4],
    );
    assert.ok(
      result.tokens.every(
        (token) => Number.isFinite(token.timeMs) && token.timeMs >= 0,
      ),
    );
    assert.deepEqual(
      streamed.map(({ token }) => token),
      result.tokens,
    );
    assert.deepEqual(
      streamed.map(({ index }) => index),
      [0, 1, 2],
    );
    assert.match(result.signature, /^[0-9a-f]{64}$/);
    assert.ok(result.lifecycle.loadMs >= 0);
    assert.ok(result.lifecycle.warmupMs >= 0);
    assert.equal(result.lifecycle.reused, false);
    assert.equal(result.lifecycle.retained, false);
    assert.equal(
      (await readdir(root)).some((name) => name.startsWith(".prompt-")),
      false,
    );
  });
}

for (const engine of ["onnx-light", "onnxruntime-genai"]) {
  test(`${engine} discovers installed native SDK libraries without library environment variables`, async (t) => {
    const { model } = await fixture(t);
    delete process.env.ONNX_LIGHT_VIEWER_GENAI_LIBRARY;
    delete process.env.ORT_LIB_PATH;
    const session = new PromptSession();
    const run = (executable) =>
      runPrompt(
        model,
        { ...options, engine, keepLoaded: true },
        executable,
        signal(),
        () => {},
        undefined,
        undefined,
        session,
      );
    try {
      const first = await run(python);
      assert.deepEqual(
        first.tokens.map((token) => token.id),
        [1, 1, 1],
      );
      assert.equal(first.lifecycle.reused, false);
      const second = await run("/no/python-needed-for-retained-native-session");
      assert.deepEqual(
        second.tokens.map((token) => token.id),
        [1, 1, 1],
      );
      assert.equal(second.lifecycle.reused, true);
    } finally {
      await session.close();
    }
  });

  test(`${engine} discovery reports missing SDK packages in the selected Python environment`, async (t) => {
    const { root, model } = await fixture(t);
    delete process.env.ONNX_LIGHT_VIEWER_GENAI_LIBRARY;
    delete process.env.ORT_LIB_PATH;
    const environment = path.join(root, "empty-python");
    await execute(python, ["-m", "venv", "--without-pip", environment]);
    const executable = path.join(
      environment,
      process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
    );
    await assert.rejects(
      runPrompt(model, { ...options, engine }, executable, signal(), () => {}),
      /Missing onnxruntime_genai in the Python environment selected in Settings/,
    );
    assert.equal(
      (await readdir(root)).some((name) => name.startsWith(".prompt-")),
      false,
    );
  });
}

test("native sessions optionally retain, reuse, reconfigure and unload their model", async (t) => {
  for (const engine of ["onnx-light", "onnxruntime-genai"]) {
    await t.test(engine, async (t) => {
      const { model } = await fixture(t);
      const session = new PromptSession();
      const phases = [];
      const run = (changes = {}) =>
        runPrompt(
          model,
          { ...options, engine, keepLoaded: true, ...changes },
          "/no/python",
          signal(),
          () => {},
          undefined,
          undefined,
          session,
          (phase) => phases.push(phase),
        );
      try {
        const first = await run();
        assert.equal(first.lifecycle.reused, false);
        assert.equal(first.lifecycle.retained, true);
        assert.equal(phases[0].warmupMs, undefined);
        assert.ok(phases[1].warmupMs >= 0);
        const second = await run({ prompt: "HH", maxTokens: 2 });
        assert.deepEqual(
          second.tokens.map((token) => token.id),
          [1, 1],
        );
        assert.deepEqual(second.lifecycle, {
          ...first.lifecycle,
          reused: true,
        });
        const settings =
          engine === "onnx-light"
            ? {
                threads: 1,
                sdkLoader: true,
                loadOverlap: false,
                loadMemoryMiB: 1,
                checkShapes: true,
              }
            : {
                intraOpThreads: 1,
                cpuArena: false,
                memoryPattern: false,
                graphOptimization: "ORT_DISABLE_ALL",
                prepacking: false,
                provider: "cpu",
              };
        assert.equal(
          (await run({ sessionSettings: settings })).lifecycle.reused,
          false,
        );
        const unloaded = await run({
          sessionSettings: settings,
          keepLoaded: false,
        });
        assert.equal(unloaded.lifecycle.reused, true);
        assert.equal(unloaded.lifecycle.retained, false);
        assert.equal(
          (await run({ sessionSettings: settings })).lifecycle.reused,
          false,
        );
        await session.close();
        assert.equal(
          (await run({ sessionSettings: settings })).lifecycle.reused,
          false,
        );
      } finally {
        await session.close();
      }
    });
  }
});

test("changing engines releases the previous model before loading another", async (t) => {
  const { model } = await fixture(t);
  const session = new PromptSession();
  try {
    for (const engine of [
      "onnxruntime-genai",
      "onnx-light",
      "onnxruntime-genai",
    ]) {
      const result = await runPrompt(
        model,
        { ...options, engine, keepLoaded: true },
        "/no/python",
        signal(),
        () => {},
        undefined,
        undefined,
        session,
      );
      assert.equal(result.lifecycle.reused, false);
    }
  } finally {
    await session.close();
  }
});

test("parallel loading/session settings use native SDK APIs and profiles exclude warmup", async (t) => {
  const { model } = await fixture(t, true, true, true);
  const result = await runPrompt(
    model,
    {
      ...options,
      engine: "onnx-light",
      sessionSettings: {
        threads: 2,
        sdkLoader: true,
        loadOverlap: true,
        loadMemoryMiB: 1,
        profile: true,
      },
    },
    "/no/python",
    signal(),
    () => {},
  );
  const files = await readdir(result.profileDirectory);
  const events = JSON.parse(
    await readFile(path.join(result.profileDirectory, files[0]), "utf8"),
  );
  assert.equal(events.length, options.maxTokens);
  assert.ok(
    events.every(
      (event) =>
        event.operator === "Gather" &&
        event.timeMs >= 0 &&
        event.effectiveThreads === 2,
    ),
  );
  const genai = await runPrompt(
    model,
    {
      ...options,
      sessionSettings: {
        profile: true,
        provider: "cpu",
        intraOpThreads: 1,
        interOpThreads: 2,
        parallelExecution: true,
      },
    },
    "/no/python",
    signal(),
    () => {},
  );
  const profiles = (await readdir(genai.profileDirectory)).filter((name) =>
    name.startsWith("genai-"),
  );
  assert.ok(profiles.length > 0);
  assert.ok(
    JSON.parse(
      await readFile(path.join(genai.profileDirectory, profiles[0]), "utf8"),
    ).some((event) => event.cat === "Node"),
  );
});

test("the released SDK owning loader supports external tensors without explicit lengths", async (t) => {
  const { root, model } = await fixture(t, true, true);
  for (const loadOverlap of [false, true]) {
    const run = () =>
      runPrompt(
        model,
        {
          ...options,
          engine: "onnx-light",
          sessionSettings: { sdkLoader: true, loadOverlap },
        },
        "/no/python",
        signal(),
        () => {},
      );
    for (const [eos, expected] of [
      [[], [1, 1, 1]],
      [[2, 1], [1]],
    ]) {
      await writeFile(
        path.join(root, "generation_config.json"),
        JSON.stringify({ eos_token_id: eos }),
      );
      assert.deepEqual(
        (await run()).tokens.map((token) => token.id),
        expected,
      );
    }
  }
});

test("SDK owning pipeline generation reuses embedding preparation and preserves interface names", async (t) => {
  const { root, model } = await fixture(t, true, true);
  const pipeline = path.join(root, "embedding.pipe-onnx");
  const overBudget = path.join(root, "over-budget.pipe-onnx");
  await execute(python, [
    "-c",
    `
import sys
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as oh
from onnx_light.onnx import PipelineProto, TensorProto as T
embedding = onnx.load(sys.argv[1], load_external_data=False)
embedding.graph.input[0].name = "tokens"
embedding.graph.node[0].input.clear()
embedding.graph.node[0].input.extend(["weights", "tokens"])
embedding.graph.node[0].output.clear()
embedding.graph.node[0].output.append("embeddings")
embedding.graph.output[0].name = "embeddings"
prefill = oh.make_model(oh.make_graph(
    [oh.make_node("Identity", ["embeddings"], ["prediction"])], "prefill",
    [oh.make_tensor_value_info("embeddings", T.FLOAT, ["batch", "sequence", 4])],
    [oh.make_tensor_value_info("prediction", T.FLOAT, ["batch", "sequence", 4])]),
    opset_imports=[oh.make_opsetid("", 18)])
pipeline = PipelineProto(embedding=embedding, prefill=prefill,
                         input_ids_name="tokens", logits_name="prediction",
                         max_new_tokens=1, temperature=9.0, top_k=1, top_p=0.9,
                         eos_token_ids=[2, 1])
with open(sys.argv[2], "wb") as stream: stream.write(pipeline.SerializeToString())
for name, stage in [("embedding", pipeline.embedding), ("prefill", pipeline.prefill)]:
    stage.graph.initializer.append(oh.make_tensor(name + "_padding", T.FLOAT,
                                                [150000], bytes(600000), raw=True))
with open(sys.argv[3], "wb") as stream: stream.write(pipeline.SerializeToString())
`,
    model,
    pipeline,
    overBudget,
  ]);
  for (const loadOverlap of [false, true]) {
    const result = await runPrompt(
      pipeline,
      {
        ...options,
        engine: "onnx-light",
        sessionSettings: {
          sdkLoader: true,
          loadOverlap,
          loadMemoryMiB: 1,
          threads: 2,
          profile: true,
        },
      },
      "/no/python",
      signal(),
      () => {},
    );
    assert.deepEqual(
      result.tokens.map((token) => token.id),
      [1, 1, 1],
    );
    const profiles = await readdir(result.profileDirectory);
    const events = JSON.parse(
      await readFile(path.join(result.profileDirectory, profiles[0]), "utf8"),
    );
    assert.equal(events.length, options.maxTokens * 2);
    assert.deepEqual(
      [...new Set(events.map((event) => event.operator))].sort(),
      ["Gather", "Identity"],
    );
    assert.ok(events.every((event) => event.effectiveThreads === 2));
    await assert.rejects(
      runPrompt(
        overBudget,
        {
          ...options,
          engine: "onnx-light",
          sessionSettings: {
            sdkLoader: true,
            loadOverlap,
            loadMemoryMiB: 1,
          },
        },
        "/no/python",
        signal(),
        () => {},
      ),
      /memory admission limit/,
    );
  }
});

test("SDK owning pipelines support sequential/overlapped preparation and retained generation", async (t) => {
  for (const externalLength of [false, true])
    await t.test(`external length ${externalLength}`, async (t) => {
      const { root } = await fixture(t, true, true, externalLength);
      const model = path.join(root, "model.pipe-onnx");
      const session = new PromptSession();
      const run = (sessionSettings) =>
        runPrompt(
          model,
          {
            ...options,
            engine: "onnx-light",
            keepLoaded: true,
            sessionSettings,
          },
          "/no/python",
          signal(),
          () => {},
          undefined,
          undefined,
          session,
        );
      try {
        for (const loadOverlap of [undefined, false, true]) {
          const settings = {
            sdkLoader: true,
            threads: 2,
            loadMemoryMiB: 1,
            ...(loadOverlap === undefined ? {} : { loadOverlap }),
          };
          for (const reused of [false, true]) {
            const result = await run(settings);
            assert.deepEqual(
              result.tokens.map((token) => token.id),
              [1, 1, 1],
            );
            assert.ok(
              result.tokens.every(
                (token) => token.score === 4 && token.timeMs >= 0,
              ),
            );
            assert.equal(result.lifecycle.reused, reused);
            assert.equal(result.lifecycle.retained, true);
          }
        }
        await writeFile(
          path.join(root, "generation_config.json"),
          JSON.stringify({ eos_token_id: [2, 1] }),
        );
        assert.deepEqual(
          (await run({ sdkLoader: true })).tokens.map((token) => token.id),
          [1],
        );
        await writeFile(
          path.join(root, "generation_config.json"),
          JSON.stringify({ eos_token_id: [] }),
        );
        assert.deepEqual(
          (await run({ sdkLoader: true })).tokens.map((token) => token.id),
          [1, 1, 1],
        );
      } finally {
        await session.close();
      }
    });
});

test("cancelling during initialization releases its model without closing the replacement", async (t) => {
  const { model } = await fixture(t);
  const session = new PromptSession();
  const controller = new AbortController();
  let replacement;
  try {
    await assert.rejects(
      runPrompt(
        model,
        { ...options, keepLoaded: false },
        "/no/python",
        controller.signal,
        () => {},
        undefined,
        undefined,
        session,
        () => {
          controller.abort();
          replacement = runPrompt(
            model,
            { ...options, keepLoaded: true },
            "/no/python",
            signal(),
            () => {},
            undefined,
            undefined,
            session,
          );
        },
      ),
      { name: "AbortError" },
    );
    assert.ok(replacement);
    const result = await replacement;
    assert.equal(result.lifecycle.reused, false);
  } finally {
    await session.close();
  }
});

test("an obsolete cancelled request cannot release a retained replacement worker", async (t) => {
  const { root, model } = await fixture(t);
  const session = new PromptSession();
  const run = () =>
    runPrompt(
      model,
      { ...options, keepLoaded: true },
      "/no/python",
      signal(),
      () => {},
      undefined,
      undefined,
      session,
    );
  try {
    await run();
    const obsolete = new AbortController();
    obsolete.abort();
    await assert.rejects(
      session.run(
        path.join(root, "unused-input.json"),
        path.join(root, "unused-output.json"),
        "obsolete-session-key",
        obsolete.signal,
        () => {},
        () => {},
      ),
      { name: "AbortError" },
    );
    assert.equal((await run()).lifecycle.reused, true);
  } finally {
    await session.close();
  }
});

test("a native worker killed during generation rejects its request and permits another run", async (t) => {
  const { model } = await fixture(t);
  const session = new PromptSession();
  let killed = false;
  try {
    await assert.rejects(
      runPrompt(
        model,
        { ...options, maxTokens: 4096, keepLoaded: true },
        "/no/python",
        signal(),
        () => {},
        undefined,
        () => {
          if (!killed) {
            killed = true;
            assert.ok(session.child.kill("SIGKILL"));
          }
        },
        session,
      ),
      /Native prompt exited/,
    );
    assert.equal(killed, true);
    const result = await runPrompt(
      model,
      { ...options, keepLoaded: true },
      "/no/python",
      signal(),
      () => {},
      undefined,
      undefined,
      session,
    );
    assert.deepEqual(
      result.tokens.map((token) => token.id),
      [1, 1, 1],
    );
    assert.equal(result.lifecycle.reused, false);
  } finally {
    await session.close();
  }
});

for (const engine of ["onnx-light", "onnxruntime-genai"]) {
  test(`the first ${engine} token arrives before completion and cancellation stops the child`, async (t) => {
    const { root, model } = await fixture(t);
    const controller = new AbortController();
    let received = 0;
    await assert.rejects(
      runPrompt(
        model,
        { ...options, engine, maxTokens: 4096 },
        "/no/python",
        controller.signal,
        () => {},
        undefined,
        () => {
          const request = readdirSync(root).find((name) =>
            name.startsWith(".prompt-"),
          );
          assert.ok(request);
          assert.equal(
            existsSync(path.join(root, request, "output.json")),
            false,
          );
          received++;
          controller.abort();
        },
      ),
      { name: "AbortError" },
    );
    assert.equal(received, 1);
    assert.equal(
      readdirSync(root).some((name) => name.startsWith(".prompt-")),
      false,
    );
  });
}

test("native incremental decoding preserves UTF-8 characters split across token IDs", async (t) => {
  const { root, model } = await fixture(t, true, true);
  const file = path.join(root, "tokenizer.json");
  const tokenizer = JSON.parse(await readFile(file, "utf8"));
  tokenizer.model.vocab = { "\u00c3": 0, "\u00a9": 1, X: 2, H: 3 };
  tokenizer.model.unk_token = "X";
  tokenizer.decoder = {
    type: "ByteLevel",
    add_prefix_space: false,
    trim_offsets: true,
    use_regex: true,
  };
  await writeFile(file, JSON.stringify(tokenizer));
  const weights = Buffer.alloc(4 * 4 * 4);
  weights.writeFloatLE(4, (3 * 4 + 0) * 4);
  weights.writeFloatLE(4, (0 * 4 + 1) * 4);
  await writeFile(path.join(root, "weights", "decoder.bin"), weights);
  const result = await runPrompt(
    model,
    { ...options, maxTokens: 2 },
    "/no/python",
    signal(),
    () => {},
  );
  assert.deepEqual(
    result.tokens.map((token) => token.id),
    [0, 1],
  );
  assert.deepEqual(
    result.tokens.map((token) => token.text),
    ["", "\u00e9"],
  );
});

test("native SDK snapshots preserve DOUBLE precision and decode FLOAT16/BFLOAT16 raw scores", async (t) => {
  for (const dtype of ["DOUBLE", "FLOAT16", "BFLOAT16"])
    await t.test(dtype, async (t) => {
      const { model } = await fixture(t);
      await execute(python, [
        "-c",
        `
import struct, sys
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto as T
model = onnx.load(sys.argv[1])
dtype = getattr(T, sys.argv[2])
if dtype == T.DOUBLE:
    model.graph.initializer[0].data_type = dtype
    model.graph.initializer[0].raw_data = struct.pack("<16d", *([0, 4.0000000001, 0, 0] * 4))
else:
    model.graph.node[0].output.clear()
    model.graph.node[0].output.append("float_logits")
    model.graph.node.append(oh.make_node("Cast", ["float_logits"], ["logits"], to=dtype))
model.graph.output[0].type.tensor_type.elem_type = dtype
onnx.save(model, sys.argv[1])
`,
        model,
        dtype,
      ]);
      const result = await runPrompt(
        model,
        { ...options, engine: "onnx-light" },
        "/no/python",
        signal(),
        () => {},
      );
      assert.deepEqual(
        result.tokens.map((token) => token.score),
        Array(3).fill(dtype === "DOUBLE" ? 4.0000000001 : 4),
      );
      assert.ok(result.tokens.every((token) => token.timeMs >= 0));
      assert.equal(result.limitations, undefined);
    });
});

test("SDK observation retains pipeline input/output names, EOS and unsupported-settings errors", async (t) => {
  const { root, model } = await fixture(t);
  const pipeline = path.join(root, "named.pipe-onnx");
  await writeFile(
    path.join(root, "config.json"),
    JSON.stringify({ architectures: ["GPT2LMHeadModel"] }),
  );
  await execute(python, [
    "-c",
    `
import sys
import onnx_light.onnx as onnx
from onnx_light.onnx import PipelineProto
model = onnx.load(sys.argv[1])
model.graph.input[0].name = "tokens"
model.graph.node[0].input.clear()
model.graph.node[0].input.extend(["weights", "tokens"])
model.graph.output[0].name = "prediction"
model.graph.node[0].output.clear()
model.graph.node[0].output.append("prediction")
pipeline = PipelineProto(prefill=model, input_ids_name="tokens", logits_name="prediction",
                         eos_token_ids=[2, 1], pad_token_id=2, seed=7)
with open(sys.argv[2], "wb") as stream: stream.write(pipeline.SerializeToString())
pipeline.top_k = 1
with open(sys.argv[3], "wb") as stream: stream.write(pipeline.SerializeToString())
`,
    model,
    pipeline,
    path.join(root, "unsupported.pipe-onnx"),
  ]);
  for (const sdkLoader of [false, true]) {
    const streamed = [];
    const result = await runPrompt(
      pipeline,
      { ...options, engine: "onnx-light", sessionSettings: { sdkLoader } },
      "/no/python",
      signal(),
      () => {},
      undefined,
      (token) => streamed.push(token),
    );
    assert.deepEqual(
      result.tokens.map((token) => token.id),
      [1],
    );
    assert.deepEqual(
      result.tokens.map((token) => token.score),
      [4],
    );
    assert.deepEqual(streamed, result.tokens);
    assert.ok(result.tokens[0].timeMs >= 0);
  }
  await assert.rejects(
    runPrompt(
      path.join(root, "unsupported.pipe-onnx"),
      { ...options, engine: "onnx-light" },
      "/no/python",
      signal(),
      () => {},
    ),
    /top-k/,
  );
});

for (const [engine, captureBudget, expectedScores] of [
  ["onnx-light", 0, [null, null, null]],
  ["onnx-light", 32, [4, 4, null]],
  ["onnxruntime-genai", 0, [4, 4, 4]],
]) {
  test(`${engine} timings survive the ${captureBudget}-byte vocabulary capture limit`, async (t) => {
    const { root, model } = await fixture(t);
    const input = path.join(root, "input.json");
    const output = path.join(root, "output.json");
    const capturePath = path.join(root, "capture.json");
    await writeFile(
      input,
      JSON.stringify({
        ...options,
        engine,
        modelPath: model,
        source: root,
        capturePath,
        captureBudget,
      }),
    );
    await execute(
      process.execPath,
      [path.resolve(__dirname, "../media/run_prompt.cjs"), input, output],
      { env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" } },
    );
    const result = JSON.parse(await readFile(output, "utf8"));
    assert.deepEqual(
      result.tokens.map((token) => token.score),
      expectedScores,
    );
    assert.ok(result.tokens.every((token) => token.timeMs >= 0));
    assert.match(result.limitations.join(" "), /retention budget/);
    assert.equal(JSON.parse(await readFile(capturePath, "utf8")).logits, null);
  });
}

test("native runtime and model failures surface without falling back to Python", async (t) => {
  const { root, model } = await fixture(t);
  process.env.ONNX_LIGHT_VIEWER_GENAI_LIBRARY = path.join(
    root,
    "missing-native-library",
  );
  for (const engine of ["onnx-light", "onnxruntime-genai"]) {
    await assert.rejects(
      runPrompt(
        model,
        { ...options, engine },
        "/no/python",
        signal(),
        () => {},
      ),
      /Could not load the native ONNX Runtime GenAI library/,
    );
  }
  assert.equal(
    (await readdir(root)).some((name) => name.startsWith(".prompt-")),
    false,
  );
});

test("rejects malformed options, tampering, changed assets and cancellation", async (t) => {
  const { root, model } = await fixture(t);
  for (const invalid of [
    null,
    { ...options, maxTokens: 0 },
    { ...options, maxTokens: 1.5 },
    { ...options, engine: "shell" },
    { ...options, validate: "yes" },
    { ...options, prompt: "" },
    { ...options, trust_remote_code: true },
    { ...options, referenceSource: 42 },
    { ...options, referenceSource: "" },
    { ...options, referenceSource: "relative/checkpoint/directory" },
    { ...options, sessionSettings: { intraOpThreads: -1 } },
    { ...options, sessionSettings: { threads: 2 } },
    { ...options, sessionSettings: { provider: "unknown" } },
    { ...options, sessionSettings: { gpuMemoryMiB: 128 } },
    { ...options, sessionSettings: JSON.parse('{"__proto__":1}') },
  ])
    await assert.rejects(
      runPrompt(model, invalid, "/no/python", signal(), () => {}),
    );
  const previous = await runPrompt(
    model,
    options,
    "/no/python",
    signal(),
    () => {},
  );
  for (const changed of [
    { ...previous, signature: "forged" },
    { ...previous, prompt: "different" },
    { ...previous, tokens: [{ text: "fake", id: 2, score: 999, timeMs: 1 }] },
    {
      ...previous,
      tokens: previous.tokens.map((token) => ({ ...token, timeMs: -1 })),
    },
  ])
    await assert.rejects(
      runPrompt(
        model,
        { ...options, validate: true },
        "/no/python",
        signal(),
        () => {},
        changed,
      ),
    );
  await writeFile(path.join(root, "tokenizer_config.json"), '{"changed":true}');
  await assert.rejects(
    runPrompt(
      model,
      { ...options, validate: true },
      "/no/python",
      signal(),
      () => {},
      previous,
    ),
    /previous prompt result/,
  );
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    runPrompt(model, options, "/no/python", controller.signal, () => {}),
    { name: "AbortError" },
  );
  assert.equal(
    (await readdir(root)).some((name) => name.startsWith(".prompt-")),
    false,
  );
});

test("validation invalidates captured logits when nested external weights change", async (t) => {
  const { root, model } = await fixture(t, true, true);
  const previous = await runPrompt(
    model,
    options,
    "/no/python",
    signal(),
    () => {},
  );
  const file = path.join(root, "weights", "decoder.bin");
  const weights = await readFile(file);
  weights.writeFloatLE(5, (3 * 4 + 1) * 4);
  await writeFile(file, weights);
  await assert.rejects(
    runPrompt(
      model,
      { ...options, validate: true },
      "/no/python",
      signal(),
      () => {},
      previous,
    ),
    /previous prompt result/,
  );
});

test("source tokenizer changes invalidate captures and replace the retained native session", async (t) => {
  const { root, model } = await fixture(t);
  const source = path.join(root, "checkpoint");
  await mkdir(source);
  for (const name of [
    "config.json",
    "genai_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
  ])
    await rename(path.join(root, name), path.join(source, name));
  await writeFile(
    provenancePath(model),
    JSON.stringify({
      version: 1,
      exporter: "mobius",
      source,
      model: "owner/model",
    }),
  );
  const session = new PromptSession();
  const settings = { ...options, engine: "onnx-light", keepLoaded: true };
  const run = () =>
    runPrompt(
      model,
      settings,
      "/no/python",
      signal(),
      () => {},
      undefined,
      undefined,
      session,
    );
  try {
    const previous = await run();
    assert.equal((await run()).lifecycle.reused, true);
    const file = path.join(source, "tokenizer.json");
    const tokenizer = JSON.parse(await readFile(file, "utf8"));
    tokenizer.model.vocab.a = tokenizer.model.vocab.o;
    delete tokenizer.model.vocab.o;
    await writeFile(file, JSON.stringify(tokenizer));
    await assert.rejects(
      runPrompt(
        model,
        { ...settings, validate: true },
        "/no/python",
        signal(),
        () => {},
        previous,
        undefined,
        session,
      ),
      /previous prompt result/,
    );
    const next = await run();
    assert.equal(next.lifecycle.reused, false);
    assert.equal(next.tokens[0].text, "a");
  } finally {
    await session.close();
  }
});

test("HF preflight rejects missing references and weights before any native loading or tokens", async (t) => {
  const { root, model } = await fixture(t);
  const phases = [];
  const tokens = [];
  for (const [reference, error] of [
    [{}, /No local Hugging Face source is available/],
    [
      { referenceSource: path.join(root, "missing-checkpoint") },
      /reference directory does not exist/,
    ],
    [{ referenceSource: root }, /checkpoint weights are missing/],
  ]) {
    await assert.rejects(
      runPrompt(
        model,
        { ...options, validate: true, ...reference },
        python,
        signal(),
        () => {},
        undefined,
        (token) => tokens.push(token),
        undefined,
        (phase) => phases.push(phase),
      ),
      error,
    );
    assert.deepEqual(phases, []);
    assert.deepEqual(tokens, []);
  }
  assert.equal(
    (await readdir(root)).some((name) => name.startsWith(".prompt-")),
    false,
  );
});

test("HF reference file checks need no model imports and catch missing checkpoint shards", async (t) => {
  const { root, model } = await fixture(t, false);
  const input = path.join(root, "check-reference.json");
  await writeFile(
    input,
    JSON.stringify({ modelPath: model, referenceSource: root }),
  );
  await writeFile(
    path.join(root, "model.safetensors.index.json"),
    JSON.stringify({ weight_map: { weight: "model-00001.safetensors" } }),
  );
  const check = () =>
    execute(python, [
      "-I",
      "-S",
      "media/run_prompt.py",
      "--check-reference",
      input,
    ]);
  await assert.rejects(check(), /weight shard is missing/);
  await writeFile(
    path.join(root, "model-00001.safetensors"),
    "checkpoint fixture",
  );
  await check();
  assert.equal(JSON.parse(await readFile(input, "utf8")).referenceSource, root);
});

test("HF prefers a pinned cached checkpoint and falls back to a complete export on cache miss", async (t) => {
  const { model } = await fixture(t, false);
  await execute(python, [
    "-B",
    "-c",
    `
import importlib.util
from pathlib import Path
import sys
import types
spec = importlib.util.spec_from_file_location("prompt", "media/run_prompt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
expected = str(Path(sys.argv[1]).parent / "pinned-checkpoint")
cached_result = str(Path(expected) / "config.json")
lookups = []
def cached_file(identity, filename, **kwargs):
    assert identity == "owner/checkpoint"
    assert filename == "config.json"
    lookups.append(kwargs)
    return cached_result
sys.modules["huggingface_hub"] = types.SimpleNamespace(try_to_load_from_cache=cached_file)
request = {"modelPath":sys.argv[1], "provenance":{
    "source":"/missing/original/source", "model":"owner/checkpoint", "revision":"pinned-revision"}}
assert module.source_for(request) == expected
assert lookups[-1] == {"revision":"pinned-revision"}
try: module.source_for({"modelPath":sys.argv[1]})
except ValueError: pass
else: raise AssertionError("Export-only directory accepted as a checkpoint")
directory = Path(sys.argv[1]).parent
for cached_result in (None, object()):
    (directory / "model.safetensors").write_text("checkpoint fixture")
    assert module.source_for(request) == str(directory)
    try:
        module.source_for({**request, "referenceSource": "owner/checkpoint"})
    except ValueError as error:
        assert "No cached Hugging Face config.json" in str(error)
    else:
        raise AssertionError("Explicit uncached reference ignored")
    assert lookups[-1] == {"revision":None}
    (directory / "model.safetensors").unlink()
`,
    model,
  ]);
});

test("HF candidate decoding preserves split UTF-8 against exported prefixes, not earlier predictions", async () => {
  await execute(python, [
    "-B",
    "-c",
    `
import importlib.util
from tokenizers import Tokenizer, decoders, models
spec = importlib.util.spec_from_file_location("prompt", "media/run_prompt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
backend = Tokenizer(models.BPE(vocab={"\\u00c3":0, "\\u00a9":1, "X":2, "H":3}, merges=[]))
backend.decoder = decoders.ByteLevel()
class Slow:
    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens":False, "clean_up_tokenization_spaces":False}
        return backend.decode(ids, skip_special_tokens=False)
class Fast(Slow):
    backend_tokenizer = backend
for tokenizer in (Fast(), Slow()):
    decoder = module.ReferenceDecoder(tokenizer, [3])
    prefix = [3]
    previews = []
    for candidate, exported in ((2, 0), (1, 1), (0, 3)):
        previews.append(decoder.decode(prefix, candidate, exported))
        prefix.append(exported)
    assert previews == ["X", "\\u00e9", ""], previews
    decoder = module.ReferenceDecoder(tokenizer, [3])
    assert decoder.decode([3], 0, 0) == ""
    assert decoder.decode([3, 0], 1, 1) == "\\u00e9"
`,
  ]);
});

test("HF reference uses the native capture, exported prefixes and cached decode timings", async (t) => {
  const { root, model } = await fixture(t);
  await writeFile(
    path.join(root, "model.safetensors"),
    "explicit HF test double",
  );
  await writeFile(
    path.join(root, "torch.py"),
    `
from contextlib import nullcontext
inference_mode = nullcontext
long = "long"
def tensor(value, **kwargs): return value
def ones(shape, **kwargs): return shape
`,
  );
  await writeFile(
    path.join(root, "transformers.py"),
    `
from types import SimpleNamespace
class Logits:
    def __getitem__(self, index): return self
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return [0,3.98,4,0]
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, source, **kwargs):
        assert kwargs == {"trust_remote_code":False,"local_files_only":True}
        return cls()
    def encode(self, prompt, **kwargs): return [3]
    def decode(self, ids, **kwargs): return str(ids[0])
class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, source, **kwargs):
        assert kwargs == {"trust_remote_code":False,"local_files_only":True}
        return cls()
    def eval(self): return self
    def __call__(self, input_ids, **kwargs):
        assert kwargs["use_cache"] is True
        step = kwargs.get("past_key_values", 0)
        assert input_ids == ([[3]] if step == 0 else [[1]])
        assert kwargs["attention_mask"] == (1, step + 1)
        return SimpleNamespace(logits=Logits(), past_key_values=step+1)
`,
  );
  const previousPath = process.env.PYTHONPATH;
  process.env.PYTHONPATH = [root, previousPath]
    .filter(Boolean)
    .join(path.delimiter);
  t.after(() => {
    if (previousPath === undefined) delete process.env.PYTHONPATH;
    else process.env.PYTHONPATH = previousPath;
  });
  for (const engine of ["onnx-light", "onnxruntime-genai"]) {
    const previous = await runPrompt(
      model,
      { ...options, engine },
      "/no/python",
      signal(),
      () => {},
    );
    const validated = await runPrompt(
      model,
      { ...options, engine, validate: true, referenceSource: root },
      python,
      signal(),
      () => {},
      previous,
    );
    assert.deepEqual(validated.tokens, previous.tokens);
    assert.ok(validated.reference.loadMs >= 0);
    assert.ok(validated.reference.warmupMs >= 0);
    assert.deepEqual(validated.lifecycle, previous.lifecycle);
    assert.deepEqual(
      validated.reference.tokens.map((token) => token.id),
      [2, 2, 2],
    );
    assert.ok(validated.reference.tokens.every((token) => token.timeMs >= 0));
    assert.ok(
      validated.discrepancies.every(
        (metric) => metric.maxAbsolute === 4 && metric.total === 4,
      ),
    );
  }
});

test("an explicit HF model ID resolves the local cache without export provenance or guessing a checkpoint", async (t) => {
  const { root, model } = await fixture(t, false);
  await execute(python, [
    "-B",
    "-c",
    `
import importlib.util
import os
from pathlib import Path
import sys

cache = Path(sys.argv[2]) / "hub"
repository = cache / "models--owner--checkpoint"
revision = "a" * 40
snapshot = repository / "snapshots" / revision
snapshot.mkdir(parents=True)
(snapshot / "config.json").write_text("{}")
(repository / "refs").mkdir()
(repository / "refs" / "main").write_text(revision)
os.environ["HF_HUB_CACHE"] = str(cache)
spec = importlib.util.spec_from_file_location("prompt", "media/run_prompt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
request = {"modelPath":sys.argv[1], "referenceSource":"owner/checkpoint"}
assert module.source_for(request) == str(snapshot)
request["provenance"] = {"source":sys.argv[2], "model":"wrong/checkpoint", "revision":"wrong-revision"}
assert module.source_for(request) == str(snapshot)
request["referenceSource"] = str(snapshot)
assert module.source_for(request) == str(snapshot)
request["referenceSource"] = "owner/uncached-checkpoint"
try:
    module.source_for(request)
except ValueError as error:
    assert "No cached Hugging Face config.json" in str(error)
else:
    raise AssertionError("Uncached explicit model silently ignored")
request["referenceSource"] = str(snapshot / "missing")
try:
    module.source_for(request)
except ValueError as error:
    assert "reference directory does not exist" in str(error)
else:
    raise AssertionError("Missing explicit checkpoint silently ignored")
try:
    module.source_for({"modelPath":sys.argv[1]})
except ValueError as error:
    assert "Hugging Face reference field" in str(error)
else:
    raise AssertionError("Export-only directory accepted without an explicit checkpoint")
`,
    model,
    root,
  ]);
});

test("numerical comparison covers all vocabulary entries and strict threshold boundaries", async () => {
  await execute(python, [
    "-B",
    "-c",
    `
import importlib.util
import math
spec = importlib.util.spec_from_file_location("prompt", "media/run_prompt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.compare_logits([0,.009,.01,.099,.1,.999,1], [0]*7)
assert result["below001"] == 2 and result["below01"] == 4 and result["below1"] == 6
assert result["total"] == 7 and result["maxAbsolute"] == 1
assert math.isfinite(result["relative"])
assert module.compare_logits([3,4], [3,0])["relative"] == 4/3
for left,right in [([],[]),([1],[1,2]),([float("nan")],[0]),([0],[float("inf")])]:
    try: module.compare_logits(left,right)
    except ValueError: pass
    else: raise AssertionError((left,right))
`,
  ]);
});
