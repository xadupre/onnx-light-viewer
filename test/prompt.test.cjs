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
  await writeFile(model, "invalid ONNX");
  await assert.rejects(run(), /decode|parse|wire|field/i);
});

test("prompt availability follows local causal metadata and supports native pipeline discovery", async (t) => {
  const { root, model } = await fixture(t, false);
  assert.deepEqual(await promptInfo(model), {
    engines: ["onnx-light", "onnxruntime-genai"],
    available: true,
  });
  await rm(path.join(root, "genai_config.json"));
  assert.deepEqual((await promptInfo(model)).engines, ["onnx-light"]);
  const pipeline = path.join(root, "model.pipe-onnx");
  await writeFile(pipeline, "");
  assert.deepEqual((await promptInfo(pipeline)).engines, ["onnx-light"]);
  await writeFile(
    path.join(root, "config.json"),
    JSON.stringify({ architectures: ["BertForSequenceClassification"] }),
  );
  assert.equal((await promptInfo(model)).available, false);
  assert.equal((await promptInfo("relative.onnx")).available, false);
});

test("snapshot-style metadata symlinks remain available", async (t) => {
  const { root, model } = await fixture(t, false);
  const blobs = path.join(root, "blobs");
  await mkdir(blobs);
  for (const name of [
    "config.json",
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

test("native GenAI streams real tokens and measures prefill and each decode step without Python", async (t) => {
  const { root, model } = await fixture(t);
  const streamed = [];
  const result = await runPrompt(
    model,
    { ...options, prompt: "HH" },
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

test("the first native token arrives before completion and cancellation stops the child", async (t) => {
  const { root, model } = await fixture(t);
  const controller = new AbortController();
  let received = 0;
  await assert.rejects(
    runPrompt(
      model,
      { ...options, maxTokens: 4096 },
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

test("native onnx-light generation reports unavailable streaming, timings and logits honestly", async (t) => {
  const { model } = await fixture(t);
  const result = await runPrompt(
    model,
    { ...options, engine: "onnx-light" },
    "/no/python",
    signal(),
    () => {},
  );
  assert.deepEqual(
    result.tokens.map((token) => token.id),
    [1, 1, 1],
  );
  assert.ok(
    result.tokens.every(
      (token) => token.score === null && token.timeMs === null,
    ),
  );
  assert.match(result.limitations.join(" "), /upstream SDK.*5692/);
});

test("native raw scores and timings survive the bounded vocabulary capture limit", async (t) => {
  const { root, model } = await fixture(t);
  const input = path.join(root, "input.json");
  const output = path.join(root, "output.json");
  const capturePath = path.join(root, "capture.json");
  await writeFile(
    input,
    JSON.stringify({
      ...options,
      modelPath: model,
      source: root,
      capturePath,
      captureBudget: 0,
    }),
  );
  await execute(
    process.execPath,
    [path.resolve(__dirname, "../media/run_prompt.cjs"), input, output],
    { env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" } },
  );
  const result = JSON.parse(await readFile(output, "utf8"));
  assert.ok(
    result.tokens.every((token) => token.score === 4 && token.timeMs >= 0),
  );
  assert.match(result.limitations.join(" "), /retention budget/);
  assert.equal(JSON.parse(await readFile(capturePath, "utf8")).logits, null);
});

test("native runtime and model failures surface without falling back to Python", async (t) => {
  const { root, model } = await fixture(t);
  process.env.ONNX_LIGHT_VIEWER_GENAI_LIBRARY = path.join(
    root,
    "missing-native-library",
  );
  await assert.rejects(
    runPrompt(model, options, "/no/python", signal(), () => {}),
    /Could not load the native ONNX Runtime GenAI library/,
  );
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

test("HF resolves a pinned cached checkpoint before an export-only tokenizer directory", async (t) => {
  const { model } = await fixture(t, false);
  await execute(python, [
    "-B",
    "-c",
    `
import importlib.util
import sys
import types
spec = importlib.util.spec_from_file_location("prompt", "media/run_prompt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def snapshot(identity, **kwargs):
    assert identity == "owner/checkpoint"
    assert kwargs == {"revision":"pinned-revision", "local_files_only":True}
    return "/local/pinned-checkpoint"
sys.modules["huggingface_hub"] = types.SimpleNamespace(snapshot_download=snapshot)
request = {"modelPath":sys.argv[1], "provenance":{
    "source":"/missing/original/source", "model":"owner/checkpoint", "revision":"pinned-revision"}}
assert module.source_for(request) == "/local/pinned-checkpoint"
try: module.source_for({"modelPath":sys.argv[1]})
except ValueError: pass
else: raise AssertionError("Export-only directory accepted as a checkpoint")
`,
    model,
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
  const previous = await runPrompt(
    model,
    options,
    "/no/python",
    signal(),
    () => {},
  );
  const validated = await runPrompt(
    model,
    { ...options, validate: true },
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
