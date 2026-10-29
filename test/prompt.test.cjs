const assert = require("node:assert/strict");
const { execFile } = require("node:child_process");
const {
  mkdtemp,
  mkdir,
  readFile,
  writeFile,
  readdir,
  rm,
  truncate,
} = require("node:fs/promises");
const path = require("node:path");
const { tmpdir } = require("node:os");
const { promisify } = require("node:util");
const { createRequire } = require("node:module");
const { runInNewContext } = require("node:vm");
const { test } = require("node:test");
const { promptInfo, runPrompt } = require("../dist/prompt");

const python = process.platform === "win32" ? "python" : "python3";
const execute = promisify(execFile);
const signal = () => new AbortController().signal;
const options = {
  engine: "onnxruntime-genai",
  prompt: "Hello ' $() 世界",
  maxTokens: 2,
};

test("the Python runner is bundled outside the VSIX-excluded scripts directory", async () => {
  const runner = path.resolve(
    path.dirname(require.resolve("../dist/prompt")),
    "..",
    "media",
    "run_prompt.py",
  );
  assert.match(await readFile(runner, "utf8"), /def generate_genai/);
  const ignored = await readFile(
    path.resolve(__dirname, "..", ".vscodeignore"),
    "utf8",
  );
  assert.match(ignored, /^scripts\/\*\*$/m);
  assert.doesNotMatch(ignored, /^(?:media\/\*\*|\*\*\/\*\.py)$/m);
});

async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), "onnx-viewer-prompt-test-"));
  t.after(() => rm(root, { force: true, recursive: true }));
  const model = path.join(root, "model.onnx");
  await writeFile(model, "explicit backend double");
  await writeFile(
    path.join(root, "genai_config.json"),
    JSON.stringify({
      model: {
        decoder: { filename: "model.onnx", outputs: { logits: "logits" } },
      },
    }),
  );
  await writeFile(
    path.join(root, "onnxruntime_genai.py"),
    `
class Row:
    shape = (1, 1, 3)
    def __init__(self, values): self.values = values
    def reshape(self, *args): return self
    def __getitem__(self, index): return self
    def astype(self, dtype): return self
    def tolist(self): return list(self.values)
class Model:
    def __init__(self, directory): pass
class TokenScalar:
    def __init__(self, value): self.value = value
    def __int__(self): return self.value
class Tokenizer:
    def __init__(self, model): pass
    def encode(self, prompt):
        assert prompt == ${JSON.stringify(options.prompt)}
        return [TokenScalar(0)]
    def create_stream(self): return self
    def decode(self, token): return ["zero", "one", "two"][token]
class GeneratorParams:
    def __init__(self, model): pass
    def set_search_options(self, **options):
        assert options == {"max_length": 3, "do_sample": False, "num_beams":1, "num_return_sequences":1}
class Generator:
    def __init__(self, model, params):
        import pathlib
        marker = pathlib.Path(__file__).parent / ".generation-count"
        marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else "1")
        self.step = 0
    def append_tokens(self, tokens): assert tokens == [0]
    def is_done(self): return self.step == 2
    def generate_next_token(self):
        self.step += 1
        self.row.values[:] = [-100,-100,-100]
    def get_next_tokens(self): return [self.step]
    def get_logits(self):
        self.row = Row([0,4,0] if self.step == 0 else [0,0,5])
        return self.row
`,
  );
  const previous = process.env.PYTHONPATH;
  process.env.PYTHONPATH = [root, previous]
    .filter(Boolean)
    .join(path.delimiter);
  t.after(() => {
    if (previous === undefined) delete process.env.PYTHONPATH;
    else process.env.PYTHONPATH = previous;
  });
  return { root, model };
}

test("prompt availability is based on GenAI assets, not an assumed LLM identity", async (t) => {
  const { root, model } = await fixture(t);
  assert.deepEqual(await promptInfo(model), {
    engines: ["onnxruntime-genai"],
    available: true,
  });
  await rm(path.join(root, "genai_config.json"));
  const unavailable = await promptInfo(model);
  assert.equal(unavailable.available, false);
  assert.match(unavailable.reason, /local causal-model\/tokenizer metadata/);
  await writeFile(path.join(root, "tokenizer_config.json"), "{}");
  await writeFile(
    path.join(root, "config.json"),
    JSON.stringify({ architectures: ["BertForSequenceClassification"] }),
  );
  assert.equal((await promptInfo(model)).available, false);
  await writeFile(
    path.join(root, "config.json"),
    JSON.stringify({ architectures: ["GPT2LMHeadModel"] }),
  );
  assert.deepEqual((await promptInfo(model)).engines, ["onnx-light"]);
  assert.equal((await promptInfo("relative.onnx")).available, false);
});

test("runs isolated Python with preserved PYTHONPATH, raw scores and safe prompt transport", async (t) => {
  const { root, model } = await fixture(t);
  const output = [];
  const result = await runPrompt(model, options, python, signal(), (s) =>
    output.push(s),
  );
  assert.equal(result.prompt, options.prompt);
  assert.equal(result.engine, options.engine);
  assert.deepEqual(result.tokens, [
    { text: "one", id: 1, score: 4 },
    { text: "two", id: 2, score: 5 },
  ]);
  assert.match(result.signature, /^[0-9a-f]{64}$/);
  assert.match(output.at(-1), /Generated 2\/2 tokens/);
  assert.equal(
    (await readdir(root)).some((entry) => entry.startsWith(".prompt-")),
    false,
  );
});

test("full vocabulary retention stops at the budget while selected raw scores remain available", async (t) => {
  const { root, model } = await fixture(t);
  const input = path.join(root, "request.json");
  const output = path.join(root, "result.json");
  const capture = path.join(root, "capture.json");
  await writeFile(
    input,
    JSON.stringify({
      modelPath: model,
      ...options,
      validate: false,
      capturePath: capture,
      captureBudget: 0,
    }),
  );
  await execute(python, [
    "-B",
    path.resolve(__dirname, "..", "media", "run_prompt.py"),
    input,
    output,
  ]);
  const result = JSON.parse(await readFile(output, "utf8"));
  assert.deepEqual(
    result.tokens.map((token) => token.score),
    [4, 5],
  );
  assert.match(result.limitations.join(" "), /retention budget/);
  assert.equal(JSON.parse(await readFile(capture, "utf8")).logits, null);
});

test("onnx-light uses native generation only with null scores and explicit external weights", async (t) => {
  const { root, model } = await fixture(t);
  await rm(path.join(root, "genai_config.json"));
  await writeFile(
    path.join(root, "config.json"),
    JSON.stringify({ architectures: ["GPT2LMHeadModel"] }),
  );
  await writeFile(path.join(root, "tokenizer_config.json"), "{}");
  await writeFile(
    path.join(root, "numpy.py"),
    `
int64 = "int64"
def asarray(value, dtype=None): return value
class Generated:
    def tolist(self): return [[0,1,2]]
`,
  );
  await writeFile(
    path.join(root, "transformers.py"),
    `
class AutoTokenizer:
    eos_token_id = 2
    @classmethod
    def from_pretrained(cls, source, **kwargs):
        assert kwargs == {"trust_remote_code": False, "local_files_only": True}
        return cls()
    def encode(self, prompt, **kwargs):
        assert isinstance(kwargs["add_special_tokens"], bool)
        return [0]
    def decode(self, ids, **kwargs): return ["zero","one","two"][ids[0]]
class Logits:
    def __init__(self, values): self.values = values
    def __getitem__(self, key): return self
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return self.values
class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, source, **kwargs):
        assert kwargs == {"trust_remote_code":False, "local_files_only":True}
        return cls()
    def eval(self): return self
    def __call__(self, input_ids, **kwargs):
        from types import SimpleNamespace
        assert input_ids in ([[0]], [[0,1]])
        return SimpleNamespace(logits=Logits([4,0,0] if input_ids == [[0]] else [0,5,0]))
`,
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
  const sdk = path.join(root, "onnx_light", "onnx");
  await mkdir(sdk, { recursive: true });
  await writeFile(path.join(root, "onnx_light", "__init__.py"), "");
  await writeFile(
    path.join(sdk, "__init__.py"),
    `
def load(path, **kwargs):
    assert kwargs == {"load_external_data":True}
    return path
`,
  );
  await writeFile(
    path.join(sdk, "reference.py"),
    `
import pathlib
from numpy import Generated
class ReferenceEvaluator:
    def __init__(self, model): self.model = model
    def generate(self, feeds, **kwargs):
        if "missing-cache" in self.model:
            raise RuntimeError("SDK Generate: missing graph feed past_key_values")
        assert kwargs["max_new_tokens"] == 2 and kwargs["temperature"] == 0
        assert kwargs["eos_token_id"] == 2
        assert feeds == {"input_ids":[[0]]}
        marker = pathlib.Path(__file__).parents[2] / ".native-generation-count"
        marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else "1")
        return Generated()
    def run(self, *args, **kwargs): raise AssertionError("Viewer must not replay inference")
`,
  );
  assert.deepEqual((await promptInfo(model)).engines, ["onnx-light"]);
  const result = await runPrompt(
    model,
    { ...options, engine: "onnx-light" },
    python,
    signal(),
    () => {},
  );
  assert.deepEqual(result.tokens, [
    { text: "one", id: 1, score: null },
    { text: "two", id: 2, score: null },
  ]);
  assert.match(result.limitations.join(" "), /exposes token IDs only/);
  const validated = await runPrompt(
    model,
    { ...options, engine: "onnx-light", validate: true },
    python,
    signal(),
    () => {},
    result,
  );
  assert.deepEqual(validated.reference.tokens, [
    { text: "zero", id: 0, score: 4 },
    { text: "one", id: 1, score: 5 },
  ]);
  assert.equal(validated.discrepancies, undefined);
  const pipeline = path.join(root, "model.pipe-onnx");
  await writeFile(pipeline, "explicit split-pipeline double");
  assert.equal((await promptInfo(pipeline)).available, false);
  assert.equal(
    await readFile(path.join(root, ".native-generation-count"), "utf8"),
    "1",
  );
  const unsupported = path.join(root, "separate-decode.pipe-onnx");
  await writeFile(unsupported, "unsupported separate decode double");
  await assert.rejects(
    runPrompt(
      unsupported,
      { ...options, engine: "onnx-light" },
      python,
      signal(),
      () => {},
    ),
    /does not directly execute PipelineProto/,
  );
  const unresolved = path.join(root, "missing-cache.onnx");
  await writeFile(unresolved, "unresolved required model input double");
  await assert.rejects(
    runPrompt(
      unresolved,
      { ...options, engine: "onnx-light" },
      python,
      signal(),
      () => {},
    ),
    /SDK Generate: missing graph feed past_key_values/,
  );
});

test("rejects malformed options, stale or tampered previous results and cancellation", async (t) => {
  const { root, model } = await fixture(t);
  for (const invalid of [
    null,
    { ...options, maxTokens: 0 },
    { ...options, maxTokens: 1.5 },
    { ...options, engine: "shell" },
    { ...options, validate: "yes" },
    { ...options, prompt: "" },
    { ...options, trust_remote_code: true },
  ])
    await assert.rejects(runPrompt(model, invalid, python, signal(), () => {}));
  const previous = await runPrompt(model, options, python, signal(), () => {});
  const rerun = await runPrompt(
    model,
    options,
    python,
    signal(),
    () => {},
    previous,
  );
  assert.deepEqual(rerun.tokens, previous.tokens);
  assert.equal(
    await readFile(path.join(root, ".generation-count"), "utf8"),
    "2",
  );
  const validate = { ...options, validate: true };
  for (const changed of [
    { ...previous, signature: "forged" },
    { ...previous, prompt: "different" },
    { ...previous, tokens: [{ text: "fake", id: 2, score: 999 }] },
  ])
    await assert.rejects(
      runPrompt(model, validate, python, signal(), () => {}, changed),
      /previous prompt result/,
    );
  await writeFile(model, "changed model bytes");
  await assert.rejects(
    runPrompt(model, validate, python, signal(), () => {}, previous),
    /previous prompt result/,
  );
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    runPrompt(model, options, python, controller.signal, () => {}),
    { name: "AbortError" },
  );
  assert.equal(
    (await readdir(root)).some((x) => x.startsWith(".prompt-")),
    false,
  );
});

test("validates an authenticated previous result against the same teacher-forced tokens", async (t) => {
  const { root, model } = await fixture(t);
  await writeFile(path.join(root, "config.json"), "{}");
  await writeFile(path.join(root, "tokenizer_config.json"), "{}");
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
    def __init__(self, values): self.values = values
    def __getitem__(self, index): return self
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return self.values
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, source, **kwargs):
        assert kwargs == {"trust_remote_code": False, "local_files_only": True}
        return cls()
    def encode(self, prompt, **kwargs): return [0]
    def decode(self, ids, **kwargs): return ["zero","one","two"][ids[0]]
class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, source, **kwargs):
        assert kwargs == {"trust_remote_code": False, "local_files_only": True}
        return cls()
    def eval(self): return self
    def __call__(self, input_ids, **kwargs):
        assert input_ids in ([[0]], [[0,1]])
        row = [0,4,.02] if input_ids == [[0]] else [0,6,5]
        return SimpleNamespace(logits=Logits(row))
`,
  );
  const previous = await runPrompt(model, options, python, signal(), () => {});
  const validated = await runPrompt(
    model,
    { ...options, validate: true },
    python,
    signal(),
    () => {},
    previous,
  );
  assert.deepEqual(validated.tokens, previous.tokens);
  assert.deepEqual(
    validated.reference.tokens.map(({ id, score }) => ({ id, score })),
    [
      { id: 1, score: 4 },
      { id: 1, score: 6 },
    ],
  );
  assert.equal(validated.discrepancies[0].maxAbsolute, 0.02);
  assert.equal(validated.discrepancies[1].maxAbsolute, 6);
  assert.ok(validated.discrepancies.every((metric) => metric.total === 3));
  assert.equal(
    await readFile(path.join(root, ".generation-count"), "utf8"),
    "1",
  );
  assert.equal("logits" in previous, false);
  assert.equal("promptIds" in previous, false);
  assert.deepEqual(Object.keys(previous).sort(), [
    "engine",
    "prompt",
    "signature",
    "tokens",
  ]);
  assert.equal(
    (await readdir(root)).some((x) => x.startsWith(".prompt-")),
    false,
  );
});

test("tokens-only results can retain HF predictions without discrepancy metrics", async (t) => {
  const { model } = await fixture(t);
  const compiled = require.resolve("../dist/prompt");
  const backend = {};
  let oversize = false;
  const localRequire = createRequire(compiled);
  runInNewContext(await readFile(compiled, "utf8"), {
    exports: backend,
    Error,
    __dirname: path.dirname(compiled),
    require(name) {
      if (name !== "./export") return localRequire(name);
      return {
        ...localRequire(name),
        async runProcess(plan) {
          const request = JSON.parse(await readFile(plan.args[2], "utf8"));
          const tokens = [{ text: "zero", id: 0, score: null }];
          const result = {
            prompt: request.prompt,
            engine: request.engine,
            tokens,
            limitations: ["Full engine vocabulary logits are unavailable."],
          };
          if (request.validate)
            result.reference = {
              tokens: [{ text: "one", id: 1, score: 4 }],
            };
          if (!request.reuseCapture)
            await writeFile(
              request.capturePath,
              JSON.stringify({ promptIds: [0], tokens, logits: null }),
            );
          if (oversize)
            await truncate(request.capturePath, 256 * 1024 * 1024 + 1);
          await writeFile(plan.args[3], JSON.stringify(result));
        },
      };
    },
  });
  const previous = await backend.runPrompt(
    model,
    options,
    python,
    signal(),
    () => {},
  );
  const result = await backend.runPrompt(
    model,
    { ...options, validate: true },
    python,
    signal(),
    () => {},
    previous,
  );
  assert.equal(result.tokens[0].score, null);
  assert.equal(result.reference.tokens[0].id, 1);
  assert.equal(result.reference.tokens[0].score, 4);
  assert.equal(result.discrepancies, undefined);
  assert.ok(result.limitations.length);
  oversize = true;
  await assert.rejects(
    backend.runPrompt(model, options, python, signal(), () => {}),
    /capture exceeds the retention budget/,
  );
});

test("numerical comparison covers every vocabulary entry and strict threshold boundaries", async () => {
  await execute(python, [
    "-B",
    "-c",
    `
import importlib.util
import math
spec = importlib.util.spec_from_file_location("prompt", "media/run_prompt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.compare_logits([0, .009, .01, .099, .1, .999, 1], [0]*7)
assert result["below001"] == 2, result
assert result["below01"] == 4, result
assert result["below1"] == 6, result
assert result["total"] == 7
assert result["maxAbsolute"] == 1
assert math.isclose(result["l2"], math.sqrt(sum(x*x for x in [0,.009,.01,.099,.1,.999,1])))
assert math.isfinite(result["relative"])
rows, used = module.retain_row([], [1,2,3], 0, 1000)
assert rows == [[1,2,3]] and used <= 1000
rows, used = module.retain_row(rows, [1]*100, used, 1000)
assert rows is None
rows, used = module.retain_row(rows, [1], used, 1000)
assert rows is None
assert module.compare_logits([3,4], [3,0])["relative"] == 4/3
for left, right in [([], []), ([1], [1,2]), ([float("nan")], [0]), ([0], [float("inf")])]:
    try: module.compare_logits(left, right)
    except ValueError: pass
    else: raise AssertionError((left, right))
`,
  ]);
});

test("HF predictions may diverge while comparison still teacher-forces exported prefixes", async () => {
  await execute(python, [
    "-B",
    "-c",
    `
import contextlib
import importlib.util
import sys
import types
spec = importlib.util.spec_from_file_location("prompt", "media/run_prompt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prefixes = []
class Logits:
    def __getitem__(self, item): return self
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return [1,2,3]
class Model:
    def eval(self): return self
    def __call__(self, **kwargs):
        prefixes.append(kwargs["input_ids"][0][:])
        assert kwargs["use_cache"] is False
        return types.SimpleNamespace(logits=Logits())
class Tokenizer:
    def encode(self, prompt, **kwargs):
        assert prompt == "same prompt" and isinstance(kwargs["add_special_tokens"], bool)
        return [0,1]
    def decode(self, ids, **kwargs): return str(ids[0])
def loader(result):
    def load(source, **kwargs):
        assert source == "local-cache"
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return result
    return types.SimpleNamespace(from_pretrained=load)
sys.modules["torch"] = types.SimpleNamespace(
    inference_mode=contextlib.nullcontext, long="long",
    tensor=lambda value, **kw: value, ones=lambda shape, **kw: shape)
sys.modules["transformers"] = types.SimpleNamespace(
    AutoModelForCausalLM=loader(Model()), AutoTokenizer=loader(Tokenizer()))
module.source_for = lambda request: "local-cache"
request = {"prompt": "same prompt"}
tokens = [{"id": 1}, {"id": 0}]
reference, metrics = module.reference_logits(request, [0,1], tokens, [[1,2,3],[1,2,3]])
assert prefixes == [[0,1], [0,1,1]], prefixes
assert [x["id"] for x in reference["tokens"]] == [2,2]
assert [x["score"] for x in reference["tokens"]] == [3,3]
assert all(x["total"] == 3 and x["maxAbsolute"] == 0 for x in metrics)
prefixes.clear()
reference, metrics = module.reference_logits(request, [0,1], tokens, None)
assert prefixes == [[0,1], [0,1,1]], prefixes
assert metrics is None
assert [x["id"] for x in reference["tokens"]] == [2,2]
try: module.reference_logits(request, [9], tokens, [[1,2,3],[1,2,3]])
except ValueError as error: assert "tokenizers" in str(error)
else: raise AssertionError("Mismatched tokenization accepted")
`,
  ]);
});
