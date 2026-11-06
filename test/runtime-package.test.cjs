const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const {
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
  link,
} = require("node:fs/promises");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const { ZipFile } = require("yazl");
const { PackageArchiveStore } = require("../dist/packageArchive");
const {
  inspectRuntimePackage,
  runtimePackageModel,
  runtimePackagePromptDirectory,
  saveRuntimePackage,
} = require("../dist/runtimePackage");
const { loadModelFile } = require("../dist/graph");

async function fixture(t) {
  const directory = await mkdtemp(
    path.join(tmpdir(), "viewer-package-zip-test-"),
  );
  const store = new PackageArchiveStore(directory);
  t.after(async () => {
    await store.close();
    await rm(directory, { recursive: true, force: true });
  });
  return { directory, store };
}

async function zipBytes(entries) {
  const zip = new ZipFile();
  const chunks = [];
  const complete = new Promise((resolve, reject) => {
    zip.outputStream.on("data", (chunk) => chunks.push(chunk));
    zip.outputStream.on("end", () => resolve(Buffer.concat(chunks)));
    zip.on("error", reject);
    zip.outputStream.on("error", reject);
  });
  for (const [name, bytes, options] of entries)
    if (name.endsWith("/")) zip.addEmptyDirectory(name);
    else
      zip.addBuffer(Buffer.from(bytes), name, { compress: false, ...options });
  zip.end({ forceZip64Format: true });
  return complete;
}

async function publish(source, target) {
  await link(source, target);
  await rm(source);
}

test("ORT package ZIPs preserve external component manifests, variants and assets on format conversion", async (t) => {
  const { directory, store } = await fixture(t);
  const bytes = await readFile(require.resolve("./models/relu.onnx"));
  const manifest = {
    schema_version: "1.0",
    package_name: "test package",
    components: { decoder: "decoder" },
    additional_metadata: { preserve: true },
  };
  const component = {
    variants: {
      cpu: {
        ep: "CPUExecutionProvider",
        executor_info: { ort: "ort_info.json" },
      },
      other: {
        ep: "OtherExecutionProvider",
        executor_info: { ort: { model_file: "model.onnx" } },
      },
      empty: {},
    },
  };
  const prefix = "wrapped.ortpackage/";
  const source = path.join(directory, "source.ortpackage");
  await writeFile(
    source,
    await zipBytes([
      [`${prefix}manifest.json`, JSON.stringify(manifest)],
      [`${prefix}decoder/component.json`, JSON.stringify(component)],
      [
        `${prefix}decoder/cpu/ort_info.json`,
        JSON.stringify({
          model_file: "model.onnx",
          session_options: { "session.intra_op_num_threads": "2" },
        }),
      ],
      [`${prefix}decoder/cpu/model.onnx`, bytes, { compress: true }],
      [`${prefix}decoder/other/model.onnx`, bytes],
      [`${prefix}provider/context.bin`, "compiled provider assets"],
      [`${prefix}decoder/empty/`, ""],
    ]),
  );
  const location = await store.resolve(source);
  const info = {
    ...(await inspectRuntimePackage(location.directory)),
    archivePath: source,
  };
  assert.equal(info.defaultModel, undefined);
  assert.deepEqual(info.models, [
    "decoder/cpu/model.onnx",
    "decoder/other/model.onnx",
  ]);
  const modelPath = runtimePackageModel(info, "decoder/cpu/model.onnx");
  const destination = path.join(directory, "saved.ortpackage");
  await saveRuntimePackage({
    destination,
    kind: "generic",
    sourcePackage: info,
    sourcePath: modelPath,
    model: await loadModelFile(modelPath, path.basename(modelPath)),
    format: "ort",
    includeRewritings: false,
    canPublish: () => true,
    publish,
  });
  assert.deepEqual(
    (await readFile(source)).subarray(0, 4),
    Buffer.from("504b0304", "hex"),
  );
  const saved = await store.resolve(destination);
  const savedInfo = await inspectRuntimePackage(saved.directory);
  assert.deepEqual(
    JSON.parse(await readFile(path.join(saved.directory, "manifest.json"))),
    manifest,
  );
  assert.deepEqual(
    JSON.parse(
      await readFile(path.join(saved.directory, "decoder/component.json")),
    ),
    component,
  );
  const ortInfo = JSON.parse(
    await readFile(path.join(saved.directory, "decoder/cpu/ort_info.json")),
  );
  assert.equal(ortInfo.model_file, "model.ort");
  assert.equal(ortInfo.session_options["session.intra_op_num_threads"], "2");
  assert.deepEqual(
    await readFile(path.join(saved.directory, "decoder/other/model.onnx")),
    bytes,
  );
  assert.equal(
    await readFile(path.join(saved.directory, "provider/context.bin"), "utf8"),
    "compiled provider assets",
  );
  assert.deepEqual(
    await readdir(path.join(saved.directory, "decoder/empty")),
    [],
  );
  assert.equal(
    (await readFile(runtimePackageModel(savedInfo, "decoder/cpu/model.ort")))
      .subarray(4, 8)
      .toString(),
    "ORTM",
  );
});

test("saving edited shared models retains hashed assets and updates every referring variant", async (t) => {
  const { directory, store } = await fixture(t);
  const bytes = await readFile(require.resolve("./models/relu.onnx"));
  const digest = createHash("sha256")
    .update(`${createHash("sha256").update(bytes).digest("hex")}  model.onnx\n`)
    .digest("hex");
  const uri = `sha256:${digest}/model.onnx`;
  const manifest = {
    schema_version: "1.0",
    components: {
      model: {
        variants: Object.fromEntries(
          ["cpu", "gpu"].map((name) => [
            name,
            { executor_info: { ort: { model_file: uri } } },
          ]),
        ),
      },
    },
  };
  const source = path.join(directory, "shared.ortpackage");
  await writeFile(
    source,
    await zipBytes([
      ["manifest.json", JSON.stringify(manifest)],
      [`shared_assets/sha256-${digest}/model.onnx`, bytes],
      ["cpu/placeholder", ""],
      ["gpu/placeholder", ""],
    ]),
  );
  const location = await store.resolve(source);
  const info = {
    ...(await inspectRuntimePackage(location.directory)),
    archivePath: source,
  };
  const modelPath = runtimePackageModel(info);
  const destination = path.join(directory, "edited.ortpackage");
  await saveRuntimePackage({
    destination,
    kind: "generic",
    sourcePackage: info,
    sourcePath: modelPath,
    model: await loadModelFile(modelPath, path.basename(modelPath)),
    format: "ort",
    includeRewritings: false,
    canPublish: () => true,
    publish,
  });
  const saved = await store.resolve(destination);
  const savedManifest = JSON.parse(
    await readFile(path.join(saved.directory, "manifest.json")),
  );
  for (const variant of Object.values(savedManifest.components.model.variants))
    assert.equal(variant.executor_info.ort.model_file, "model.ort");
  for (const name of ["cpu", "gpu"])
    assert.equal(
      (await readFile(path.join(saved.directory, name, "model.ort")))
        .subarray(4, 8)
        .toString(),
      "ORTM",
    );
  assert.deepEqual(
    await readFile(
      path.join(
        saved.directory,
        "shared_assets",
        `sha256-${digest}`,
        "model.onnx",
      ),
    ),
    bytes,
  );
});

test("generic package ZIPs restore external initializer payloads after extraction", async (t) => {
  const { directory, store } = await fixture(t);
  const sourcePath = require.resolve("../media/tiny-llm.onnx");
  const model = await loadModelFile(sourcePath, "tiny-llm.onnx");
  const destination = path.join(directory, "external.ortpackage");
  await saveRuntimePackage({
    destination,
    kind: "generic",
    model,
    sourcePath,
    format: "onnx",
    maxExternalFileSize: 1024 * 1024,
    includeRewritings: false,
    canPublish: () => true,
    publish,
  });
  const location = await store.resolve(destination);
  const info = await inspectRuntimePackage(location.directory);
  const modelPath = runtimePackageModel(info);
  const loaded = await loadModelFile(modelPath, path.basename(modelPath));
  assert.ok(loaded.externalDataLocations().length > 0);
  for (const initializer of model.graph().initializers)
    assert.deepEqual(
      await loaded.tensorStatistics(initializer.name),
      await model.tensorStatistics(initializer.name),
    );
});

test("shared GenAI models do not silently pick one of several variant configurations", async (t) => {
  const { directory, store } = await fixture(t);
  const bytes = await readFile(require.resolve("./models/relu.onnx"));
  const digest = createHash("sha256")
    .update(`${createHash("sha256").update(bytes).digest("hex")}  model.onnx\n`)
    .digest("hex");
  const source = path.join(directory, "genai-variants.ortpackage");
  const manifest = {
    schema_version: "1.0",
    components: { model: { variants: { cpu: {}, gpu: {} } } },
  };
  const config = JSON.stringify({
    model: { decoder: { filename: `sha256:${digest}/model.onnx` } },
  });
  await writeFile(
    source,
    await zipBytes([
      ["manifest.json", JSON.stringify(manifest)],
      [`shared_assets/sha256-${digest}/model.onnx`, bytes],
      ["cpu/genai_config.json", config],
      ["cpu/tokenizer.json", "{}"],
      ["gpu/genai_config.json", config],
      ["gpu/tokenizer.json", "{}"],
    ]),
  );
  const location = await store.resolve(source);
  const info = await inspectRuntimePackage(location.directory);
  const modelPath = runtimePackageModel(info);
  assert.equal(
    (await loadModelFile(modelPath, "model.onnx")).graph().nodes[0].opType,
    "Relu",
  );
  assert.throws(
    () => runtimePackagePromptDirectory(info, modelPath),
    /several GenAI configurations/,
  );
});

test("ZIP extraction rejects traversal, links, duplicate members and corrupt data without retaining partial assets", async (t) => {
  const { directory, store } = await fixture(t);
  const safe = await zipBytes([["safe.onnx", "checksum-test-payload"]]);
  const traversal = Buffer.from(safe);
  for (let start = 0; ;) {
    const offset = traversal.indexOf("safe.onnx", start);
    if (offset < 0) break;
    traversal.write("../a.onnx", offset);
    start = offset + 9;
  }
  const corrupt = Buffer.from(safe);
  corrupt[corrupt.indexOf("checksum-test-payload")] ^= 1;
  const cases = [
    ["traversal", traversal, /relative path|Unsafe ZIP|invalid relative path/i],
    [
      "symlink",
      await zipBytes([["link.onnx", "outside", { mode: 0o120777 }]]),
      /links and special files/,
    ],
    [
      "duplicate",
      await zipBytes([
        ["model.onnx", "first"],
        ["model.onnx", "second"],
      ]),
      /Duplicate ZIP/,
    ],
    ["checksum", corrupt, /checksum mismatch/],
    [
      "truncated",
      safe.subarray(0, safe.length - 50),
      /end of central directory|invalid|zip64/i,
    ],
  ];
  for (const [name, bytes, error] of cases) {
    const source = path.join(directory, `${name}.ortpackage`);
    await writeFile(source, bytes);
    await assert.rejects(store.resolve(source), error, name);
    assert.equal(
      (await readdir(directory)).some((name) =>
        name.startsWith("onnx-light-viewer-package-"),
      ),
      false,
    );
  }
  await assert.rejects(readFile(path.join(directory, "a.onnx")), {
    code: "ENOENT",
  });
});

test("extracted ZIPs remain usable during active model actions and are released when the owner closes", async (t) => {
  const { directory, store } = await fixture(t);
  const source = path.join(directory, "model.ortpackage");
  const bytes = await readFile(require.resolve("./models/relu.onnx"));
  await writeFile(source, await zipBytes([["model.onnx", bytes]]));
  const location = await store.resolve(source);
  assert.deepEqual(await store.resolve(source), location);
  await rm(source);
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const action = store.use(async () => {
    await gate;
    return (
      await loadModelFile(
        path.join(location.directory, "model.onnx"),
        "model.onnx",
      )
    ).graph();
  });
  const closing = store.close();
  assert.deepEqual(
    await readFile(path.join(location.directory, "model.onnx")),
    bytes,
  );
  release();
  assert.equal((await action).nodes[0].opType, "Relu");
  await closing;
  await assert.rejects(readdir(location.directory), { code: "ENOENT" });
});

test("failed or cancelled ZIP saves preserve their source and leave no staging directory", async (t) => {
  const { directory } = await fixture(t);
  const model = await loadModelFile(
    require.resolve("./models/relu.onnx"),
    "relu.onnx",
  );
  const destination = path.join(directory, "saved.ortpackage");
  const options = {
    destination,
    kind: "generic",
    model,
    format: "onnx",
    includeRewritings: false,
    canPublish: () => true,
    publish,
  };
  await assert.rejects(
    saveRuntimePackage({
      ...options,
      publish: async () => {
        throw new Error("publication failure");
      },
    }),
    /publication failure/,
  );
  assert.deepEqual(await readdir(directory), []);
  assert.equal(
    await saveRuntimePackage({ ...options, canPublish: () => false }),
    undefined,
  );
  assert.deepEqual(await readdir(directory), []);
  await saveRuntimePackage(options);
  const bytes = await readFile(destination);
  await assert.rejects(saveRuntimePackage(options), /already exists/);
  assert.deepEqual(await readFile(destination), bytes);
});
