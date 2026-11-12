const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { mkdir, readFile, readdir, writeFile } = require("node:fs/promises");
const { basename, join } = require("node:path");
const { test } = require("node:test");
const { loadModelFile } = require("../dist/graph");
const { foundryHub } = require("./helpers/foundry-hub.cjs");
const {
  filterFoundryModels,
  inspectFoundryMemory,
} = require("../dist/foundry");

test("Foundry memory filtering compares execution estimates, not download size, against the correct memory pool", () => {
  const GiB = 1024 ** 3;
  const base = {
    name: "fixture",
    device: "cpu",
    provider: "CPUExecutionProvider",
    size: 1,
    license: "MIT",
  };
  const models = [
    { ...base, id: "cpu-too-large", memory: 33 * GiB },
    { ...base, id: "cpu-exact", memory: 32 * GiB },
    { ...base, id: "large-download", size: 80 * GiB, memory: 2 * GiB },
    {
      ...base,
      id: "cuda-too-large",
      device: "gpu",
      provider: "CUDAExecutionProvider",
      memory: 9 * GiB,
    },
    {
      ...base,
      id: "cuda-exact",
      device: "gpu",
      provider: "CUDAExecutionProvider",
      memory: 8 * GiB,
    },
    { ...base, id: "npu-too-large", device: "npu", memory: 40 * GiB },
    { ...base, id: "unknown-footprint" },
    {
      ...base,
      id: "unknown-gpu",
      device: "gpu",
      provider: "OpenVINOExecutionProvider",
      memory: 40 * GiB,
    },
  ];
  const filtered = filterFoundryModels(models, {
    ram: 32 * GiB,
    nvidia: 8 * GiB,
  });
  assert.deepEqual(
    filtered.models.map(({ id }) => id),
    [
      "cpu-exact",
      "large-download",
      "cuda-exact",
      "unknown-footprint",
      "unknown-gpu",
    ],
  );
  assert.equal(filtered.hidden, 3);
  assert.equal(filtered.unchecked, 2);
  assert.equal(filtered.models[0].memoryLimit, 32 * GiB);
  assert.equal(filtered.models[2].memoryLimit, 8 * GiB);
  const unknown = filterFoundryModels(models, {
    ram: 32 * GiB,
    warning: "GPU query unavailable",
  });
  assert.ok(unknown.models.some(({ id }) => id === "cuda-too-large"));
  assert.equal(unknown.memory.warning, "GPU query unavailable");
});

test("Foundry memory filtering reads the host RAM ceiling and ignores unknown rather than inventing estimates", async (t) => {
  const hub = await foundryHub(t);
  const memory = await inspectFoundryMemory(false);
  assert.ok(memory.ram > 0 && memory.ram <= require("node:os").totalmem());
  const constrained = process.constrainedMemory();
  if (constrained > 0) assert.ok(memory.ram <= constrained);
  const metadata = hub.catalogEntry.properties.variantInfo.variantMetadata;
  metadata.vRamFootprintBytes = Number.MAX_SAFE_INTEGER;
  assert.equal((await hub.api.localFoundryModels("cpu")).hidden, 1);
  metadata.vRamFootprintBytes = 456;
  assert.equal((await hub.api.localFoundryModels("cpu")).models[0].memory, 456);
  delete metadata.vRamFootprintBytes;
  assert.equal((await hub.api.localFoundryModels("cpu")).unchecked, 1);
  metadata.vRamFootprintBytes = -1;
  await assert.rejects(hub.api.listFoundryModels("cpu"), /metadata/);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(inspectFoundryMemory(true, controller.signal), /abort/i);
});

test("Foundry Local lists public variants directly and detects pagination and HTTP failures", async (t) => {
  const hub = await foundryHub(t);
  const models = await hub.api.listFoundryModels("cpu");
  assert.equal(models.length, 1);
  assert.equal(models[0].id, hub.asset);
  assert.equal(models[0].provider, "CPUExecutionProvider");
  assert.equal(models[0].license, "MIT");
  assert.equal(models[0].memory, 456);
  const body = JSON.parse(hub.requests[0].body);
  assert.ok(
    body.filters.some(({ values }) =>
      values.includes("Foundry Local on Devices"),
    ),
  );
  assert.ok(
    body.filters.some(
      ({ field, values }) => field.endsWith("/device") && values[0] === "cpu",
    ),
  );
  await assert.rejects(
    hub.api.listFoundryModels("invalid"),
    /supported.*device/,
  );
  hub.state.catalogPages = true;
  assert.equal((await hub.api.listFoundryModels("cpu")).length, 2);
  hub.state.catalogPages = false;
  hub.state.catalogToken = true;
  await assert.rejects(hub.api.listFoundryModels("gpu"), /continuation token/);
  hub.state.catalogToken = false;
  hub.state.catalogStatus = 503;
  await assert.rejects(hub.api.listFoundryModels(""), /HTTP 503/);
});

test("Foundry Local downloads all package assets, opens a real native graph and reuses the complete cache", async (t) => {
  const hub = await foundryHub(t);
  hub.state.listingPages = true;
  const progress = [];
  const directory = await hub.api.downloadFoundryModel(
    hub.asset,
    hub.root,
    undefined,
    (value) => progress.push(value),
  );
  for (const [file, bytes] of hub.files)
    assert.deepEqual(await readFile(join(directory, file)), bytes);
  const native = await loadModelFile(
    join(directory, "v1/model.onnx"),
    "model.onnx",
  );
  assert.equal(native.graph().nodes[0].opType, "Relu");
  assert.equal(progress.at(-1).received, progress.at(-1).total);
  const count = hub.requests.length;
  assert.equal(
    await hub.api.downloadFoundryModel(hub.asset, hub.root),
    directory,
  );
  assert.equal(hub.requests.length, count);
  const manifest = await readFile(
    join(directory, ".viewer-download.json"),
    "utf8",
  );
  assert.equal(manifest.includes("fixture-signature"), false);
  assert.equal(hub.requests[0].query.get("assetId"), hub.asset);
  await writeFile(join(directory, "v1/model.onnx.data"), "truncated");
  await assert.rejects(
    hub.api.downloadFoundryModel(hub.asset, hub.root),
    /Incomplete.*cache/,
  );
});

test("Foundry Local rejects unsafe paths, untrusted blob hosts and repeated blob pages before downloading", async (t) => {
  const hub = await foundryHub(t);
  await assert.rejects(
    hub.api.downloadFoundryModel("https://example.com/model", hub.root),
    /model ID/,
  );
  hub.state.blobHost = "example.com";
  await assert.rejects(
    hub.api.downloadFoundryModel(hub.asset, hub.root),
    /blob host/,
  );
  hub.state.blobHost = "fixtures.blob.core.windows.net";
  for (const name of [
    "../outside",
    "v1/../../outside",
    "C:/file",
    "v1\\file",
    "v1/CON",
    "v1/file.",
  ]) {
    hub.files.set(name, Buffer.from("unsafe"));
    await assert.rejects(
      hub.api.downloadFoundryModel(hub.asset, hub.root),
      /Unsafe.*path/,
    );
    hub.files.delete(name);
  }
  hub.state.listingMarker = true;
  await assert.rejects(
    hub.api.downloadFoundryModel(hub.asset, hub.root),
    /duplicate|continuation/,
  );
  assert.deepEqual(await readdir(hub.root), []);
  assert.equal(
    hub.requests.some(
      ({ path, query }) => path.startsWith("/package/") && !query.has("comp"),
    ),
    false,
  );
});

test(
  "Foundry Local concurrent downloads reuse the winning complete package",
  { timeout: 10000 },
  async (t) => {
    const hub = await foundryHub(t);
    let release;
    const ready = new Promise((resolve) => {
      release = resolve;
    });
    let arrivals = 0;
    hub.state.beforeTransfer = async (name) => {
      if (name !== "v1/model.onnx") return;
      if (++arrivals === 2) release();
      await ready;
    };
    const directories = await Promise.all([
      hub.api.downloadFoundryModel(hub.asset, hub.root),
      hub.api.downloadFoundryModel(hub.asset, hub.root),
    ]);
    assert.equal(arrivals, 2);
    assert.equal(directories[0], directories[1]);
    assert.deepEqual(await readdir(join(hub.root, "foundry")), [
      basename(directories[0]),
    ]);
    for (const [file, bytes] of hub.files)
      assert.deepEqual(await readFile(join(directories[0], file)), bytes);
    const count = hub.requests.length;
    assert.equal(
      await hub.api.downloadFoundryModel(hub.asset, hub.root),
      directories[0],
    );
    assert.equal(hub.requests.length, count);
  },
);

test(
  "Foundry Local publication collisions reject an invalid winning cache",
  { timeout: 10000 },
  async (t) => {
    const hub = await foundryHub(t);
    let arrived, release;
    const started = new Promise((resolve) => {
      arrived = resolve;
    });
    const ready = new Promise((resolve) => {
      release = resolve;
    });
    hub.state.beforeTransfer = async () => {
      arrived();
      await ready;
    };
    const download = hub.api.downloadFoundryModel(hub.asset, hub.root);
    const directory = join(
      hub.root,
      "foundry",
      createHash("sha256").update(hub.asset).digest("hex"),
    );
    try {
      await started;
      await mkdir(directory);
      await writeFile(
        join(directory, ".viewer-download.json"),
        JSON.stringify({ id: "another-model", files: [] }),
      );
    } finally {
      release();
    }
    await assert.rejects(download, /Invalid Foundry Local cache manifest/);
    assert.deepEqual(await readdir(join(hub.root, "foundry")), [
      basename(directory),
    ]);
  },
);

test("Foundry Local never publishes failed, truncated or cancelled downloads", async (t) => {
  const hub = await foundryHub(t);
  hub.state.transferStatus = 403;
  await assert.rejects(
    hub.api.downloadFoundryModel(hub.asset, hub.root),
    /HTTP 403/,
  );
  assert.deepEqual(await readdir(join(hub.root, "foundry")), []);
  hub.state.transferStatus = 200;
  hub.state.transferSuffix = "extra";
  await assert.rejects(
    hub.api.downloadFoundryModel(hub.asset, hub.root),
    /Incomplete.*download/,
  );
  assert.deepEqual(await readdir(join(hub.root, "foundry")), []);
  hub.state.transferSuffix = "";
  const controller = new AbortController();
  await assert.rejects(
    hub.api.downloadFoundryModel(hub.asset, hub.root, controller.signal, () =>
      controller.abort(),
    ),
    /abort/i,
  );
  assert.deepEqual(await readdir(join(hub.root, "foundry")), []);
  const directory = await hub.api.downloadFoundryModel(hub.asset, hub.root);
  assert.deepEqual(
    await readFile(join(directory, "v1/model.onnx")),
    hub.files.get("v1/model.onnx"),
  );
});
