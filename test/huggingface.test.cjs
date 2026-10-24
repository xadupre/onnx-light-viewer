const assert = require("node:assert/strict");
const { once } = require("node:events");
const { mkdtemp, readFile, readdir, rm } = require("node:fs/promises");
const { createServer } = require("node:http");
const { createRequire } = require("node:module");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const { test } = require("node:test");
const { readFileSync } = require("node:fs");
const vm = require("node:vm");
const { huggingFaceTasks } = require("../dist/huggingFaceTasks");
const { loadModel } = require("../dist/graph");
const {
  cachedHuggingFaceModel,
  parseHuggingFaceUrl,
  parseHuggingFaceInput,
  listHuggingFaceModels,
} = require("../dist/huggingface");

async function fixture(t, respond) {
  const root = await mkdtemp(join(tmpdir(), "onnx-viewer-huggingface-"));
  const requests = [];
  const server = createServer((request, response) => {
    requests.push(request.url);
    respond(request, response);
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const stop = async () => {
    if (!server.listening) return;
    server.closeAllConnections();
    await new Promise((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
  };
  t.after(async () => {
    await stop();
    await rm(root, { recursive: true });
  });
  return {
    root,
    requests,
    stop,
    source: (path) => ({
      url: `http://127.0.0.1:${server.address().port}${path}`,
      fileName: "remote.onnx",
    }),
  };
}

async function load(path) {
  return loadModel(await readFile(path), "remote.onnx").graph();
}

test("searches ONNX repositories by task, ranks by downloads and validates public model metadata", async (t) => {
  let payload = [
    {
      id: "author/popular",
      downloads: 1200,
      pipeline_tag: "feature-extraction",
      private: false,
      gated: false,
    },
    { id: "author/gated", downloads: 900, gated: "manual" },
    { id: "author/private", downloads: 800, private: true },
    { id: "author/unspecified", downloads: 2, pipeline_tag: null },
  ];
  let status = 200;
  const server = await fixture(t, (_request, response) => {
    response.writeHead(status, { "Content-Type": "application/json" });
    response.end(JSON.stringify(payload));
  });
  const module = { exports: {} };
  const execute = vm.runInThisContext(
    `(function(require, module, exports, fetch) { ${readFileSync(require.resolve("../dist/huggingface"), "utf8")}\n})`,
  );
  const requireHuggingFace = createRequire(
    require.resolve("../dist/huggingface"),
  );
  execute(requireHuggingFace, module, module.exports, (url, options) => {
    const remote = new URL(url);
    const local = new URL(server.source(remote.pathname).url);
    local.search = remote.search;
    return fetch(local, options);
  });
  const search = module.exports.searchHuggingFaceModels;
  const models = await search(" small bert ", "feature-extraction");
  assert.deepEqual(models, [
    {
      id: "author/popular",
      url: "https://huggingface.co/author/popular",
      downloads: 1200,
      task: "feature-extraction",
    },
    {
      id: "author/unspecified",
      url: "https://huggingface.co/author/unspecified",
      downloads: 2,
    },
  ]);
  const request = new URL(server.requests[0], "http://localhost");
  assert.equal(request.pathname, "/api/models");
  assert.equal(request.searchParams.get("filter"), "onnx");
  assert.equal(request.searchParams.get("search"), "small bert");
  assert.equal(request.searchParams.get("pipeline_tag"), "feature-extraction");
  assert.equal(request.searchParams.get("sort"), "downloads");
  assert.equal(request.searchParams.get("direction"), "-1");
  assert.equal(request.searchParams.get("limit"), "12");
  await search("");
  const popular = new URL(server.requests.at(-1), "http://localhost");
  assert.equal(popular.searchParams.has("search"), false);
  assert.equal(popular.searchParams.has("pipeline_tag"), false);
  for (const { value } of huggingFaceTasks) {
    await search("", value);
    const request = new URL(server.requests.at(-1), "http://localhost");
    assert.equal(request.searchParams.get("pipeline_tag"), value || null);
  }
  const count = server.requests.length;
  await assert.rejects(search("", "invalid-task"), /supported.*task/);
  assert.equal(server.requests.length, count);
  payload = {};
  await assert.rejects(search(""), /Invalid.*search response/);
  payload = [{ id: "../outside", downloads: 0 }];
  await assert.rejects(search(""), /Invalid.*search result/);
  status = 429;
  await assert.rejects(search(""), /HTTP 429/);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(search("", "", controller.signal), /abort/i);
});

test("normalizes Hugging Face file links and rejects non-model or credential-bearing URLs", () => {
  const source = parseHuggingFaceUrl(
    " https://huggingface.co/owner/repo/blob/main/onnx/model%20q4.onnx?download=true#file ",
  );
  assert.deepEqual(source, {
    url: "https://huggingface.co/owner/repo/resolve/main/onnx/model%20q4.onnx",
    fileName: "model q4.onnx",
  });
  assert.deepEqual(parseHuggingFaceUrl(source.url), source);
  for (const input of [
    "https://example.com/owner/repo/resolve/main/model.onnx",
    "http://huggingface.co/owner/repo/resolve/main/model.onnx",
    "https://huggingface.co/owner/repo",
    "https://huggingface.co/owner/repo/blob/main/weights.bin",
    "https://user:password@huggingface.co/owner/repo/blob/main/model.onnx",
    "https://huggingface.co/owner/repo/blob/main/model.onnx?token=not-a-real-token",
  ])
    assert.throws(() => parseHuggingFaceUrl(input));
});

test("accepts repository IDs and URLs without changing direct file links", () => {
  const repository = "onnx-community/Kokoro-82M-v1.0-ONNX";
  for (const input of [
    repository,
    `https://huggingface.co/${repository}`,
    ` https://huggingface.co/${repository}/ `,
  ])
    assert.deepEqual(parseHuggingFaceInput(input), {
      url: `https://huggingface.co/${repository}`,
    });
  const file = `https://huggingface.co/${repository}/blob/main/onnx/model.onnx`;
  assert.deepEqual(parseHuggingFaceInput(file), parseHuggingFaceUrl(file));
  for (const input of [
    "https://example.com/owner/repo",
    "http://huggingface.co/owner/repo",
    "owner",
    "owner/repo/extra",
    "https://huggingface.co/owner/repo?token=invalid",
  ])
    assert.throws(() => parseHuggingFaceInput(input));
});

test("lists nested ONNX variants through HTTP and reports empty, malformed or inaccessible repositories", async (t) => {
  let payload = {
    siblings: [
      { rfilename: "README.md" },
      { rfilename: "onnx/model_q8.onnx", lfs: { size: 1024 } },
      { rfilename: "onnx/model.onnx", size: 4096 },
      { rfilename: "onnx/model.onnx_data" },
    ],
  };
  let status = 200;
  const server = await fixture(t, (_request, response) => {
    response.writeHead(status, { "Content-Type": "application/json" });
    response.end(JSON.stringify(payload));
  });
  const repository = { url: server.source("/owner/repo").url };
  const files = await listHuggingFaceModels(repository);
  assert.deepEqual(
    files.map(({ path }) => path),
    ["onnx/model.onnx", "onnx/model_q8.onnx"],
  );
  assert.equal(
    files[1].url,
    `${repository.url}/resolve/main/onnx/model_q8.onnx`,
  );
  assert.equal(files[0].size, 4096);
  assert.equal(files[1].size, 1024);
  const metadataRequest = new URL(server.requests[0], "http://localhost");
  assert.equal(metadataRequest.pathname, "/api/models/owner/repo");
  assert.equal(metadataRequest.searchParams.get("blobs"), "true");
  assert.equal(metadataRequest.searchParams.get("expand[]"), "siblings");
  payload = {
    siblings: [
      { rfilename: "unknown.onnx" },
      { rfilename: "empty.onnx", size: 0 },
    ],
  };
  const unknown = await listHuggingFaceModels(repository);
  assert.equal(
    unknown.find(({ path }) => path === "unknown.onnx").size,
    undefined,
  );
  assert.equal(unknown.find(({ path }) => path === "empty.onnx").size, 0);
  payload = { siblings: [{ rfilename: "invalid.onnx", size: -1 }] };
  await assert.rejects(listHuggingFaceModels(repository), /Invalid.*size/);
  payload = { siblings: [{ rfilename: "README.md" }] };
  await assert.rejects(listHuggingFaceModels(repository), /No ONNX files/);
  payload = { siblings: [{ rfilename: "../model.onnx" }] };
  await assert.rejects(listHuggingFaceModels(repository), /Invalid.*path/);
  payload = {};
  await assert.rejects(listHuggingFaceModels(repository), /Invalid.*listing/);
  status = 403;
  await assert.rejects(listHuggingFaceModels(repository), /HTTP 403/);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    listHuggingFaceModels(repository, controller.signal),
    /abort/i,
  );
});

test("follows download redirects, separates source URLs and reuses validated models offline", async (t) => {
  const bytes = await readFile(join(__dirname, "models/relu.onnx"));
  const server = await fixture(t, (request, response) => {
    if (request.url === "/revision-a/model.onnx") {
      response.writeHead(302, { Location: "/payload" });
      response.end();
    } else response.end(bytes);
  });
  const first = server.source("/revision-a/model.onnx");
  const second = server.source("/revision-b/model.onnx");
  const downloaded = await cachedHuggingFaceModel(first, server.root, load);
  assert.equal(downloaded.cached, false);
  assert.equal(downloaded.value.nodes[0].opType, "Relu");
  assert.deepEqual(await readFile(downloaded.filePath), bytes);
  const other = await cachedHuggingFaceModel(second, server.root, load);
  assert.notEqual(other.filePath, downloaded.filePath);
  assert.deepEqual(server.requests, [
    "/revision-a/model.onnx",
    "/payload",
    "/revision-b/model.onnx",
  ]);
  await server.stop();
  const cached = await cachedHuggingFaceModel(first, server.root, load);
  assert.equal(cached.cached, true);
  assert.deepEqual(cached.value, downloaded.value);
  assert.equal(server.requests.length, 3);
});

test("does not cache HTTP failures or unparseable models and allows a later retry", async (t) => {
  const bytes = await readFile(join(__dirname, "models/relu.onnx"));
  let valid = false;
  const server = await fixture(t, (request, response) => {
    if (request.url === "/private/model.onnx") {
      response.writeHead(403);
      response.end("Access denied");
    } else response.end(valid ? bytes : "<html>Not an ONNX model</html>");
  });

  await assert.rejects(
    cachedHuggingFaceModel(
      server.source("/private/model.onnx"),
      server.root,
      load,
    ),
    /HTTP 403/,
  );
  const source = server.source("/public/model.onnx");
  await assert.rejects(cachedHuggingFaceModel(source, server.root, load));
  assert.ok(
    !(await readdir(server.root, { recursive: true })).some(
      (path) => path.endsWith(".partial") || path.endsWith(".onnx"),
    ),
  );
  valid = true;
  const downloaded = await cachedHuggingFaceModel(source, server.root, load);
  assert.equal(downloaded.cached, false);
  assert.equal(downloaded.value.nodes[0].opType, "Relu");
});

test("reports streamed bytes with and without a known total and skips download progress for cache hits", async (t) => {
  const bytes = await readFile(join(__dirname, "models/relu.onnx"));
  const server = await fixture(t, (request, response) => {
    if (request.url === "/known.onnx")
      response.setHeader("Content-Length", bytes.length);
    response.write(bytes.subarray(0, 8));
    setImmediate(() => response.end(bytes.subarray(8)));
  });
  for (const path of ["/known.onnx", "/chunked.onnx"]) {
    const updates = [];
    const source = server.source(path);
    await cachedHuggingFaceModel(
      source,
      server.root,
      load,
      undefined,
      (progress) => updates.push(progress),
    );
    assert.equal(updates[0].received, 0);
    assert.equal(updates.at(-1).received, bytes.length);
    assert.equal(
      updates.at(-1).total,
      path === "/known.onnx" ? bytes.length : undefined,
    );
    assert.ok(
      updates.every(
        (update, index) =>
          index === 0 || update.received >= updates[index - 1].received,
      ),
    );
    const count = updates.length;
    await cachedHuggingFaceModel(
      source,
      server.root,
      load,
      undefined,
      (progress) => updates.push(progress),
    );
    assert.equal(updates.length, count);
  }
});

test("cancels an unfinished HTTP transfer without retaining a partial model", async (t) => {
  let started;
  const receiving = new Promise((resolve) => {
    started = resolve;
  });
  const server = await fixture(t, (_request, response) => {
    response.writeHead(200);
    response.write(Buffer.from([8]));
    started();
  });
  const controller = new AbortController();
  const failed = assert.rejects(
    cachedHuggingFaceModel(
      server.source("/slow/model.onnx"),
      server.root,
      load,
      controller.signal,
    ),
    /abort/i,
  );
  await receiving;
  controller.abort();
  await failed;
  assert.ok(
    !(await readdir(server.root, { recursive: true })).some(
      (path) => path.endsWith(".partial") || path.endsWith(".onnx"),
    ),
  );
});
