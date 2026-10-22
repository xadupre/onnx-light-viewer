const assert = require("node:assert/strict");
const { once } = require("node:events");
const { mkdtemp, readFile, readdir, rm } = require("node:fs/promises");
const { createServer } = require("node:http");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const { test } = require("node:test");
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
      { rfilename: "onnx/model_q8.onnx" },
      { rfilename: "onnx/model.onnx" },
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
  assert.deepEqual(server.requests, ["/api/models/owner/repo"]);
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
