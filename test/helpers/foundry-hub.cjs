const { once } = require("node:events");
const { readFileSync } = require("node:fs");
const { mkdtemp, rm } = require("node:fs/promises");
const { createServer } = require("node:http");
const { createRequire } = require("node:module");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const vm = require("node:vm");

const asset = "azureml://registries/azureml/models/viewer-fixture/versions/1";

async function foundryHub(t) {
  const root = await mkdtemp(join(tmpdir(), "viewer-foundry-"));
  const files = new Map([
    ["v1/model.onnx", readFileSync(require.resolve("../models/relu.onnx"))],
    ["v1/model.onnx.data", Buffer.from("external weights")],
    ["v1/tokenizer.json", Buffer.from('{"version":"1.0"}')],
    ["v1/special & tokens.json", Buffer.from("{}")],
  ]);
  const requests = [];
  const state = {
    files,
    catalogStatus: 200,
    catalogToken: false,
    catalogPages: false,
    listingMarker: false,
    listingPages: false,
    blobHost: "fixtures.blob.core.windows.net",
    transferStatus: 200,
    transferSuffix: "",
  };
  const catalogEntry = {
    assetId: asset,
    annotations: {
      systemCatalogData: { license: "MIT" },
    },
    properties: {
      name: "viewer-fixture",
      version: 1,
      variantInfo: {
        variantMetadata: {
          modelType: "ONNX",
          device: "cpu",
          executionProvider: "CPUExecutionProvider",
          fileSizeBytes: 123,
          vRamFootprintBytes: 456,
        },
      },
    },
  };
  const escape = (text) =>
    text
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://localhost");
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = Buffer.concat(chunks).toString();
    requests.push({ path: url.pathname, query: url.searchParams, body });
    if (url.pathname === "/asset-gallery/v1.0/models") {
      const next = JSON.parse(body).continuationToken === "page2";
      res.writeHead(state.catalogStatus, {
        "Content-Type": "application/json",
      });
      res.end(
        JSON.stringify({
          value: state.catalogEntries ?? [
            next && state.catalogPages
              ? {
                  ...catalogEntry,
                  assetId: asset.replace(/\/1$/, "/2"),
                  properties: { ...catalogEntry.properties, version: 2 },
                }
              : catalogEntry,
          ],
          ...(state.catalogToken || (state.catalogPages && !next)
            ? { continuationToken: "page2" }
            : {}),
        }),
      );
    } else if (url.pathname.startsWith("/modelregistry/")) {
      res.setHeader("Content-Type", "application/json");
      res.end(
        JSON.stringify({
          blobSasUri: `https://${state.blobHost}/package?sig=fixture-signature`,
        }),
      );
    } else if (url.searchParams.get("comp") === "list") {
      const next = url.searchParams.get("marker") === "page2";
      const entries = [...files];
      const page = state.listingPages
        ? next
          ? entries.slice(2)
          : entries.slice(0, 2)
        : entries;
      res.setHeader("Content-Type", "application/xml");
      res.end(
        `<?xml version="1.0"?><EnumerationResults><Blobs>${page
          .map(
            ([name, data]) =>
              `<Blob><Name>${escape(name)}</Name><Properties><Content-Length>${data.length}</Content-Length></Properties></Blob>`,
          )
          .join(
            "",
          )}</Blobs><NextMarker>${state.listingMarker || (state.listingPages && !next) ? "page2" : ""}</NextMarker></EnumerationResults>`,
      );
    } else {
      const name = decodeURIComponent(url.pathname.slice("/package/".length));
      const file = files.get(name);
      if (!file) {
        res.writeHead(404);
        res.end();
        return;
      }
      res.writeHead(state.transferStatus, {
        "Content-Type": "application/octet-stream",
      });
      res.end(Buffer.concat([file, Buffer.from(state.transferSuffix)]));
    }
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  });
  const localFetch = (remote, options) => {
    const url = new URL(remote);
    const local = new URL(
      `http://127.0.0.1:${server.address().port}${url.pathname}`,
    );
    local.search = url.search;
    return fetch(local, options);
  };
  const filename = require.resolve("../../dist/foundry");
  const module = { exports: {} };
  const execute = vm.runInThisContext(
    `(function(require, module, exports, fetch) { ${readFileSync(filename, "utf8")}\n})`,
    { filename },
  );
  execute(createRequire(filename), module, module.exports, localFetch);
  return {
    root,
    asset,
    files,
    requests,
    state,
    catalogEntry,
    api: module.exports,
  };
}

module.exports = { foundryHub };
