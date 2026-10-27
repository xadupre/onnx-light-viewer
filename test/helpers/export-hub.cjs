const { createServer } = require("node:http");
const { createHash } = require("node:crypto");

async function exportHub(t, onRequest = () => {}) {
  const requests = [];
  const revision = "1234567890abcdef1234567890abcdef12345678";
  const files = (model) => ({
    "config.json": Buffer.from(JSON.stringify({ modelId: model })),
    ...(model === "owner/custom"
      ? { "modeling_custom.py": Buffer.from("# Custom model code\n") }
      : {}),
    ...(model === "owner/config-only"
      ? {}
      : model === "owner/sharded"
        ? {
            "model.safetensors.index.json": Buffer.from(
              JSON.stringify({
                weight_map: {
                  a: "model-00001.safetensors",
                  b: "model-00002.safetensors",
                },
              }),
            ),
            "model-00001.safetensors": Buffer.from("first shard"),
            "model-00002.safetensors": Buffer.from("second shard"),
          }
        : { "model.safetensors": Buffer.from("source weights") }),
  });
  const server = createServer((request, response) => {
    const pathname = decodeURIComponent(
      new URL(request.url, "http://hub").pathname,
    );
    requests.push({ method: request.method, pathname });
    onRequest(request.method, pathname);
    if (pathname.startsWith("/api/models/")) {
      const model = pathname
        .slice("/api/models/".length)
        .replace(/\/(?:revision|tree)\/[^/]+$/, "");
      if (model === "download-fail") {
        response.writeHead(404, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ error: "Repository not found" }));
        return;
      }
      response.setHeader("Content-Type", "application/json");
      if (pathname.includes("/tree/")) {
        response.end(
          JSON.stringify(
            Object.entries(files(model)).map(([path, data]) => ({
              type: "file",
              path,
              size: data.length,
              oid: createHash("sha1").update(data).digest("hex"),
            })),
          ),
        );
        return;
      }
      response.end(
        JSON.stringify({
          id: model,
          sha: revision,
          siblings: [
            ...Object.keys(files(model)),
            ...(model === "owner/config-only"
              ? []
              : ["pytorch_model.bin", "tf_model.h5", "model.fp16.safetensors"]),
          ].map((rfilename) => ({ rfilename })),
        }),
      );
      return;
    }
    const match = /^\/(.+)\/resolve\/([^/]+)\/(.+)$/.exec(pathname);
    const data = match && files(match[1])[match[3]];
    if (!data) {
      response.writeHead(404);
      response.end();
      return;
    }
    response.writeHead(200, {
      "Content-Length": data.length,
      ETag: `"${createHash("sha256").update(data).digest("hex")}"`,
      "X-Repo-Commit": revision,
    });
    response.end(request.method === "HEAD" ? undefined : data);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(
    () =>
      new Promise((resolve) => {
        server.close(resolve);
        server.closeAllConnections();
      }),
  );
  return {
    endpoint: `http://127.0.0.1:${server.address().port}`,
    requests,
    revision,
  };
}

module.exports = { exportHub };
