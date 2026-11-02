const { readFileSync, writeFileSync, writeSync } = require("node:fs");
const { join } = require("node:path");
const { createInterface } = require("node:readline");
const { serialize } = require("node:v8");
const { loadModel } = require("../dist/graph");

async function main(directory, retainedModel) {
  const { inputs, options } = JSON.parse(
    readFileSync(join(directory, "request.json"), "utf8"),
  );
  const model =
    retainedModel ??
    loadModel(readFileSync(join(directory, "model.onnx")), "model.onnx");
  const sources = inputs.map((input, index) =>
    typeof input !== "string" && input.source === "protobuf"
      ? {
          ...input,
          data: readFileSync(join(directory, `input-${index}.pb`)),
        }
      : input,
  );
  const { log, ...result } = await model.run(sources, [], options);
  if (log) writeSync(1, log);
  writeFileSync(join(directory, "result.bin"), serialize(result), {
    flag: "wx",
    mode: 0o600,
  });
  return options.keepLoaded ? model : undefined;
}

async function worker() {
  const input = createInterface({ input: process.stdin });
  let model;
  for await (const line of input) {
    try {
      const request = JSON.parse(line);
      model = await main(request.input, model);
      writeSync(1, `\x1e${JSON.stringify({ type: "done" })}\n`);
    } catch (error) {
      model?.releaseRunSession();
      model = undefined;
      writeSync(
        1,
        `\x1e${JSON.stringify({ type: "error", error: error.stack ?? String(error) })}\n`,
      );
    }
  }
  model?.releaseRunSession();
}

(process.argv[2] === "--worker" ? worker() : main(process.argv[2])).catch(
  (error) => {
    console.error(error.stack ?? String(error));
    process.exitCode = 1;
  },
);
