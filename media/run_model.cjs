const { readFileSync, writeFileSync } = require("node:fs");
const { join } = require("node:path");
const { serialize } = require("node:v8");
const { loadModel } = require("../dist/graph");

async function main(directory) {
  const { inputs, options } = JSON.parse(
    readFileSync(join(directory, "request.json"), "utf8"),
  );
  const model = loadModel(
    readFileSync(join(directory, "model.onnx")),
    "model.onnx",
  );
  const sources = inputs.map((input, index) =>
    typeof input !== "string" && input.source === "protobuf"
      ? {
          ...input,
          data: readFileSync(join(directory, `input-${index}.pb`)),
        }
      : input,
  );
  const { log, ...result } = await model.run(sources, [], options);
  if (log) process.stdout.write(log);
  writeFileSync(join(directory, "result.bin"), serialize(result), {
    flag: "wx",
    mode: 0o600,
  });
}

main(process.argv[2]).catch((error) => {
  console.error(error.stack ?? String(error));
  process.exitCode = 1;
});
