#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROOT"

for tool in node npm code; do
    if ! command -v "$tool" > /dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$tool" >&2
        exit 1
    fi
done

if [[ ! -d node_modules ]]; then
    npm ci
fi
npm run compile

node <<'NODE'
const { writeFileSync } = require("node:fs");
const { onnx } = require("onnx-proto");
const { demoGraph } = require("./dist/graph");

const valueInfo = (name) => ({
    name,
    type: {
        tensorType: {
            elemType: onnx.TensorProto.DataType.FLOAT,
            shape: { dim: [{ dimParam: "N" }] },
        },
    },
});
const model = onnx.ModelProto.create({
    irVersion: 9,
    producerName: "onnx-light-viewer",
    opsetImport: [{ domain: "", version: 13 }],
    graph: {
        name: demoGraph.name,
        input: demoGraph.inputs.map(valueInfo),
        output: demoGraph.outputs.map(valueInfo),
        node: demoGraph.nodes.map((node) => ({
            name: node.name,
            opType: node.opType,
            domain: node.domain === "ai.onnx" ? "" : node.domain,
            input: node.inputs,
            output: node.outputs,
        })),
    },
});
writeFileSync("dist/demo.onnx", onnx.ModelProto.encode(model).finish());
NODE

exec code --new-window --disable-extensions --extensionDevelopmentPath="$ROOT" \
    "$ROOT" "$ROOT/dist/demo.onnx"
