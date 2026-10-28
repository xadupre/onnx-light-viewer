import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

const script = `
import json
import logging
import sys
import onnx
from onnxscript import optimizer, rewriter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

def log_model(stage, model):
    nodes = model.graph.node
    identities = sum(node.op_type == "Identity" and node.domain in ("", "ai.onnx") for node in nodes)
    logging.info("%s: %d main-graph nodes, %d Identity nodes", stage, len(nodes), identities)

source, destination, status = sys.argv[1:]
model = onnx.load(source)
original = model.SerializeToString(deterministic=True)
standard = onnx.ModelProto()
standard.CopyFrom(model)
standard.DiscardUnknownFields()
if standard.SerializeToString(deterministic=True) != original:
    raise ValueError("ONNX Script cannot optimize unsupported ONNX protobuf extensions.")
onnx.checker.check_model(model)
log_model("Input", model)
model = optimizer.optimize(model)
log_model("After optimization", model)
model = rewriter.rewrite(model)
log_model("After fusion", model)
onnx.save_model(model, destination, save_as_external_data=False)
with open(status, "w", encoding="utf-8") as stream:
    json.dump(model.SerializeToString(deterministic=True) != original, stream)
`;

/** Runs optional Python tooling against an embedded snapshot, never the source file. */
export async function optimizeWithOnnxScript(
  bytes: Uint8Array,
  pythonPath: string,
  storageDirectory: string,
): Promise<{ bytes: Uint8Array; changed: boolean; logs: string }> {
  await mkdir(storageDirectory, { recursive: true });
  const directory = await mkdtemp(
    join(resolve(storageDirectory), "onnxscript-"),
  );
  try {
    const source = join(directory, "input.onnx");
    const destination = join(directory, "output.onnx");
    const status = join(directory, "changed.json");
    await writeFile(source, bytes, { mode: 0o600 });
    let logs: string;
    try {
      const { stdout, stderr } = await promisify(execFile)(
        pythonPath,
        ["-P", "-c", script, source, destination, status],
        { cwd: directory, timeout: 300000, maxBuffer: 1024 * 1024 },
      );
      logs = [stdout, stderr].filter(Boolean).join("\n");
    } catch (error) {
      throw new Error(
        `ONNX Script optimization failed. Install onnx and onnxscript in the configured Python environment. ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    let result: Uint8Array;
    try {
      result = await readFile(destination);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException | null | undefined)?.code;
      if (code === "ENOENT") {
        throw new Error(
          `ONNX Script optimization succeeded but did not produce the expected artifact: ${destination}. ` +
            `Check the Python environment, filesystem permissions, and Python output/logs.`,
        );
      }
      throw new Error(
        `ONNX Script optimization produced an unreadable output model at ${destination}. ${error instanceof Error ? error.message : String(error)}`,
      );
    }

    let changedText: string;
    try {
      changedText = await readFile(status, "utf8");
    } catch (error) {
      const code = (error as NodeJS.ErrnoException | null | undefined)?.code;
      if (code === "ENOENT") {
        throw new Error(
          `ONNX Script optimization succeeded but did not produce the expected artifact: ${status}. ` +
            `Check the Python environment, filesystem permissions, and Python output/logs.`,
        );
      }
      throw new Error(
        `ONNX Script optimization produced an unreadable change status file at ${status}. ${error instanceof Error ? error.message : String(error)}`,
      );
    }

    const changed: unknown = JSON.parse(changedText);
    if (typeof changed !== "boolean")
      throw new Error("ONNX Script returned an invalid change status.");
    return { bytes: result, changed, logs };
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
