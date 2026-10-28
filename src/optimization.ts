import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

const script = `
import json
import sys
import onnx
from onnxscript import optimizer, rewriter

source, destination, status = sys.argv[1:]
model = onnx.load(source)
original = model.SerializeToString(deterministic=True)
standard = onnx.ModelProto()
standard.CopyFrom(model)
standard.DiscardUnknownFields()
if standard.SerializeToString(deterministic=True) != original:
    raise ValueError("ONNX Script cannot optimize unsupported ONNX protobuf extensions.")
onnx.checker.check_model(model)
model = optimizer.optimize(model)
model = rewriter.rewrite(model)
onnx.save_model(model, destination, save_as_external_data=False)
with open(status, "w", encoding="utf-8") as stream:
    json.dump(model.SerializeToString(deterministic=True) != original, stream)
`;

/** Runs optional Python tooling against an embedded snapshot, never the source file. */
export async function optimizeWithOnnxScript(
  bytes: Uint8Array,
  pythonPath: string,
  storageDirectory: string,
): Promise<{ bytes: Uint8Array; changed: boolean }> {
  await mkdir(storageDirectory, { recursive: true });
  const directory = await mkdtemp(
    join(resolve(storageDirectory), "onnxscript-"),
  );
  try {
    const source = join(directory, "input.onnx");
    const destination = join(directory, "output.onnx");
    const status = join(directory, "changed.json");
    await writeFile(source, bytes, { mode: 0o600 });
    try {
      await promisify(execFile)(
        pythonPath,
        ["-P", "-c", script, source, destination, status],
        { cwd: directory, timeout: 300000, maxBuffer: 1024 * 1024 },
      );
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
    return { bytes: result, changed };
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
