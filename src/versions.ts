import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

export interface ModelVersions {
  opset?: number;
  irVersion?: number;
}

export function validateModelVersions(value: {
  opset?: unknown;
  irVersion?: unknown;
}): asserts value is ModelVersions {
  for (const [name, version] of Object.entries({
    opset: value.opset,
    ir_version: value.irVersion,
  })) {
    if (
      version !== undefined &&
      (typeof version !== "number" ||
        !Number.isSafeInteger(version) ||
        version < 1)
    )
      throw new Error(`${name} must be a positive safe integer.`);
  }
  if (value.opset === undefined && value.irVersion === undefined)
    throw new Error("Specify an opset or ir_version.");
}

const script = `
import json
import sys
import onnx
from onnx.version_converter import convert_version

source, destination, status, options = sys.argv[1:]
options = json.loads(options)
model = onnx.load(source)
original = model.SerializeToString(deterministic=True)
standard = onnx.ModelProto()
standard.CopyFrom(model)
standard.DiscardUnknownFields()
if standard.SerializeToString(deterministic=True) != original:
    raise ValueError("Version conversion cannot preserve unsupported ONNX protobuf extensions.")
onnx.checker.check_model(model)
original_ir = model.ir_version
if "opset" in options:
    model = convert_version(model, options["opset"])
model.ir_version = options.get("irVersion", original_ir)
onnx.checker.check_model(model)
onnx.save_model(model, destination, save_as_external_data=False)
with open(status, "w", encoding="utf-8") as stream:
    json.dump({
        "changed": model.SerializeToString(deterministic=True) != original,
        "opset": next((item.version for item in model.opset_import if item.domain in ("", "ai.onnx")), None),
        "irVersion": model.ir_version,
    }, stream)
`;

/** Converts the current embedded snapshot using optional, explicitly configured Python tooling. */
export async function changeModelVersions(
  bytes: Uint8Array,
  versions: ModelVersions,
  pythonPath: string,
  storageDirectory: string,
): Promise<{
  bytes: Uint8Array;
  changed: boolean;
  opset: number | null;
  irVersion: number;
}> {
  validateModelVersions(versions);
  await mkdir(storageDirectory, { recursive: true });
  const directory = await mkdtemp(join(resolve(storageDirectory), "versions-"));
  try {
    const source = join(directory, "input.onnx");
    const destination = join(directory, "output.onnx");
    const status = join(directory, "status.json");
    await writeFile(source, bytes, { mode: 0o600 });
    try {
      await promisify(execFile)(
        pythonPath,
        [
          "-P",
          "-c",
          script,
          source,
          destination,
          status,
          JSON.stringify(versions),
        ],
        { cwd: directory, timeout: 300000, maxBuffer: 1024 * 1024 },
      );
    } catch (error) {
      throw new Error(
        `ONNX version conversion failed. Requires onnx in the configured Python environment. ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    const result = JSON.parse(await readFile(status, "utf8"));
    if (
      typeof result.changed !== "boolean" ||
      (result.opset !== null &&
        (!Number.isSafeInteger(result.opset) || result.opset < 1)) ||
      !Number.isSafeInteger(result.irVersion) ||
      result.irVersion < 1
    )
      throw new Error("ONNX version conversion returned an invalid status.");
    return {
      bytes: await readFile(destination),
      changed: result.changed,
      opset: result.opset,
      irVersion: result.irVersion,
    };
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
