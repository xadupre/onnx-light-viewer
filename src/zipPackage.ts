import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const native: {
  extractModelPackage(bytes: Uint8Array, directory: string): Promise<string[]>;
  createModelPackage(
    modelName: string,
    model: Uint8Array,
    weights: { name: string; bytes: Uint8Array }[],
  ): Promise<Uint8Array>;
} = require("./native/onnx_light_viewer.node");

export interface ModelPackage {
  directory: string;
  models: string[];
  dispose(): Promise<void>;
}

export async function extractModelPackage(
  bytes: Uint8Array,
): Promise<ModelPackage> {
  // Snapshot before yielding to the directory creation.
  bytes = bytes.slice();
  const directory = await mkdtemp(
    join(resolve(tmpdir()), "onnx-light-viewer-package-"),
  );
  const dispose = () => rm(directory, { recursive: true, force: true });
  try {
    const models = await native.extractModelPackage(bytes, directory);
    return { directory, models, dispose };
  } catch (error) {
    await dispose();
    throw error;
  }
}

export async function createModelPackage(
  modelName: string,
  model: Uint8Array,
  weights: { name: string; bytes: Uint8Array }[],
): Promise<Uint8Array> {
  return native.createModelPackage(modelName, model, weights);
}
