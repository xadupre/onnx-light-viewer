import {
  copyFile,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from "node:path";
import { randomBytes } from "node:crypto";
import type { NativeModel } from "./graph";
import { writePackageArchive } from "./packageArchive";

export type RuntimePackageKind = "generic" | "genai" | "compiled";

export interface RuntimePackage {
  directory: string;
  archivePath?: string;
  models: string[];
  defaultModel?: string;
  genai: boolean;
  manifest?: Record<string, unknown>;
  genaiDirectories?: Record<string, string>;
  genaiConfigurationDirectories?: string[];
  genaiModels?: { model: string; directory: string; shared: boolean }[];
  manifestReferences?: ManifestModelReference[];
}

interface ManifestModelReference {
  document: string;
  pointer: string[];
  variantDirectory: string;
  model: string;
  shared: boolean;
}

function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function packageReference(
  directory: string,
  base: string,
  name: string,
  manifest?: Record<string, unknown>,
): string {
  const asset = /^sha256:([a-f0-9]{64})(?:\/(.*))?$/.exec(name);
  if (asset) {
    const overrides = manifest?.shared_assets;
    const override = object(overrides)
      ? overrides[`sha256:${asset[1]}`]
      : undefined;
    const assetPath = override ?? `shared_assets/sha256-${asset[1]}`;
    if (typeof assetPath !== "string")
      throw new Error("Invalid package shared asset path.");
    const root = packageReference(directory, "", assetPath);
    const tail = asset[2] ?? "";
    if (tail.split("/").includes(".."))
      throw new Error(
        "Shared asset references must remain inside the asset directory.",
      );
    return packageReference(directory, root, tail);
  }
  if (
    isAbsolute(name) ||
    name.includes(":") ||
    name.includes("\\") ||
    name.includes("\0") ||
    name.split("/").includes("..")
  )
    throw new Error(
      "Package references must remain inside the package directory.",
    );
  const path = relative(directory, resolve(directory, base, name));
  if (path === ".." || path.startsWith(`..${sep}`) || isAbsolute(path))
    throw new Error(
      "Package references must remain inside the package directory.",
    );
  return path.split(sep).join("/");
}

async function packageManifest(
  directory: string,
  files: string[],
): Promise<{
  manifest?: Record<string, unknown>;
  references?: ManifestModelReference[];
}> {
  if (!files.includes("manifest.json") || files.includes("genai_config.json"))
    return {};
  const manifest: unknown = JSON.parse(
    await readFile(join(directory, "manifest.json"), "utf8"),
  );
  if (!object(manifest) || !object(manifest.components))
    throw new Error("The runtime package manifest must declare components.");
  if (!/^1(?:\.\d+)?$/.test(String(manifest.schema_version)))
    throw new Error(
      "The viewer supports runtime package manifest schema version 1.",
    );
  const references: ManifestModelReference[] = [];
  for (const [componentName, declaration] of Object.entries(
    manifest.components,
  )) {
    let document = "manifest.json";
    let pointer = ["components", componentName];
    let component = declaration;
    let base = "";
    if (typeof component === "string") {
      const path = packageReference(directory, "", component, manifest);
      document = files.includes(path) ? path : `${path}/component.json`;
      base = dirname(document).split(sep).join("/");
      component = JSON.parse(
        await readFile(memberPath(directory, document), "utf8"),
      );
      pointer = [];
    }
    if (!object(component) || !object(component.variants))
      throw new Error(`Invalid runtime package component: ${componentName}`);
    for (const [variantName, variant] of Object.entries(component.variants)) {
      if (!object(variant))
        throw new Error(`Invalid runtime package variant: ${variantName}`);
      const variantPath = variant.variant_directory ?? variantName;
      if (typeof variantPath !== "string")
        throw new Error(
          `Invalid runtime package variant directory: ${variantName}`,
        );
      const variantDirectory = packageReference(
        directory,
        base,
        variantPath,
        manifest,
      );
      if (!(await lstat(join(directory, variantDirectory))).isDirectory())
        throw new Error(
          `Missing runtime package variant directory: ${variantName}`,
        );
      const executor = object(variant.executor_info)
        ? variant.executor_info.ort
        : undefined;
      if (executor === undefined) continue;
      let info = executor;
      let infoDocument = document;
      let infoPointer = [
        ...pointer,
        "variants",
        variantName,
        "executor_info",
        "ort",
      ];
      if (typeof executor === "string") {
        infoDocument = packageReference(
          directory,
          variantDirectory,
          executor,
          manifest,
        );
        info = JSON.parse(
          await readFile(memberPath(directory, infoDocument), "utf8"),
        );
        infoPointer = [];
      }
      if (!object(info))
        throw new Error(`Invalid ORT executor information: ${variantName}`);
      if (info.model_file === undefined) continue;
      if (typeof info.model_file !== "string")
        throw new Error(`Invalid ORT model reference: ${variantName}`);
      references.push({
        document: infoDocument,
        pointer: [...infoPointer, "model_file"],
        variantDirectory,
        model: packageReference(
          directory,
          variantDirectory,
          info.model_file,
          manifest,
        ),
        shared: info.model_file.startsWith("sha256:"),
      });
    }
  }
  return { manifest, references };
}

function memberPath(directory: string, name: string): string {
  if (
    !name ||
    isAbsolute(name) ||
    /^[a-z]:/i.test(name) ||
    name.includes("\0") ||
    name.includes("\\")
  )
    throw new Error(
      "Package members must use relative paths inside the package directory.",
    );
  const path = resolve(directory, name);
  const member = relative(directory, path);
  if (
    !member ||
    member === ".." ||
    member.startsWith(`..${sep}`) ||
    isAbsolute(member)
  )
    throw new Error(
      "Package members must remain inside the package directory.",
    );
  return path;
}

async function packageFiles(
  directory: string,
  prefix = "",
  visitDirectory?: (name: string) => Promise<void>,
): Promise<string[]> {
  const files: string[] = [];
  for (const entry of await readdir(join(directory, prefix), {
    withFileTypes: true,
  })) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isSymbolicLink())
      throw new Error(`Package symbolic links are not supported: ${name}`);
    if (entry.isDirectory()) {
      await visitDirectory?.(name);
      files.push(...(await packageFiles(directory, name, visitDirectory)));
    } else if (entry.isFile()) files.push(name);
    else throw new Error(`Package assets must be regular files: ${name}`);
  }
  return files.sort();
}

function filenames(value: unknown): string[] {
  if (!value || typeof value !== "object") return [];
  return Object.entries(value).flatMap(([key, child]) =>
    key === "filename" && typeof child === "string"
      ? [child]
      : filenames(child),
  );
}

async function genaiConfiguration(directory: string): Promise<unknown> {
  return JSON.parse(
    await readFile(join(directory, "genai_config.json"), "utf8"),
  );
}

function genaiProvider(config: unknown): string {
  const model = object(config) ? config.model : undefined;
  const decoder = object(model) ? model.decoder : undefined;
  const options = object(decoder) ? decoder.session_options : undefined;
  const providers = object(options) ? options.provider_options : undefined;
  if (
    providers === undefined ||
    (Array.isArray(providers) && !providers.length)
  )
    return "CPUExecutionProvider";
  if (!Array.isArray(providers) || !object(providers[0]))
    throw new Error("Invalid GenAI execution provider configuration.");
  const name = Object.keys(providers[0])[0];
  const names: Record<string, string> = {
    cpu: "CPUExecutionProvider",
    cuda: "CUDAExecutionProvider",
    dml: "DmlExecutionProvider",
    openvino: "OpenVINOExecutionProvider",
    qnn: "QNNExecutionProvider",
    tensorrt: "TensorrtExecutionProvider",
    rocm: "ROCMExecutionProvider",
    webgpu: "WebGpuExecutionProvider",
  };
  if (name?.endsWith("ExecutionProvider")) return name;
  if (names[name]) return names[name];
  throw new Error(`The GenAI provider has no known ORT package name: ${name}`);
}

/** Packages are directories of runtime assets, not another model serialization format. */
export async function inspectRuntimePackage(
  directory: string,
): Promise<RuntimePackage> {
  directory = await realpath(directory);
  if (!(await lstat(directory)).isDirectory())
    throw new Error("Choose a package directory.");
  const files = await packageFiles(directory);
  const { manifest, references: manifestReferences } = await packageManifest(
    directory,
    files,
  );
  const models = files.filter((name) =>
    /\.(?:onnx|ort|pipe-onnx)$/i.test(name),
  );
  if (!models.length)
    throw new Error(
      "The package contains no ONNX, ORT FlatBuffer or pipeline model.",
    );
  for (const reference of manifestReferences ?? [])
    if (!models.includes(reference.model))
      throw new Error(`Runtime package model is missing: ${reference.model}`);
  const genaiDirectories: Record<string, string> = {};
  const genaiModels: NonNullable<RuntimePackage["genaiModels"]> = [];
  const configs = files.filter((name) =>
    /(?:^|\/)genai_config\.json$/.test(name),
  );
  const genai = configs.length > 0;
  let defaultModel = models.length === 1 ? models[0] : undefined;
  if (manifestReferences?.length === 1)
    defaultModel = manifestReferences[0].model;
  for (const configPath of configs) {
    const configDirectory = dirname(configPath).split(sep).join("/");
    const config = await genaiConfiguration(join(directory, configDirectory));
    const configured = filenames(config).map((name) =>
      packageReference(directory, configDirectory, name, manifest),
    );
    for (const name of filenames(config))
      genaiModels.push({
        model: packageReference(directory, configDirectory, name, manifest),
        directory: packageReference(directory, "", configDirectory),
        shared: name.startsWith("sha256:"),
      });
    if (!configured.length)
      throw new Error("genai_config.json does not reference a model filename.");
    for (const name of configured) {
      memberPath(directory, name);
      if (!models.includes(name))
        throw new Error(`GenAI package model is missing: ${name}`);
      genaiDirectories[name] = configDirectory;
    }
    const tokenizerDirectory =
      object(config) &&
      object(config.model) &&
      typeof config.model.tokenizer_dir === "string"
        ? packageReference(
            directory,
            configDirectory,
            config.model.tokenizer_dir,
            manifest,
          )
        : configDirectory;
    if (
      !files.some((name) =>
        /^(?:tokenizer\.json|tokenizer_config\.json|tokenizer\.model|vocab\.json)$/.test(
          relative(join(directory, tokenizerDirectory), join(directory, name))
            .split(sep)
            .join("/"),
        ),
      )
    )
      throw new Error("The GenAI package is missing tokenizer assets.");
    const decoder =
      config &&
      typeof config === "object" &&
      "model" in config &&
      config.model &&
      typeof config.model === "object" &&
      "decoder" in config.model
        ? filenames(config.model.decoder)[0]
        : undefined;
    if (configs.length === 1)
      defaultModel = decoder
        ? packageReference(directory, configDirectory, decoder, manifest)
        : configured[0];
  }
  return {
    directory,
    models,
    defaultModel,
    genai,
    genaiDirectories,
    genaiModels,
    genaiConfigurationDirectories: configs.map((name) =>
      dirname(name).split(sep).join("/"),
    ),
    ...(manifest ? { manifest } : {}),
    ...(manifestReferences ? { manifestReferences } : {}),
  };
}

export function runtimePackagePromptDirectory(
  packageInfo: RuntimePackage,
  modelPath: string,
): string | undefined {
  const name = relative(packageInfo.directory, modelPath).split(sep).join("/");
  const roots = new Set(
    packageInfo.genaiModels
      ?.filter((reference) => reference.model === name)
      .map((reference) => reference.directory),
  );
  if (roots.size > 1)
    throw new Error(
      "Prompt cannot choose between several GenAI configurations that reference this shared model.",
    );
  const directory =
    roots.values().next().value ?? packageInfo.genaiDirectories?.[name];
  return directory === undefined
    ? undefined
    : join(packageInfo.directory, directory);
}

export function assertPackageWeightLocations(
  packageInfo: RuntimePackage,
  modelPath: string,
  locations: readonly string[],
): void {
  for (const location of locations) {
    const path = relative(
      packageInfo.directory,
      resolve(dirname(modelPath), location),
    );
    if (path === ".." || path.startsWith(`..${sep}`) || isAbsolute(path))
      throw new Error(
        `Package weights must remain inside the extracted package: ${location}`,
      );
  }
}

export function runtimePackageModel(
  packageInfo: RuntimePackage,
  selected?: unknown,
): string {
  const name =
    selected === undefined || selected === ""
      ? packageInfo.defaultModel
      : selected;
  if (typeof name !== "string" || !packageInfo.models.includes(name))
    throw new Error(
      "Choose a model from the package (Inspect package lists its models).",
    );
  return memberPath(packageInfo.directory, name);
}

function updateGenaiFilename(
  value: unknown,
  directory: string,
  base: string,
  source: string,
  target: string,
  manifest?: Record<string, unknown>,
): void {
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) {
    if (
      key === "filename" &&
      typeof child === "string" &&
      packageReference(directory, base, child, manifest) === source
    )
      Reflect.set(
        value,
        key,
        relative(join(directory, base), join(directory, target))
          .split(sep)
          .join("/"),
      );
    else updateGenaiFilename(child, directory, base, source, target, manifest);
  }
}

export async function saveRuntimePackage(options: {
  destination: string;
  archive?: boolean;
  kind: RuntimePackageKind;
  sourcePackage?: RuntimePackage;
  sourcePath?: string;
  model: NativeModel;
  format: "onnx" | "ort";
  maxExternalFileSize?: number;
  includeRewritings: boolean;
  canPublish: () => boolean;
  publish: (temporary: string, destination: string) => Promise<void>;
}): Promise<string | undefined> {
  const { model, sourcePackage, sourcePath, kind, format } = options;
  if (kind === "genai" && !sourcePackage?.genai)
    throw new Error(
      "Open an existing GenAI package directory first. Save preserves its configuration and tokenizer; it does not generate them.",
    );
  if (
    kind === "compiled" &&
    (!sourcePackage ||
      !model.graph().nodes.some((node) => node.opType === "EPContext"))
  )
    throw new Error(
      "Open an existing EPContext package directory first. Save preserves compiled provider assets; it does not compile a model.",
    );
  let destination = resolve(options.destination);
  let ancestor = dirname(destination);
  const missing = [basename(destination)];
  for (;;) {
    try {
      destination = join(await realpath(ancestor), ...missing);
      break;
    } catch (error) {
      if (!(
        error instanceof Error &&
        "code" in error &&
        error.code === "ENOENT"
      ))
        throw error;
      missing.unshift(basename(ancestor));
      ancestor = dirname(ancestor);
    }
  }
  const parent = dirname(destination);
  if (sourcePackage) {
    const target = join(parent, basename(destination));
    const member = relative(sourcePackage.directory, target);
    if (
      !member ||
      member === ".." ||
      (!member.startsWith(`..${sep}`) && !isAbsolute(member))
    )
      throw new Error(
        "Save the package outside its source directory to preserve its assets.",
      );
  }
  try {
    await lstat(destination);
    throw new Error(
      "The package destination already exists. Choose a new directory.",
    );
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "ENOENT"))
      throw error;
  }
  await mkdir(parent, { recursive: true });
  const staging = await mkdtemp(join(parent, ".onnx-runtime-package-"));
  const archive = options.archive ?? /\.(?:ortpackage|zip)$/i.test(destination);
  const temporary = archive ? join(staging, "content") : staging;
  let published = false;
  try {
    if (archive) await mkdir(temporary);
    const snapshots = new Map<string, Awaited<ReturnType<typeof lstat>>>();
    if (sourcePackage) {
      for (const name of await packageFiles(
        sourcePackage.directory,
        "",
        async (name) => {
          await mkdir(memberPath(temporary, name), { recursive: true });
        },
      )) {
        if (!options.canPublish()) return;
        const source = memberPath(sourcePackage.directory, name);
        const target = memberPath(temporary, name);
        const before = await lstat(source);
        if (!before.isFile())
          throw new Error(`Package asset is not a regular file: ${name}`);
        await mkdir(dirname(target), { recursive: true });
        await copyFile(source, target);
        snapshots.set(name, before);
        const after = await lstat(source);
        if (
          !after.isFile() ||
          before.dev !== after.dev ||
          before.ino !== after.ino ||
          before.size !== after.size ||
          before.mtimeMs !== after.mtimeMs
        )
          throw new Error(`Package asset changed while saving: ${name}`);
      }
    }
    const originalName =
      sourcePackage && sourcePath
        ? relative(sourcePackage.directory, sourcePath).split(sep).join("/")
        : undefined;
    const references = (sourcePackage?.manifestReferences ?? []).filter(
      (reference) => reference.model === originalName,
    );
    const genaiModels = (sourcePackage?.genaiModels ?? []).filter(
      (reference) => reference.model === originalName,
    );
    const sharedModel =
      references.some((reference) => reference.shared) ||
      genaiModels.some((reference) => reference.shared) ||
      /^shared_assets\/sha256-[a-f0-9]{64}\//.test(originalName ?? "");
    const referenceDirectories = new Set([
      ...references.map((reference) => reference.variantDirectory),
      ...genaiModels.map((reference) => reference.directory),
    ]);
    const editedDirectory =
      referenceDirectories.values().next().value ?? "edited";
    let modelName = (
      (sharedModel && originalName
        ? join(editedDirectory, basename(originalName)).split(sep).join("/")
        : originalName) ??
      (model.graph().pipeline ? "model.pipe-onnx" : "model.onnx")
    ).replace(/\.(?:onnx|ort)$/i, format === "ort" ? ".ort" : ".onnx");
    if (
      originalName &&
      originalName !== modelName &&
      snapshots.has(modelName)
    ) {
      const stem = modelName.replace(/\.(?:onnx|ort)$/i, "");
      const extension = format === "ort" ? ".ort" : ".onnx";
      for (let index = 1; ; index++) {
        const candidate = `${stem}-copy${index === 1 ? "" : `-${index}`}${extension}`;
        if (!snapshots.has(candidate)) {
          modelName = candidate;
          break;
        }
      }
    }
    const modelPath = memberPath(temporary, modelName);
    await mkdir(dirname(modelPath), { recursive: true });
    const serialized = options.maxExternalFileSize
      ? await model.serializeFiles(
          options.maxExternalFileSize,
          `${basename(modelName)}.${randomBytes(12).toString("hex")}.data`,
          sourcePath,
          false,
          options.includeRewritings,
        )
      : {
          model: await model.serialize(
            undefined,
            sourcePath,
            options.includeRewritings,
            format,
          ),
          weights: [],
        };
    if (!options.canPublish()) return;
    for (const weight of serialized.weights)
      await writeFile(
        memberPath(dirname(modelPath), weight.name),
        weight.bytes,
        { flag: "wx" },
      );
    await writeFile(modelPath, serialized.model);
    const referenceTargets = new Map<string, string>();
    for (const referenceDirectory of referenceDirectories) {
      const base = sharedModel
        ? join(referenceDirectory, basename(modelName)).split(sep).join("/")
        : modelName;
      let target = base;
      if (base !== modelName && snapshots.has(base)) {
        const stem = base.replace(/\.(?:onnx|ort)$/i, "");
        for (let index = 1; ; index++) {
          const candidate = `${stem}-copy${index === 1 ? "" : `-${index}`}.${format}`;
          if (!snapshots.has(candidate)) {
            target = candidate;
            break;
          }
        }
      }
      if (!referenceTargets.has(referenceDirectory)) {
        referenceTargets.set(referenceDirectory, target);
        if (target !== modelName) {
          const path = memberPath(temporary, target);
          await mkdir(dirname(path), { recursive: true });
          await copyFile(modelPath, path);
          for (const weight of serialized.weights)
            await copyFile(
              memberPath(dirname(modelPath), weight.name),
              memberPath(dirname(path), weight.name),
            );
        }
      }
    }
    if (sourcePackage?.genai && originalName && originalName !== modelName) {
      for (const configDirectory of sourcePackage.genaiConfigurationDirectories ?? [
        ".",
      ]) {
        const root = join(temporary, configDirectory);
        const config = await genaiConfiguration(root);
        updateGenaiFilename(
          config,
          temporary,
          configDirectory,
          originalName,
          referenceTargets.get(
            packageReference(temporary, "", configDirectory),
          ) ?? modelName,
          sourcePackage.manifest,
        );
        await writeFile(
          join(root, "genai_config.json"),
          JSON.stringify(config, null, 2) + "\n",
        );
      }
    }
    if (originalName && originalName !== modelName)
      for (const reference of references) {
        const document = memberPath(temporary, reference.document);
        const info: unknown = JSON.parse(await readFile(document, "utf8"));
        let container = info;
        for (const key of reference.pointer.slice(0, -1)) {
          if (!object(container))
            throw new Error("Invalid runtime package model reference.");
          container = container[key];
        }
        if (!object(container))
          throw new Error("Invalid runtime package model reference.");
        container[reference.pointer.at(-1)!] = relative(
          join(temporary, reference.variantDirectory),
          memberPath(
            temporary,
            referenceTargets.get(reference.variantDirectory) ?? modelName,
          ),
        )
          .split(sep)
          .join("/");
        await writeFile(document, JSON.stringify(info, null, 2) + "\n");
      }
    if (originalName && originalName !== modelName && !sharedModel)
      await rm(memberPath(temporary, originalName));
    if (sourcePackage) {
      const files = await packageFiles(sourcePackage.directory);
      if (
        files.length !== snapshots.size ||
        files.some((name) => !snapshots.has(name))
      )
        throw new Error(
          "Package contents changed while saving. Reopen the package.",
        );
      for (const [name, before] of snapshots) {
        const after = await lstat(memberPath(sourcePackage.directory, name));
        if (
          !after.isFile() ||
          before.dev !== after.dev ||
          before.ino !== after.ino ||
          before.size !== after.size ||
          before.mtimeMs !== after.mtimeMs
        )
          throw new Error(`Package asset changed while saving: ${name}`);
      }
    }
    if (!options.canPublish()) return;
    if (archive) {
      if (!sourcePackage?.manifestReferences) {
        const variant = join(staging, "variant");
        await rename(temporary, variant);
        await mkdir(temporary);
        await rename(variant, join(temporary, "default"));
        const ep =
          kind === "genai"
            ? genaiProvider(
                await genaiConfiguration(join(temporary, "default")),
              )
            : kind === "generic" &&
                !model.graph().pipeline &&
                !model.graph().nodes.some((node) => node.opType === "EPContext")
              ? "CPUExecutionProvider"
              : undefined;
        const manifest = {
          schema_version: "1.0",
          components: {
            model: {
              variants: {
                default: {
                  ...(ep
                    ? {
                        ep,
                        ...(ep === "CPUExecutionProvider"
                          ? { device: "cpu" }
                          : {}),
                      }
                    : {}),
                  variant_directory: "default",
                  executor_info: {
                    [model.graph().pipeline ? "onnx-light-viewer" : "ort"]: {
                      model_file: modelName,
                    },
                  },
                },
              },
            },
          },
        };
        await writeFile(
          join(temporary, "manifest.json"),
          JSON.stringify(manifest, null, 2) + "\n",
        );
        modelName = `default/${modelName}`;
      }
      const archivePath = join(staging, "package.ortpackage");
      await writePackageArchive(temporary, archivePath);
      if (!options.canPublish()) return;
      await options.publish(archivePath, destination);
    } else {
      await options.publish(temporary, destination);
      published = true;
    }
    return modelName;
  } finally {
    if (!published) await rm(staging, { recursive: true, force: true });
  }
}
