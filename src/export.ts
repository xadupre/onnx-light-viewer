import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readdir, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, isAbsolute, join, resolve } from "node:path";
import { stripVTControlCharacters } from "node:util";
import type { Memento } from "vscode";

const exportEnvironment = {
  PYTHONUNBUFFERED: "1",
  HF_HUB_DISABLE_PROGRESS_BARS: "0",
  TQDM_POSITION: "-1",
};

type Exporter = "mbext" | "modelbuilder" | "mobius";

function isExporter(value: unknown): value is Exporter {
  return value === "mbext" || value === "modelbuilder" || value === "mobius";
}

interface ExportHistorySnapshot {
  models: string[];
  directories: string[];
  exporter?: Exporter;
}

/** Keeps recent export inputs across sessions and serializes edits across panels. */
export class ExportHistory {
  private pending: Promise<void> = Promise.resolve();

  constructor(private readonly storage: Pick<Memento, "get" | "update">) {}

  snapshot(): ExportHistorySnapshot {
    const value = this.storage.get<unknown>("exportHistory");
    if (value === undefined) return { models: [], directories: [] };
    if (
      !value ||
      typeof value !== "object" ||
      !("models" in value) ||
      !("directories" in value) ||
      !Array.isArray(value.models) ||
      !Array.isArray(value.directories) ||
      ![...value.models, ...value.directories].every(
        (entry) =>
          typeof entry === "string" && entry.trim() && !/[\0\r\n]/.test(entry),
      )
    )
      throw new Error("Invalid saved export history.");
    const exporter = "exporter" in value ? value.exporter : undefined;
    if (exporter !== undefined && !isExporter(exporter))
      throw new Error("Invalid saved conversion tool.");
    return {
      models: [...value.models],
      directories: [...value.directories],
      ...(exporter === undefined ? {} : { exporter }),
    };
  }

  remember(plan: ExportPlan): Promise<void> {
    const operation = this.pending.then(async () => {
      const history = this.snapshot();
      const recent = (value: string, entries: string[]) =>
        [value, ...entries.filter((entry) => entry !== value)].slice(0, 50);
      const directories = recent(plan.directory, history.directories);
      await this.storage.update("exportHistory", {
        exporter: plan.exporter,
        models: recent(plan.model, history.models),
        directories: plan.directoryTemplate
          ? recent(plan.directoryTemplate, directories)
          : directories,
      });
    });
    this.pending = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }
}

export function defaultExportDirectory(root: string): string {
  return join(root, "<model_id>", "<exporter>");
}

export interface ExportPlan {
  exporter: Exporter;
  model: string;
  precision: string;
  provider: "cpu" | "cuda";
  executable: string;
  args: string[];
  directory: string;
  directoryTemplate?: string;
  command: string;
  script: string;
}

/** Builds an argument vector, never an executable shell command. */
export function exportPlan(
  value: unknown,
  pythonPath: string,
  downloadDirectory?: string,
): ExportPlan {
  if (
    !value ||
    typeof value !== "object" ||
    !("exporter" in value) ||
    !isExporter(value.exporter) ||
    !("model" in value) ||
    typeof value.model !== "string" ||
    !value.model.trim() ||
    value.model.trim().startsWith("-") ||
    /[\0\r\n]/.test(value.model) ||
    !("directory" in value) ||
    typeof value.directory !== "string" ||
    /[\0\r\n]/.test(value.directory) ||
    !("precision" in value) ||
    typeof value.precision !== "string" ||
    !["fp32", "fp16", "bf16", "int4", "int2", "fp4"].includes(
      value.precision,
    ) ||
    !("provider" in value) ||
    (value.provider !== "cpu" && value.provider !== "cuda")
  )
    throw new Error(
      "Choose an exporter, model, absolute output directory, precision and device.",
    );
  const model = value.model.trim();
  let directory = value.directory;
  if (directory.includes("<model_id>")) {
    const parts = isAbsolute(model) ? [basename(model)] : model.split(/[\\/]/);
    if (
      parts.some(
        (part) =>
          !part ||
          part === "." ||
          part === ".." ||
          /[<>:"|?*\x00-\x1f]/.test(part) ||
          /[. ]$/.test(part),
      )
    )
      throw new Error("The model ID cannot be used in a directory template.");
    directory = directory.replaceAll("<model_id>", join(...parts));
  }
  directory = directory.replaceAll("<exporter>", value.exporter);
  if (/[<>]/.test(directory))
    throw new Error(
      "Directory templates support only <model_id> and <exporter>.",
    );
  if (!isAbsolute(directory)) {
    if (!downloadDirectory || !isAbsolute(downloadDirectory))
      throw new Error("Enter an absolute output directory.");
    directory = resolve(downloadDirectory, directory);
  }
  const precision = value.precision;
  const mobius = value.exporter === "mobius";
  const args = mobius
    ? [
        "-u",
        "-m",
        "mobius",
        "build",
        isAbsolute(model) ? "--config" : "--model",
        model,
        "--output",
        directory,
        "--dtype",
        precision === "fp32" ? "f32" : precision === "fp16" ? "f16" : precision,
        "--ep",
        value.provider,
      ]
    : [
        "-u",
        "-m",
        value.exporter === "mbext"
          ? "modelbuilder.builder"
          : "onnxruntime_genai.models.builder",
        isAbsolute(model) ? "-i" : "-m",
        model,
        "-o",
        directory,
        "-p",
        precision,
        "-e",
        value.provider,
      ];
  const quote = (argument: string) => {
    const safe =
      process.platform === "win32" ? /^[\w./\\:-]+$/ : /^[\w@%+=:,./-]+$/;
    if (safe.test(argument)) return argument;
    return process.platform === "win32"
      ? `'${argument.replaceAll("'", "''")}'`
      : `'${argument.replaceAll("'", "'\\''")}'`;
  };
  return {
    exporter: value.exporter,
    model,
    precision,
    provider: value.provider,
    executable: pythonPath,
    args,
    directory,
    ...(directory === value.directory
      ? {}
      : { directoryTemplate: value.directory }),
    command: `${process.platform === "win32" ? "& " : ""}${[pythonPath, ...args].map(quote).join(" ")}`,
    script: join(directory, "export_model.py"),
  };
}

/** Keeps existing output intact, reserving a fresh subfolder for another attempt. */
export async function prepareExportPlan(
  plan: ExportPlan,
  signal: AbortSignal,
): Promise<ExportPlan> {
  signal.throwIfAborted();
  await mkdir(plan.directory, { recursive: true });
  const entries = await readdir(plan.directory);
  signal.throwIfAborted();
  if (!entries.length) return plan;
  const directory = await mkdtemp(join(plan.directory, `${plan.exporter}-`));
  signal.throwIfAborted();
  return {
    ...exportPlan({ ...plan, directory }, plan.executable),
    ...(plan.directoryTemplate === undefined
      ? {}
      : { directoryTemplate: plan.directoryTemplate }),
  };
}

/** Runs the selected CLI with bounded, throttled output and explicit cancellation. */
export async function runExport(
  plan: ExportPlan,
  signal: AbortSignal,
  progress: (output: string) => void,
): Promise<string[]> {
  signal.throwIfAborted();
  await mkdir(plan.directory, { recursive: true });
  if ((await readdir(plan.directory)).length)
    throw new Error(
      "The conversion directory is no longer empty. Retry to create a new subfolder.",
    );
  const script = `import os\nimport subprocess\n\nsubprocess.run(${JSON.stringify([plan.executable, ...plan.args])}, cwd=${JSON.stringify(homedir())}, env={**os.environ, **${JSON.stringify(exportEnvironment)}}, check=True)\n`;
  await writeFile(plan.script, script, { flag: "wx" });
  signal.throwIfAborted();
  await new Promise<void>((resolve, reject) => {
    const child = spawn(plan.executable, plan.args, {
      cwd: homedir(),
      env: { ...process.env, ...exportEnvironment },
      stdio: ["ignore", "pipe", "pipe"],
      signal,
    });
    let output = "";
    let timer: ReturnType<typeof setTimeout> | undefined;
    let forceKill: ReturnType<typeof setTimeout> | undefined;
    let finished = false;
    const readableOutput = () =>
      stripVTControlCharacters(output).replace(/\r\n?/g, "\n");
    const flush = () => {
      timer = undefined;
      progress(readableOutput());
    };
    const finish = (error?: Error) => {
      if (finished) return;
      finished = true;
      if (timer) clearTimeout(timer);
      if (forceKill) clearTimeout(forceKill);
      flush();
      if (error) reject(error);
      else resolve();
    };
    for (const stream of [child.stdout, child.stderr]) {
      stream.setEncoding("utf8");
      stream.on("data", (chunk: string) => {
        output = (output + chunk).slice(-65536);
        if (!timer) timer = setTimeout(flush, 100);
      });
    }
    child.once("error", (error) => {
      if (signal.aborted && "code" in error && error.code === "ABORT_ERR")
        forceKill = setTimeout(() => child.kill("SIGKILL"), 2000);
      else finish(error);
    });
    child.once("close", (code, termination) =>
      finish(
        signal.aborted
          ? new DOMException("Export cancelled.", "AbortError")
          : code === 0
            ? undefined
            : new Error(
                `Exporter exited with ${termination ? `signal ${termination}` : `code ${code}`}.\n${readableOutput()}`,
              ),
      ),
    );
  });
  signal.throwIfAborted();
  const models: string[] = [];
  const visit = async (directory: string) => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      signal.throwIfAborted();
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (entry.isFile() && /\.(?:onnx|pipe-onnx)$/i.test(entry.name))
        models.push(path);
    }
  };
  await visit(plan.directory);
  if (!models.length)
    throw new Error(
      "The exporter completed but produced no ONNX model or pipeline.",
    );
  return models.sort();
}
