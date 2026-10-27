import { execFile } from "node:child_process";
import {
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { promisify } from "node:util";
import { runProcess } from "./export";

export interface OliveRecipe {
  id: string;
  name: string;
  requirements: string[];
  target: string;
  manualReason?: string;
}

export interface OliveRecipeCatalog {
  revision: string;
  recipes: OliveRecipe[];
}

export interface PinnedOliveRecipe extends OliveRecipe {
  revision: string;
  snapshot: string;
}

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

function inside(root: string, path: string): boolean {
  const suffix = relative(root, path);
  return (
    suffix !== ".." && !suffix.startsWith(`..${sep}`) && !isAbsolute(suffix)
  );
}

async function catalog(snapshot: string): Promise<OliveRecipe[]> {
  const { parse } = await import("yaml");
  snapshot = await realpath(snapshot);
  const recipes: OliveRecipe[] = [];
  const requirements = new Map<string, string[]>();
  const requirementsAt = async (directory: string): Promise<string[]> => {
    const cached = requirements.get(directory);
    if (cached) return cached;
    const result =
      directory === snapshot
        ? []
        : [...(await requirementsAt(dirname(directory)))];
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (!entry.isFile() || !/^requirements.*\.txt$/i.test(entry.name))
        continue;
      const contents = await readFile(join(directory, entry.name), "utf8");
      for (const line of contents.split(/\r?\n/)) {
        const value = line.trim();
        if (
          /^(?:olive-ai\b|(?:-e\s+)?git\+https:\/\/github\.com\/microsoft\/Olive\b)/i.test(
            value,
          )
        )
          result.push(value);
      }
    }
    requirements.set(directory, result);
    return result;
  };
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (entry.name.startsWith(".")) continue;
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (entry.isFile() && /^info\.ya?ml$/i.test(entry.name)) {
        const info: unknown = parse(await readFile(path, "utf8"));
        if (!record(info) || info.recipes === undefined) continue;
        if (!Array.isArray(info.recipes))
          throw new Error(`Invalid Olive recipe metadata: ${path}`);
        for (const [index, value] of info.recipes.entries()) {
          if (
            !record(value) ||
            (value.name !== undefined && typeof value.name !== "string") ||
            (value.file !== undefined && typeof value.file !== "string") ||
            (value.name === undefined && value.file === undefined)
          )
            throw new Error(`Invalid Olive recipe entry: ${path}`);
          const filename =
            typeof value.file === "string"
              ? value.file
              : typeof value.name === "string" && value.name.endsWith(".json")
                ? value.name
                : undefined;
          let file = filename ? resolve(directory, filename) : path;
          if (!inside(snapshot, file))
            throw new Error(
              `Recipe configuration must stay inside its snapshot: ${filename}`,
            );
          let manualReason = filename?.endsWith(".json")
            ? undefined
            : "This recipe has no standard Olive JSON workflow. Follow its source instructions.";
          try {
            file = await realpath(file);
          } catch (error) {
            if (!(
              error instanceof Error &&
              "code" in error &&
              error.code === "ENOENT"
            ))
              throw error;
            manualReason =
              "The recipe entry points to a missing file. Follow its source instructions.";
          }
          if (!inside(snapshot, file))
            throw new Error(
              `Recipe configuration must stay inside its snapshot: ${filename}`,
            );
          const id =
            relative(snapshot, file).split(sep).join("/") +
            (manualReason ? `#${index}` : "");
          const target = [value.devices ?? value.device, value.eps ?? value.ep]
            .flat()
            .filter((item): item is string => typeof item === "string")
            .join(", ");
          recipes.push({
            id,
            name:
              typeof value.name === "string"
                ? value.name
                : id.replace(/\.json$/, ""),
            requirements: [...new Set(await requirementsAt(directory))],
            target,
            ...(manualReason ? { manualReason } : {}),
          });
        }
      }
    }
  };
  await visit(snapshot);
  recipes.sort(
    (a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id),
  );
  if (!recipes.length)
    throw new Error("The Olive recipe repository contains no recipes.");
  return recipes;
}

/** Keeps immutable checkouts and their model lists keyed by the upstream Git commit. */
export class OliveRecipes {
  private current?: OliveRecipeCatalog;
  private pending?: Promise<OliveRecipeCatalog>;

  constructor(
    private readonly directory: string,
    private readonly remote = "https://github.com/microsoft/olive-recipes.git",
  ) {}

  refresh(signal: AbortSignal): Promise<OliveRecipeCatalog> {
    if (this.pending) return this.pending;
    this.pending = this.update(signal).then(
      (value) => {
        this.pending = undefined;
        return value;
      },
      (error: unknown) => {
        this.pending = undefined;
        throw error;
      },
    );
    return this.pending;
  }

  private async update(signal: AbortSignal): Promise<OliveRecipeCatalog> {
    const { stdout } = await promisify(execFile)(
      "git",
      ["ls-remote", "--", this.remote, "HEAD"],
      {
        signal,
        timeout: 30000,
        maxBuffer: 1024 * 1024,
        env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
      },
    );
    const revision = stdout.split(/\s+/)[0];
    if (!/^[a-f0-9]{40}$/.test(revision))
      throw new Error("Olive recipes did not provide a Git commit.");
    if (this.current?.revision === revision) return this.current;
    const destination = join(this.directory, revision);
    try {
      const value: unknown = JSON.parse(
        await readFile(join(destination, "catalog.json"), "utf8"),
      );
      if (
        !record(value) ||
        value.revision !== revision ||
        !Array.isArray(value.recipes) ||
        !value.recipes.length ||
        !value.recipes.every(
          (entry) =>
            record(entry) &&
            typeof entry.id === "string" &&
            typeof entry.name === "string" &&
            typeof entry.target === "string" &&
            (entry.manualReason === undefined ||
              typeof entry.manualReason === "string") &&
            inside(
              join(destination, "source"),
              resolve(destination, "source", entry.id),
            ) &&
            Array.isArray(entry.requirements) &&
            entry.requirements.every((item) => typeof item === "string"),
        )
      )
        throw new Error("Invalid cached Olive recipe catalog.");
      this.current = { revision, recipes: value.recipes };
      return this.current;
    } catch (error) {
      if (!(
        error instanceof Error &&
        "code" in error &&
        error.code === "ENOENT"
      ))
        throw error;
    }
    await mkdir(this.directory, { recursive: true });
    const staging = await mkdtemp(join(this.directory, ".checkout-"));
    try {
      const snapshot = join(staging, "source");
      const run = (args: string[]) =>
        runProcess(
          { executable: "git", args },
          signal,
          () => {},
          { GIT_TERMINAL_PROMPT: "0" },
          "Olive recipe checkout",
        );
      await run([
        "clone",
        "--depth",
        "1",
        "--no-checkout",
        "--",
        this.remote,
        snapshot,
      ]);
      await run(["-C", snapshot, "fetch", "--depth", "1", "origin", revision]);
      await run(["-C", snapshot, "checkout", "--detach", revision]);
      const value = { revision, recipes: await catalog(snapshot) };
      signal.throwIfAborted();
      const serialized = JSON.stringify(value);
      await writeFile(join(staging, "catalog.json"), serialized);
      try {
        await rename(staging, destination);
      } catch (error) {
        if (!(
          error instanceof Error &&
          "code" in error &&
          (error.code === "EEXIST" ||
            error.code === "ENOTEMPTY" ||
            (process.platform === "win32" &&
              (error.code === "EPERM" || error.code === "EACCES")))
        ))
          throw error;
        if (
          (await readFile(join(destination, "catalog.json"), "utf8")) !==
          serialized
        )
          throw new Error(
            "Conflicting Olive recipe catalogs for the same Git commit.",
          );
      }
      this.current = value;
      return value;
    } finally {
      await rm(staging, { recursive: true, force: true });
    }
  }

  resolve(id: unknown, revision: unknown): PinnedOliveRecipe {
    if (!this.current || revision !== this.current.revision)
      throw new Error("Refresh the Olive recipe list before converting.");
    const recipe = this.current.recipes.find((entry) => entry.id === id);
    if (!recipe) throw new Error("Choose a recipe from the Olive recipe list.");
    if (recipe.manualReason) throw new Error(recipe.manualReason);
    return {
      ...recipe,
      revision: this.current.revision,
      snapshot: join(this.directory, this.current.revision, "source"),
    };
  }
}
