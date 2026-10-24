import type { Memento } from "vscode";

export type HistoryTarget = "open" | "save";
export type HistorySource = "local" | "huggingface" | "backend";
export interface HistoryEntry {
  source: HistorySource;
  kind: "file" | "directory";
  path: string;
}
export type HistorySnapshot = Record<HistoryTarget, HistoryEntry[]>;
type HistoryChange =
  | { action: "remember"; entries: HistoryEntry[] }
  | { action: "remove"; entry: HistoryEntry }
  | { action: "clear" };

export function isHistoryEntry(value: unknown): value is HistoryEntry {
  return (
    value !== null &&
    typeof value === "object" &&
    "source" in value &&
    typeof value.source === "string" &&
    ["local", "huggingface", "backend"].includes(value.source) &&
    "kind" in value &&
    (value.kind === "file" || value.kind === "directory") &&
    "path" in value &&
    typeof value.path === "string" &&
    value.path.length > 0 &&
    !value.path.includes("\0")
  );
}

function sameEntry(left: HistoryEntry, right: HistoryEntry): boolean {
  return (
    left.source === right.source &&
    left.kind === right.kind &&
    left.path === right.path
  );
}

export function historyEntries(
  source: HistorySource,
  path: string,
  directory?: string,
): HistoryEntry[] {
  return [
    { source, kind: "file" as const, path },
    ...(directory
      ? [{ source, kind: "directory" as const, path: directory }]
      : []),
  ].map((entry) => {
    if (
      /^[a-z][a-z\d+.-]*:/i.test(entry.path) &&
      !/^[a-z]:/i.test(entry.path)
    ) {
      const url = new URL(entry.path);
      url.username = "";
      url.password = "";
      url.search = "";
      url.hash = "";
      return { ...entry, path: url.toString() };
    }
    return entry;
  });
}

/** Serializes history edits across panels and publishes only persisted snapshots. */
export class PathHistory {
  private pending: Promise<void> = Promise.resolve();
  private listeners = new Set<(history: HistorySnapshot) => void>();

  constructor(
    private readonly storage: Pick<Memento, "get" | "update">,
    private readonly reportInvalid: (message: string) => void = console.warn,
  ) {}

  snapshot(): HistorySnapshot {
    const stored = this.storage.get<unknown>("pathHistory");
    if (stored === undefined) return { open: [], save: [] };
    const history = stored && typeof stored === "object" ? stored : {};
    let invalid = false;
    const entries = (value: unknown): HistoryEntry[] => {
      if (!Array.isArray(value)) {
        invalid = true;
        return [];
      }
      const valid = value.filter(isHistoryEntry);
      if (valid.length !== value.length) invalid = true;
      return valid.map(({ source, kind, path }) => ({ source, kind, path }));
    };
    const snapshot = {
      open: entries("open" in history ? history.open : undefined),
      save: entries("save" in history ? history.save : undefined),
    };
    if (invalid)
      this.reportInvalid("Ignoring invalid saved path history entries.");
    return snapshot;
  }

  subscribe(listener: (history: HistorySnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  update(target: HistoryTarget, change: HistoryChange): Promise<void> {
    const operation = this.pending.then(async () => {
      const history = this.snapshot();
      const previous = history[target];
      history[target] =
        change.action === "clear"
          ? []
          : change.action === "remove"
            ? previous.filter((entry) => !sameEntry(entry, change.entry))
            : [
                ...change.entries,
                ...previous.filter(
                  (entry) =>
                    !change.entries.some((recent) => sameEntry(entry, recent)),
                ),
              ].slice(0, 50);
      await this.storage.update("pathHistory", history);
      for (const listener of this.listeners) listener(history);
    });
    this.pending = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }
}
