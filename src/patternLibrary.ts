import {
  FunctionPatternDefinition,
  NativeModel,
  validateFunctionPatterns,
} from "./graph";

export interface PatternLibrarySnapshot {
  revision: number;
  patterns: FunctionPatternDefinition[];
}

/** Session-wide templates; portable files are the only cross-session persistence. */
export class PatternLibrary {
  private definitions = new Map<string, FunctionPatternDefinition>();
  private revision = 0;
  private installed = new WeakMap<NativeModel, number>();
  private listeners = new Set<
    (snapshot: PatternLibrarySnapshot, source?: object) => void
  >();

  snapshot(): PatternLibrarySnapshot {
    return {
      revision: this.revision,
      patterns: [...this.definitions.values()].map((pattern) => ({
        ...pattern,
      })),
    };
  }

  subscribe(
    listener: (snapshot: PatternLibrarySnapshot, source?: object) => void,
  ): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  install(model: NativeModel): void {
    if (this.installed.get(model) === this.revision) return;
    model.setFunctionPatterns(this.snapshot().patterns);
    this.installed.set(model, this.revision);
  }

  merge(patterns: FunctionPatternDefinition[], source?: object): void {
    const validated = validateFunctionPatterns(patterns);
    let changed = false;
    for (const pattern of validated) {
      const previous = this.definitions.get(pattern.name);
      if (
        previous?.text !== pattern.text ||
        previous.replacement !== pattern.replacement
      )
        changed = true;
    }
    if (!changed) return;
    for (const pattern of validated)
      this.definitions.set(pattern.name, pattern);
    this.revision++;
    const snapshot = this.snapshot();
    for (const listener of this.listeners) listener(snapshot, source);
  }

  update(
    name: string,
    text: string,
    replacement: string,
    revision: number,
  ): void {
    if (revision !== this.revision)
      throw new Error(
        "The pattern library changed. Revert edits before editing again.",
      );
    if (!this.definitions.has(name))
      throw new Error(`Unknown function pattern: ${name}`);
    this.merge([{ name, text, replacement }]);
  }

  conflicts(patterns: FunctionPatternDefinition[]): string[] {
    return patterns
      .filter((pattern) => {
        const previous = this.definitions.get(pattern.name);
        return (
          previous &&
          (previous.text !== pattern.text ||
            previous.replacement !== pattern.replacement)
        );
      })
      .map((pattern) => pattern.name);
  }

  serialize(): Uint8Array {
    return Buffer.from(
      `${JSON.stringify(
        {
          format: "onnx-light-viewer.patterns",
          version: 1,
          patterns: this.snapshot().patterns,
        },
        null,
        2,
      )}\n`,
      "utf8",
    );
  }

  parse(bytes: Uint8Array): FunctionPatternDefinition[] {
    const file: unknown = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    );
    if (
      !file ||
      typeof file !== "object" ||
      !("format" in file) ||
      file.format !== "onnx-light-viewer.patterns" ||
      !("version" in file) ||
      file.version !== 1 ||
      !("patterns" in file) ||
      !Array.isArray(file.patterns)
    )
      throw new Error(
        "Expected an ONNX Light Viewer pattern library (version 1).",
      );
    return validateFunctionPatterns(file.patterns);
  }
}
