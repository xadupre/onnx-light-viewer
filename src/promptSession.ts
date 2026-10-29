import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { resolve } from "node:path";
import { terminateExport } from "./export";

/** One native worker per panel; never retains two engines or models together. */
export class PromptSession {
  private child?: ChildProcessWithoutNullStreams;
  private closed: Promise<void> = Promise.resolve();
  private closing?: Promise<void>;
  private key?: string;
  private pending?: {
    event: (value: unknown) => void;
    progress: (value: string) => void;
    resolve: () => void;
    reject: (error: Error) => void;
  };

  constructor(
    private readonly reportError: (error: Error) => void = console.error,
  ) {}

  private start(): void {
    const child = spawn(
      process.execPath,
      [resolve(__dirname, "..", "media", "run_prompt.cjs"), "--worker"],
      {
        env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" },
        stdio: ["pipe", "pipe", "pipe"],
        detached: process.platform !== "win32",
      },
    );
    this.child = child;
    let pending = "";
    let output = "";
    let timer: ReturnType<typeof setTimeout> | undefined;
    const flush = () => {
      timer = undefined;
      this.pending?.progress(output);
    };
    const log = (chunk: string) => {
      output = (output + chunk).slice(-65536);
      if (!timer) timer = setTimeout(flush, 100);
    };
    const fail = (error: Error) => {
      if (this.pending) this.pending.reject(error);
      else if (this.child === child) this.reportError(error);
    };
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", log);
    child.stdin.on("error", fail);
    child.stdout.on("data", (chunk: string) => {
      try {
        pending += chunk;
        let newline: number;
        while ((newline = pending.indexOf("\n")) !== -1) {
          const line = pending.slice(0, newline);
          pending = pending.slice(newline + 1);
          const marker = line.indexOf("\x1e");
          if (marker === -1) {
            log(line + "\n");
            continue;
          }
          if (marker) log(line.slice(0, marker));
          const event: unknown = JSON.parse(line.slice(marker + 1));
          if (!event || typeof event !== "object" || !("type" in event))
            throw new Error("Invalid native prompt event.");
          if (event.type === "done") {
            if (timer) clearTimeout(timer);
            flush();
            this.pending?.resolve();
            this.pending = undefined;
            output = "";
          } else if (event.type === "error") {
            throw new Error(
              "error" in event && typeof event.error === "string"
                ? event.error
                : "Native prompt failed.",
            );
          } else {
            this.pending?.event(event);
          }
        }
        if (pending.length > 1_000_000)
          throw new Error("The native runner returned an oversized event.");
      } catch (error) {
        fail(error instanceof Error ? error : new Error(String(error)));
      }
    });
    child.once("error", fail);
    this.closed = new Promise((resolve) => {
      child.once("close", (code, signal) => {
        if (timer) clearTimeout(timer);
        if (this.child === child) {
          fail(
            new Error(`Native prompt exited (${signal ?? code}).\n${output}`),
          );
          this.child = undefined;
        }
        resolve();
      });
    });
  }

  async run(
    input: string,
    output: string,
    key: string,
    signal: AbortSignal,
    progress: (output: string) => void,
    event: (value: unknown) => void,
  ): Promise<void> {
    if (this.closing) await this.closing;
    if (this.child && this.key !== key) await this.close();
    signal.throwIfAborted();
    if (this.pending) throw new Error("A native prompt is already running.");
    if (!this.child) this.start();
    this.key = key;
    const child = this.child!;
    let abort: () => void = () => {};
    try {
      await new Promise<void>((resolve, reject) => {
        const request = { resolve, reject, progress, event };
        this.pending = request;
        abort = () =>
          reject(new DOMException("Prompt cancelled.", "AbortError"));
        signal.addEventListener("abort", abort, { once: true });
        if (signal.aborted) abort();
        else child.stdin.write(JSON.stringify({ input, output, key }) + "\n");
      });
    } catch (error) {
      await this.close();
      throw error;
    } finally {
      signal.removeEventListener("abort", abort);
    }
  }

  close(): Promise<void> {
    if (this.closing) return this.closing;
    const child = this.child;
    if (!child) return Promise.resolve();
    const closed = this.closed;
    const running = this.pending;
    this.child = undefined;
    this.pending = undefined;
    running?.reject(new DOMException("Prompt session released.", "AbortError"));
    this.closing = (async () => {
      if (running) {
        await terminateExport(child);
      } else {
        child.stdin.end();
        let timer: ReturnType<typeof setTimeout> | undefined;
        const timedOut = await Promise.race([
          closed.then(() => false),
          new Promise<boolean>((resolve) => {
            timer = setTimeout(() => resolve(true), 2000);
          }),
        ]);
        if (timer) clearTimeout(timer);
        if (timedOut) await terminateExport(child);
      }
      await closed;
    })().finally(() => {
      this.closing = undefined;
    });
    return this.closing;
  }
}
