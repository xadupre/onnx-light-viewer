/**
 * Panel-local operation history shared by host actions and webview interactions.
 * Entries are updated by ID so start/completion events occupy the same row.
 */

/** Operation state; INVALIDATED records a discarded native handle after failure. */
export type LogStatus =
  "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED" | "INVALIDATED";

/** Serializable action snapshot sent to the webview, with optional error details. */
export interface LogEntry {
  id: string;
  /** Original start time, preserved across updates to the same ID. */
  timestamp: string;
  action: string;
  status: LogStatus;
  details?: string;
}

/** Retains the latest entry per ID in insertion order for one viewer panel. */
export class ActionLog {
  private readonly entries = new Map<string, LogEntry>();
  private nextId = 0;

  /** @param publish Receives each new or updated entry synchronously. */
  constructor(private readonly publish: (entry: LogEntry) => void) {}

  /** Returns history in insertion order; entries themselves are not cloned. */
  snapshot(): LogEntry[] {
    return [...this.entries.values()];
  }

  /**
   * Creates or replaces an entry and publishes it, retaining its first timestamp.
   * @param status New operation state.
   * @param action Human-readable action description.
   * @param id Correlation ID; defaults to a generated host:N identifier.
   * @param details Optional diagnostic text, such as an error stack.
   * @param timestamp Initial timestamp; defaults to the current ISO time.
   */
  record(
    status: LogStatus,
    action: string,
    id = `host:${++this.nextId}`,
    details?: string,
    timestamp?: string,
  ): LogEntry {
    const entry: LogEntry = {
      id,
      timestamp:
        this.entries.get(id)?.timestamp ??
        timestamp ??
        new Date().toISOString(),
      action,
      status,
      ...(details ? { details } : {}),
    };
    this.entries.set(id, entry);
    this.publish(entry);
    return entry;
  }

  /**
   * Validates and records an untrusted webview interaction.
   * Only ui: IDs, parseable timestamps and RUNNING/SUCCEEDED/FAILED are accepted.
   * @throws If the payload does not match the interaction contract.
   */
  interaction(value: unknown): void {
    if (
      typeof value !== "object" ||
      value === null ||
      !("id" in value) ||
      typeof value.id !== "string" ||
      !value.id.startsWith("ui:") ||
      !("action" in value) ||
      typeof value.action !== "string" ||
      !("timestamp" in value) ||
      typeof value.timestamp !== "string" ||
      Number.isNaN(Date.parse(value.timestamp)) ||
      !("status" in value) ||
      (value.status !== "RUNNING" &&
        value.status !== "SUCCEEDED" &&
        value.status !== "FAILED") ||
      ("details" in value && typeof value.details !== "string")
    ) {
      throw new Error("Invalid interaction log entry.");
    }
    this.record(
      value.status,
      value.action,
      value.id,
      "details" in value && typeof value.details === "string"
        ? value.details
        : undefined,
      value.timestamp,
    );
  }

  /**
   * Records RUNNING, awaits work and records SUCCEEDED if still running.
   * Work may record another terminal status (for example CANCELLED) using its ID.
   * Failures always record FAILED with diagnostic details and rethrow the error.
   * @param action Human-readable operation name.
   * @param work Receives the correlation ID for explicit status updates.
   * @param id Existing UI/host ID to update, or omitted to allocate a host ID.
   * @returns The callback's result, unchanged.
   */
  async run<T>(
    action: string,
    work: (id: string) => T | Promise<T>,
    id?: string,
  ): Promise<T> {
    const entry = this.record("RUNNING", action, id);
    try {
      const result = await work(entry.id);
      if (this.entries.get(entry.id)?.status === "RUNNING")
        this.record("SUCCEEDED", action, entry.id);
      return result;
    } catch (error) {
      const details =
        error instanceof Error ? (error.stack ?? error.message) : String(error);
      this.record("FAILED", action, entry.id, details);
      throw error;
    }
  }
}
