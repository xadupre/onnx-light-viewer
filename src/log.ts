export type LogStatus =
  "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED" | "INVALIDATED";

export interface LogEntry {
  id: string;
  timestamp: string;
  action: string;
  status: LogStatus;
  details?: string;
}

export class ActionLog {
  private readonly entries = new Map<string, LogEntry>();
  private nextId = 0;

  constructor(private readonly publish: (entry: LogEntry) => void) {}

  snapshot(): LogEntry[] {
    return [...this.entries.values()];
  }

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
