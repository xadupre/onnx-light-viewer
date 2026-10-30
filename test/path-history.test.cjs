const assert = require("node:assert/strict");
const { test } = require("node:test");
const { PathHistory, historyEntries } = require("../dist/pathHistory");

test("reopening a remembered path preserves every entry and its order without persisting an edit", async () => {
  const first = historyEntries("local", "/models/first.onnx", "/models");
  const second = historyEntries("local", "/models/second.onnx", "/models");
  let stored = { open: [second[0], ...first], save: first };
  const expected = structuredClone(stored);
  const writes = [];
  const published = [];
  const history = new PathHistory({
    get: () => stored,
    update: async (key, value) => {
      writes.push([key, value]);
      stored = value;
    },
  });
  history.subscribe((value) => published.push(value));
  for (const target of ["open", "save"])
    await history.update(target, { action: "remember", entries: first });
  assert.deepEqual(history.snapshot(), expected);
  assert.deepEqual(writes, []);
  assert.deepEqual(published, []);
});

test("serializes recent path edits, deduplicates, bounds and persists separate action histories", async () => {
  const values = new Map();
  const storage = {
    get: (key) => values.get(key),
    update: async (key, value) => {
      values.set(key, value);
    },
  };
  const history = new PathHistory(storage);
  const first = historyEntries("local", "/models/first.onnx", "/models");
  const second = historyEntries("local", "/models/second.onnx", "/models");
  await Promise.all([
    history.update("open", { action: "remember", entries: first }),
    history.update("open", { action: "remember", entries: second }),
    history.update("save", { action: "remember", entries: first }),
  ]);
  assert.deepEqual(
    history.snapshot().open.map(({ path }) => path),
    ["/models/second.onnx", "/models/first.onnx", "/models"],
  );
  await history.update("open", { action: "remember", entries: first });
  assert.deepEqual(
    history.snapshot().open.map(({ path }) => path),
    ["/models/second.onnx", "/models/first.onnx", "/models"],
  );
  await history.update("open", { action: "remove", entry: first[0] });
  const reopened = new PathHistory(storage);
  assert.ok(
    !reopened.snapshot().open.some(({ path }) => path === first[0].path),
  );
  assert.deepEqual(reopened.snapshot().save, first);
  for (let index = 0; index < 60; index++)
    await history.update("open", {
      action: "remember",
      entries: historyEntries("backend", String(index)),
    });
  assert.equal(history.snapshot().open.length, 50);
  assert.equal(history.snapshot().open[0].path, "59");
  await history.update("open", { action: "clear" });
  assert.deepEqual(reopened.snapshot().open, []);
  assert.deepEqual(reopened.snapshot().save, first);
});

test("prunes missing entries in order, persists once, and serializes cleanup with new paths", async () => {
  const missing = historyEntries("local", "/missing.onnx")[0];
  const existing = historyEntries("local", "/existing.onnx")[0];
  const added = historyEntries("local", "/added.onnx")[0];
  let stored = { open: [missing, existing], save: [existing, missing] };
  const writes = [];
  const published = [];
  const history = new PathHistory({
    get: () => stored,
    update: async (_key, value) => {
      writes.push(value);
      stored = value;
    },
  });
  history.subscribe((value) => published.push(value));
  const [changed] = await Promise.all([
    history.prune(async (entry) => entry.path !== missing.path),
    history.update("open", { action: "remember", entries: [added] }),
  ]);
  assert.equal(changed, true);
  assert.deepEqual(stored, { open: [added, existing], save: [existing] });
  assert.deepEqual(published, writes);
  assert.equal(writes.length, 2);
  assert.equal(await history.prune(async () => true), false);
  assert.equal(writes.length, 2);
});

test("does not prune or publish paths when checking or persisting history fails", async () => {
  const entry = historyEntries("local", "/existing.onnx")[0];
  let stored = { open: [entry], save: [] };
  let fail = false;
  const history = new PathHistory({
    get: () => stored,
    update: async (_key, value) => {
      if (fail) throw new Error("Storage unavailable");
      stored = value;
    },
  });
  const published = [];
  history.subscribe((value) => published.push(value));
  const expected = structuredClone(stored);
  await assert.rejects(
    history.prune(async () => {
      throw new Error("Permission denied");
    }),
    /Permission denied/,
  );
  fail = true;
  await assert.rejects(
    history.prune(async () => false),
    /Storage unavailable/,
  );
  assert.deepEqual(stored, expected);
  assert.deepEqual(published, []);
  fail = false;
  assert.equal(await history.prune(async () => false), true);
  assert.deepEqual(stored, { open: [], save: [] });
  assert.equal(published.length, 1);
});

test("removes URL credentials and does not publish failed persistence as saved history", async () => {
  const entries = historyEntries(
    "local",
    "https://user:password@example.invalid/model.onnx?token=secret#secret",
    "https://user:password@example.invalid/models?token=secret",
  );
  assert.deepEqual(
    entries.map(({ path }) => path),
    ["https://example.invalid/model.onnx", "https://example.invalid/models"],
  );
  let persisted = { open: [], save: [] };
  let fail = true;
  const history = new PathHistory({
    get: () => persisted,
    update: async (_key, value) => {
      if (fail) throw new Error("Storage unavailable");
      persisted = value;
    },
  });
  const snapshots = [];
  history.subscribe((value) => snapshots.push(value));
  await assert.rejects(
    history.update("open", { action: "remember", entries }),
    /Storage unavailable/,
  );
  assert.deepEqual(history.snapshot().open, []);
  assert.equal(snapshots.length, 0);
  fail = false;
  await history.update("open", { action: "remember", entries });
  assert.deepEqual(history.snapshot().open, entries);
});

test("sanitizes authority-less URIs without treating Windows drive paths as schemes", () => {
  for (const [input, expected] of [
    ["provider:/models/a.onnx?token=secret#secret", "provider:/models/a.onnx"],
    ["provider:models/a.onnx?token=secret#secret", "provider:models/a.onnx"],
    ["C:\\models\\a.onnx", "C:\\models\\a.onnx"],
    ["C:/models/a.onnx", "C:/models/a.onnx"],
    ["C:models\\a.onnx", "C:models\\a.onnx"],
    ["/models/a?b#c.onnx", "/models/a?b#c.onnx"],
  ]) {
    assert.equal(historyEntries("local", input)[0].path, expected);
  }
  assert.deepEqual(
    historyEntries(
      "local",
      "provider:/models/a.onnx?token=secret",
      "provider:/models?token=secret",
    ).map(({ path }) => path),
    ["provider:/models/a.onnx", "provider:/models"],
  );
});

test("filters malformed stored histories and isolates returned entry objects", async () => {
  let stored;
  const errors = [];
  const history = new PathHistory(
    {
      get: () => stored,
      update: async (_key, value) => {
        stored = value;
      },
    },
    (message) => errors.push(message),
  );
  for (const value of [null, "old-format", { open: {}, save: false }]) {
    stored = value;
    assert.deepEqual(history.snapshot(), { open: [], save: [] });
  }
  assert.equal(errors.length, 3);
  const entry = { source: "local", kind: "file", path: "/models/valid.onnx" };
  stored = {
    open: [null, {}, { ...entry, path: 42 }, entry],
    save: [entry, { ...entry, source: "unknown" }],
  };
  const snapshot = history.snapshot();
  assert.deepEqual(snapshot, { open: [entry], save: [entry] });
  assert.equal(errors.length, 4);
  snapshot.open[0].path = "/changed.onnx";
  assert.equal(stored.open.at(-1).path, "/models/valid.onnx");
  assert.equal(history.snapshot().open[0].path, "/models/valid.onnx");
  await history.update("open", { action: "remove", entry });
  assert.deepEqual(stored, { open: [], save: [entry] });
});
