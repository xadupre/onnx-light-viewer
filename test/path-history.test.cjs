const assert = require("node:assert/strict");
const { test } = require("node:test");
const { PathHistory, historyEntries } = require("../dist/pathHistory");

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
    ["/models/second.onnx", "/models", "/models/first.onnx"],
  );
  await history.update("open", { action: "remember", entries: first });
  assert.deepEqual(
    history.snapshot().open.map(({ path }) => path),
    ["/models/first.onnx", "/models", "/models/second.onnx"],
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
