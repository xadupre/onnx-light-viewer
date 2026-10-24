const assert = require("node:assert/strict");
const { test } = require("node:test");
const {
  tensorVisits,
  encodedVisits,
} = require("../build/native/Release/test_model_storage.node");

test("collects shared type/codec tensors once by identity, retaining first-visit order and layout classification", () => {
  for (let i = 0; i < 2; i++) {
    const result = tensorVisits();
    assert.deepEqual(result.visits, [0, 1]);
    assert.deepEqual(result.layout, [true, true]);
    assert.equal(result.layoutCount, 2);
  }
});

test("collects encoded payloads and their affine constants once through shared graphs", () => {
  assert.deepEqual(encodedVisits(), {
    encodedCount: 2,
    tensorCount: 2,
    firstVisit: true,
    secondVisit: true,
  });
});
