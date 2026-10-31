const assert = require("node:assert/strict");
const { test } = require("node:test");
const { genaiSessionOptions } = require("../media/prompt-settings.cjs");

test("multi-provider overlay retains selected options and removes deselected providers before SDK parsing", () => {
  const json = JSON.stringify({
    provider_options: [
      { cuda: { device_id: "7", gpu_mem_limit: "1048576" } },
      { dml: { device_id: "3" } },
      { ViewerUnsupportedProvider: {} },
    ],
    "session.intra_op.allow_spinning": "0",
    "session.disable_cpu_ep_fallback": "1",
    intra_op_num_threads: 9,
  });
  const { options, providers } = genaiSessionOptions({
    providers: ["cpu", "cuda", "dml"],
    cudaDeviceId: 1,
    dmlDeviceId: 2,
    gpuMemoryMiB: 64,
    intraOpThreads: 4,
    sessionOptions: json,
  });
  assert.deepEqual(providers, ["cpu", "cuda", "dml"]);
  assert.deepEqual(options.provider_options, [
    { cuda: { device_id: "7", gpu_mem_limit: "1048576" } },
    { dml: { device_id: "3" } },
  ]);
  assert.equal(options.intra_op_num_threads, 4);
  assert.equal(options["session.intra_op.allow_spinning"], "0");
  assert.equal(options["session.disable_cpu_ep_fallback"], "0");
  assert.equal(JSON.parse(json).provider_options.length, 3);
  assert.deepEqual(
    genaiSessionOptions({ providers: ["cpu"], sessionOptions: json }).options
      .provider_options,
    [],
  );
});

test("legacy provider selection and full JSON/model provider escape hatch remain supported", () => {
  assert.deepEqual(genaiSessionOptions({ provider: "cuda" }).providers, [
    "cuda",
  ]);
  assert.deepEqual(genaiSessionOptions({ provider: "cpu" }).providers, ["cpu"]);
  const sessionOptions =
    '{"provider_options":[{"CPU":{}},{"cuda":{"device_id":"2"}}],"session.disable_cpu_ep_fallback":"1"}';
  const overlay = genaiSessionOptions({ sessionOptions });
  assert.deepEqual(overlay.providers, ["CPU", "cuda"]);
  assert.deepEqual(overlay.options, JSON.parse(sessionOptions));
  assert.equal(genaiSessionOptions({}).providers, undefined);
});
