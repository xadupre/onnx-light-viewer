const { execFileSync } = require("node:child_process");
const { availableParallelism } = require("node:os");

execFileSync(
  process.execPath,
  [
    require.resolve("cmake-js/bin/cmake-js"),
    "build",
    "--out",
    "build/native",
    "--parallel",
    String(availableParallelism()),
  ],
  { stdio: "inherit" },
);
