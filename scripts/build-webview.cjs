const { copyFileSync, mkdirSync } = require("node:fs");
const { resolve, join } = require("node:path");

const output = resolve(__dirname, "../media/vendor");
mkdirSync(output, { recursive: true });
copyFileSync(
  require.resolve("markdown-it/browser"),
  join(output, "markdown-it.min.js"),
);

for (const [name, license] of [
  ["markdown-it", "LICENSE"],
  ["entities", "LICENSE"],
  ["linkify-it", "LICENSE"],
  ["mdurl", "LICENSE"],
  ["punycode.js", "LICENSE-MIT.txt"],
  ["uc.micro", "LICENSE.txt"],
]) {
  copyFileSync(
    resolve(__dirname, "../node_modules", name, license),
    join(output, `${name}.LICENSE`),
  );
}
