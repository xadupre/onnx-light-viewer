const {
  copyFileSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} = require("node:fs");
const { resolve, join, dirname } = require("node:path");

const output = resolve(__dirname, "../media/vendor");
mkdirSync(output, { recursive: true });
copyFileSync(
  require.resolve("markdown-it/browser"),
  join(output, "markdown-it.min.js"),
);
const dagre = dirname(require.resolve("@dagrejs/dagre/package.json"));
writeFileSync(
  join(output, "dagre.min.js"),
  `${readFileSync(join(dagre, "dist", "dagre.min.js"), "utf8")}\nglobalThis.dagre = dagre;\n`,
);
copyFileSync(
  join(dagre, "dist", "dagre.min.js.LEGAL.txt"),
  join(output, "dagre.min.js.LEGAL.txt"),
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
