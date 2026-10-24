const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

const workflows = path.resolve(__dirname, "..", ".github", "workflows");
const documentation = readFileSync(
  path.join(workflows, "documentation.yml"),
  "utf8",
);

test("documentation runs weekly or manually, not on pushes or pull requests", () => {
  assert.match(documentation, /cron: "0 3 \* \* 1"/);
  assert.match(documentation, /^  workflow_dispatch:/m);
  assert.doesNotMatch(documentation, /^  (push|pull_request):/m);
  assert.match(
    documentation,
    /python -m sphinx -n -W --keep-going -b html docs build\/docs\/html/,
  );
});

test("regular CI does not build documentation", () => {
  const tests = readFileSync(path.join(workflows, "tests.yml"), "utf8");
  assert.doesNotMatch(tests, /^  docs:|sphinx|docs\/requirements\.txt/m);
  assert.match(tests, /npm test/);
});

test("documentation failures open an issue with a link to the failed run", () => {
  assert.match(documentation, /needs: docs/);
  assert.match(
    documentation,
    /if: \$\{\{ always\(\) && needs\.docs\.result == 'failure' \}\}/,
  );
  assert.match(documentation, /permissions:\n      issues: write/);
  assert.match(documentation, /GH_TOKEN: \$\{\{ github\.token \}\}/);
  assert.match(documentation, /GH_REPO: \$\{\{ github\.repository \}\}/);
  assert.match(documentation, /\/actions\/runs\/\$\{\{ github\.run_id \}\}/);
  assert.match(
    documentation,
    /gh issue create --title "Documentation build failed"/,
  );
  assert.match(documentation, /--body ".*\$RUN_URL"/);
});
