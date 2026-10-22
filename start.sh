#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROOT"

for tool in node npm cmake code; do
    if ! command -v "$tool" > /dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$tool" >&2
        exit 1
    fi
done

CODE=""
while IFS= read -r candidate; do
    if [[ "$candidate" != */remote-cli/* ]]; then
        CODE="$candidate"
        break
    fi
done < <(type -a -p code)

if [[ -z "$CODE" ]]; then
    printf '%s\n' \
        'The remote VS Code CLI cannot launch an extension development window.' \
        'Add the desktop VS Code bin directory to PATH, or run from a local terminal.' >&2
    exit 1
fi

if [[ ! -d node_modules || ! -f node_modules/cmake-js/bin/cmake-js ||
      ! -f node_modules/node-addon-api/napi.h ]]; then
    npm ci
fi
npm run compile

exec env -u VSCODE_IPC_HOOK_CLI "$CODE" \
    --new-window --disable-extensions --extensionDevelopmentPath="$ROOT" \
    "$ROOT/.vscode/development.code-workspace" "$ROOT/test/models/relu.onnx"
