#!/usr/bin/env bash
# Installs the latest onnx-light-viewer release matching this machine into VS Code.
set -euo pipefail

REPO="xadupre/onnx-light-viewer"
CODE="${CODE:-code}"

case "$(uname -s)" in
  Linux) platform=linux ;;
  Darwin) platform=darwin ;;
  MINGW* | MSYS* | CYGWIN*) platform=win32 ;;
  *) echo "Unsupported operating system: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64 | amd64) arch=x64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

asset="onnx-light-viewer-${platform}-${arch}.vsix"
url="https://github.com/${REPO}/releases/latest/download/${asset}"

if ! command -v "$CODE" > /dev/null 2>&1; then
  echo "VS Code CLI '$CODE' not found; set CODE to its path (e.g. CODE=code-insiders)." >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Downloading $url"
if ! curl -fL --retry 3 -o "$tmpdir/$asset" "$url"; then
  echo "No $asset in the latest release of $REPO." >&2
  exit 1
fi

"$CODE" --install-extension "$tmpdir/$asset" --force
