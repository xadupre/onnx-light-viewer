<p align="center">
  <img src="media/logo.png" width="96" alt="ONNX Light Viewer logo">
</p>

# ONNX Light Viewer

[![Code coverage](https://codecov.io/gh/xadupre/onnx-light-viewer/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/onnx-light-viewer)

ONNX Light Viewer explores ONNX models in VS Code, powered by native onnx-light.
The current viewer inspects models without executing them or changing source
files.

## Implemented features

- Open local ONNX files, download and cache public Hugging Face models, or browse
  bundled onnx-light backend test graphs.
- Search nodes and operators with live suggestions; zoom, pan, select and move
  nodes, including in large graphs.
- Navigate nested subgraphs and model-local function definitions with breadcrumbs.
- Inspect shapes, attributes, dense/sparse initializer metadata and small available
  tensor values; open operator documentation rendered as Markdown.
- Refresh inferred shapes/types and static constant, tag and lifetime metadata,
  with visible update counts.
- Use **Logs**, **Status** and **Controls** for diagnostics and shortcuts, and
  **Repositories** for ONNX ecosystem links.

See the [feature guide](docs/features.rst) for controls, examples and limitations.
Model saving and encrypted I/O are under review in [#115](https://github.com/xadupre/onnx-light-viewer/pull/115),
not yet available on `main`. Graph rewriting, weight quantization, full tensor
analytics, execution and benchmarking are planned, not current viewer features.

<img src="docs/_static/viewer-snapshot.png" width="75%" alt="ONNX Light Viewer showing a graph with symbolic tensor shapes and the Reshape node's parameters">

## Quick start

Requires Node.js 22+, npm, VS Code 1.96+ with `code` on `PATH`, CMake 3.24+,
a C++20 compiler, and OpenSSL development packages.

From a Bash terminal in this checkout:

```sh
./start.sh
```

`start.sh` installs dependencies when needed, builds the extension, and opens a
new VS Code development window with the small Relu model already displayed.
Choose **Open ONNX model**, then **Local file** to load your own file or
**Hugging Face** to enter a repository ID, repository URL or ONNX file URL and
download and cache the selected file. After editing the application, close the
development window and run `./start.sh` again.

For a guided first inspection, including a Python snippet that creates a small
ONNX model with symbolic shapes and a weight initializer, see
[getting started](docs/getting_started.rst). Python is optional, not a viewer
dependency.

## Documentation

Use **Fuse** in the right sidebar on the main graph. Check one or more patterns,
click **Run Matching**, then check candidates and click **Fuse selected**.
Clicking a candidate highlights its nodes. **Open** and **Save** use mutually
exclusive inline file selectors; **Save** writes a separate copy with embedded
weights, fresh external files, or reused unchanged external weights.
See the [fusion walkthrough](docs/features/fusion_patterns.rst).

Sphinx documentation: [getting started](docs/getting_started.rst),
[implemented features and viewer controls](docs/features.rst),
[application design and source guide](docs/design.rst), and
[development, tests and documentation build](docs/development.rst).

Licensed under [Apache-2.0](LICENSE).
