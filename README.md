<p align="center">
  <img src="media/logo.png" width="96" alt="ONNX Light Viewer logo">
</p>

# ONNX Light Viewer

[![Uses ONNX](media/onnx-badge.png)](https://onnx.ai/)
[![CI](https://github.com/xadupre/onnx-light-viewer/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/xadupre/onnx-light-viewer/actions/workflows/tests.yml)
[![Code coverage](https://codecov.io/gh/xadupre/onnx-light-viewer/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/onnx-light-viewer)

ONNX Light Viewer explores ONNX models in VS Code, powered by native onnx-light.
The current viewer inspects models without executing them or changing source
files.

## Implemented features

- Open local ONNX files, download and cache public Hugging Face models, or browse
  bundled onnx-light backend test graphs.
- Use **Convert** to run installed mbext, modelbuilder or Mobius CLIs in a trusted
  workspace, with live output, a chosen output directory, a reproduction script
  and automatic model opening. See [exporting models](docs/features/exporting_models.rst).
- Search nodes and operators with live suggestions; zoom, pan, select and move
  nodes, including in large graphs.
- Navigate nested subgraphs and model-local function definitions with breadcrumbs.
- Inspect shapes, attributes, opaque/structured types, dense/sparse/encoded
  initializer metadata and small available tensor values; open operator
  documentation rendered as Markdown.
- Show up to two short attributes/initializer previews inside each operator;
  otherwise show their count, with full details in the inspector.
- Refresh inferred shapes/types and static constant, tag and lifetime metadata,
  with visible update counts.
- Use **Logs**, **Status** and **Controls** for diagnostics and shortcuts, and
  **Repositories** for ONNX ecosystem links.
- Use **Settings** to change the download cache directory and saved sidebar
  preferences, or check the app and compiled onnx-light versions. Python is
  optional: save its executable path, then choose **Refresh package versions**
  in a trusted workspace to inspect installed ecosystem/exporter packages.
  Downloads default to `~/.cache/onnx-light-viewer/downloads`.
  Inspection excludes user-site packages and disables startup hooks
  (`sitecustomize` and executable `.pth` lines). It reads virtualenv metadata,
  path-only `.pth` entries, and `PYTHONPATH` sources without importing model
  libraries or installing anything. Source versions are read statically; dynamic
  versions are shown as unavailable rather than executing package code.

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

## Reusable ONNX badge

Applications using ONNX can copy [the PNG badge](media/onnx-badge.png) into their
repository and add it to their README. PNG images are supported by VSIX packaging:

```markdown
[![Uses ONNX](media/onnx-badge.png)](https://onnx.ai/)
```

The PNG is rendered from [the SVG badge](media/onnx-badge.svg). The SVG is
self-contained and includes a miniature of the
[ONNX logo](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/_static/ONNX-Logo.svg)
(copyright ONNX Project Contributors, Apache-2.0). Keep the attribution embedded
in the SVG when reusing it. The badge indicates ONNX usage, not official
certification or endorsement.

Licensed under [Apache-2.0](LICENSE).
