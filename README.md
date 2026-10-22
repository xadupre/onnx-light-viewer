<p align="center">
  <img src="media/logo.svg" width="96" alt="ONNX Light Viewer logo">
</p>

# ONNX Light Viewer

ONNX Light Viewer is a lightweight, interactive ONNX graph viewer for VS Code.
Explore model graphs and nested subgraphs, inspect operators, tensor shapes and
parameters, and refresh shape inference and metadata analysis. Models are decoded
locally with the native onnx-light C++ library; no model data is uploaded, and
source files are not modified.

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
Choose **Open ONNX model** to load your own file. After editing the application,
close the development window and run `./start.sh` again.

## Documentation

Sphinx documentation: [getting started](docs/getting_started.rst),
[implemented features and viewer controls](docs/features.rst),
[application design and source guide](docs/design.rst), and
[development, tests and documentation build](docs/development.rst).

Licensed under [Apache-2.0](LICENSE).
