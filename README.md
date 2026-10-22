<p align="center">
  <img src="media/logo.png" width="96" alt="ONNX Light Viewer logo">
</p>

# ONNX Light Viewer

ONNX Light Viewer is an interactive environment for exploring, optimizing and
running ONNX models in VS Code, powered by native onnx-light. Open local models
or download and cache them from Hugging Face; navigate graphs and subgraphs,
inspect operators and parameters, and refresh shapes and metadata with visible
update counts. Inspect tensor statistics, histograms and color maps, including
quantized tensors and low-bit encoded values, with support for OpaqueProto,
StructProto and EncodedValueProto.

Apply fusion patterns, including Layer patterns, quantize selected weights, and
step through rewriting scenarios and pipelines. Save transformed models with
embedded or separate weights, and encrypt saved protos or decrypt them on load.
Run and benchmark whole models or selected subgraphs using onnx-light,
onnx-light-cpu or ONNX Runtime on CPU or CUDA, compare execution engines, and
export inputs and outputs with a Python script to reproduce runs locally.
An action log records operations, timing, status and errors.

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
Choose **Open ONNX model** to load your own file, or **Hugging Face** to download
and cache a public ONNX file by URL. After editing the application, close the
development window and run `./start.sh` again.

## Documentation

Sphinx documentation: [getting started](docs/getting_started.rst),
[implemented features and viewer controls](docs/features.rst),
[application design and source guide](docs/design.rst), and
[development, tests and documentation build](docs/development.rst).

Licensed under [Apache-2.0](LICENSE).
