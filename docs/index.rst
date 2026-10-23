ONNX Light Viewer
=================

.. image:: https://codecov.io/gh/xadupre/onnx-light-viewer/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/xadupre/onnx-light-viewer
   :alt: Code coverage

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

.. figure:: _static/viewer-snapshot.png
   :alt: ONNX Light Viewer showing a graph with symbolic tensor shapes and the Reshape node's parameters.
   :width: 75%

   Explore tensor shapes and inspect the selected node's parameters.

.. toctree::
   :maxdepth: 1

   getting_started
   features
   design
   development
   links
