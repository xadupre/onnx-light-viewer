ONNX Light Viewer
=================

ONNX Light Viewer is a lightweight, interactive ONNX graph viewer for VS Code.
Explore model graphs and nested subgraphs, inspect operators, tensor shapes and
parameters, and refresh shape inference and metadata analysis. Models are decoded
locally with the native onnx-light C++ library; no model data is uploaded, and
source files are not modified.

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
