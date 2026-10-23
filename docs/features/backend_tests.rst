Backend tests
=============

Backend tests are ONNX model cases used by onnx-light to check operator and
runtime behavior. They include single-operator cases and multi-node models,
including large Qwen3 cases.

Choose **Open ONNX model**, then **Backend test**, to open a searchable catalog
from the bundled native onnx-light release. Search by test name, model name or
category, then press Enter or click a result. The chosen graph opens in the viewer
with normal inspection, moving, subgraph navigation and shape refresh. Escape
cancels the selector.

The viewer opens the test model; it does not run the test, compare reference
outputs, report pass/fail or benchmark it. No Python or network access is needed
to browse these cases, and external weight files are not downloaded or opened.
