Backend tests
=============

Backend tests are ONNX model cases used by onnx-light to check operator and
runtime behavior. They include single-operator cases and multi-node models,
including large Qwen3 cases.

Choose **Open**, then **Backend test**, to show a searchable catalog inside the
right sidebar, from the bundled native onnx-light release. Filter by test name,
model name or category, then double-click a result to open it immediately.
A single click only selects it; **Open model** also opens the selection.
The chosen graph opens with normal inspection, moving, subgraph navigation and
shape refresh. Click **Open** again or **Close** to hide the selector.

The viewer opens the test model; it does not run the test, compare reference
outputs, report pass/fail or benchmark it. No Python or network access is needed
to browse these cases, and external weight files are not downloaded or opened.
