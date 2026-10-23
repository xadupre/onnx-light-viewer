---
name: onnx-helper-alias
description: Use when writing or updating Python ONNX model generators, examples, or tests to import the helper module as oh.
---

# ONNX helper alias

Import the ONNX helper module as `oh`, not `helper`:

```python
import onnx.helper as oh
from onnx import TensorProto

node = oh.make_node("Relu", ["x"], ["y"])
```

Preserve the existing ONNX implementation. This repository's generators use
`onnx_light.onnx`, so use its equivalent import rather than switching dependencies:

```python
import onnx_light.onnx.helper as oh
from onnx_light.onnx import TensorProto
```

Remove `helper` from grouped imports and update every corresponding `helper.*`
reference to `oh.*`. Leave unrelated helpers and model-generation behavior unchanged.
Run the affected generators when their dependencies are available, and do not commit
regenerated model files for an alias-only change.
