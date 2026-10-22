Native model loading
====================

The extension host loads a Node-API 8 C++ binding linked to onnx-light's
``onnx_proto``, ``onnx_core``, ``onnx_op``, ``onnx_shape`` and
``onnx_backend_test`` libraries (the latter also links native kernels). It does
not use the npm ``onnx-proto`` package, protobufjs, upstream ``onnx``, a Python
process, or a custom ONNX decoder.

Published library
-----------------

The first build downloads the already published
`onnx-light 0.1.35 C++ release
<https://github.com/xadupre/onnx-light/releases/tag/0.1.35>`_,
selecting the archive for the current platform and verifying its SHA-256
checksum. Only the small Node-API binding is compiled; onnx-light is not rebuilt.
There is no fallback to compiling onnx-light from source.

The archive is cached under ``build/native/`` and reused on subsequent builds.
The platform asset names and checksums are pinned in ``CMakeLists.txt``.

Operator documentation and attribute defaults come from the released
``LightOpSchema`` records. Lookup matches the operator domain and selects the
newest schema whose ``since_version`` does not exceed the model's imported
opset. Missing schemas or documentation simply disable the **Doc** button.
No external documentation service is used.

The refresh action registers the released ``onnx_shapes`` algorithms and calls
``core::shapes::InferShapesModel``. See :doc:`shape_information` for the
displayed metadata, refresh behavior and inference limitations.

The backend-test selector calls ``core::backend_test::CollectTestCases("", true)``
for both listing and loading, including large cases such as Qwen3.
Listing reads case metadata without materializing models or datasets. Only the
chosen case is materialized, with reference-output generation disabled, and its
native owner is retained alongside the model so borrowed buffers remain valid.
Cases come from the released C++ library, not Python backend tests or downloads.

The retained native model also exposes a read-only metadata snapshot. It uses
``core::compute::InferConstants``, ``InferValueAndNodeTags`` and
``ComputeResultLifetimeInfo`` from the published library, without shape
inference or tensor evaluation. Lifetime analysis includes nested-subgraph
captures and preserves graph outputs and caller-owned inputs/initializers.
Release lists are exposed as arrays of names, not parsed from semicolon-delimited
metadata strings. Stored ``metadata_props`` remain separate from computed
classifications, and arbitrary keys and values are displayed as plain text.

Extension host compatibility
----------------------------

The compiled binding is written to ``dist/native/onnx_light_viewer.node``.
It must match the extension host's operating system and architecture, rather
than necessarily those of the desktop running VS Code. For example, Remote WSL
uses the Linux binding.

Node-API allows the binding to be loaded by Node.js and VS Code without an
Electron-specific rebuild. The native binding and dependency licenses are
included when packaging the extension.
