---
name: onnx-light-boundary
description: Use when integrating onnx-light or addressing a missing capability, incorrect behavior, or review finding involving the native SDK.
---

# Keep SDK responsibilities in onnx-light

onnx-light-viewer must never compensate for a defect or missing capability in
onnx-light. Fix the owning library, not its consumer.

1. Use the supported SDK APIs. Do not add per-operator inference wrappers,
   duplicate memory/scratch formulas, SDK arithmetic validators, fallback
   implementations, or feature restrictions to work around SDK behavior.
2. Search existing issues in `xadupre/onnx-light`. Create or update an issue
   there with the affected SDK version, reproduction, expected behavior and
   originating viewer PR/comment. Never file an SDK defect in the viewer.
3. Put SDK regressions and fixes in onnx-light's native tests and implementation.
   Do not silently patch or vendor the SDK inside this repository.
4. Link the upstream issue from affected review threads. Explicitly distinguish
   a defect deferred upstream from one fixed; do not claim that filing an issue
   fixes the SDK or that the viewer is protected while awaiting that fix.
5. Integrate a corrected SDK release when available. Do not restore local
   workarounds while waiting.
6. Keep actual viewer responsibilities here: UI, model ownership, asynchronous
   lifecycle, API argument/transport validation, JavaScript exact-number limits,
   and transparent propagation of SDK errors. These are not permission to
   reimplement SDK validation or operator semantics.
