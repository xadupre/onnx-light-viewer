Implemented features
====================

The viewer inspects ONNX graphs locally with onnx-light. Models are not uploaded
or executed, and viewer actions do not save or modify the source file.

Available in the viewer
-----------------------

This checklist describes the default branch, not every capability of the
underlying onnx-light libraries. Start with the worked example in
:doc:`getting_started`; the linked guides explain each control in detail.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Area
     - Implemented
     - Guide / boundary
   * - Opening models
     - Local ``.onnx`` and ``.pipe-onnx`` files, bundled demos, public Hugging Face downloads and
       persistent caching.
     - :doc:`features/opening_models`. No authentication or automatic external
       weight downloads.
   * - Backend catalog
     - Search bundled onnx-light cases and inspect their model graphs.
     - :doc:`features/backend_tests`. Opens graphs, not a test runner.
   * - Converting models
     - Run installed mbext, modelbuilder or Mobius exporters with visible commands,
       live output, cancellation and a saved reproduction script; open their output.
     - :doc:`features/exporting_models`. Requires Python and a trusted workspace;
       exporter packages and checkpoints are not installed automatically.
   * - Navigation
     - Zoom, pan, area zoom, readable-size views, multi-selection, dragging and
       sidebar node/operator search with live suggestions.
     - :doc:`features/navigation`. Moving a node changes only its display position.
   * - Inspection
     - Types, symbolic shapes, attributes/schema defaults, dense/sparse/encoded
       initializer metadata, opaque/structured types, bounded previews and
       Markdown operator documentation.
     - :doc:`features/inspection`. External or omitted payloads are not read for
       previews; this is not a full tensor-data browser.
   * - Subgraphs and functions
     - Graph/list attributes, captured inputs, model-local function definitions,
       nested navigation and breadcrumbs.
     - :doc:`features/subgraphs`. Function bodies are definitions, not specialized
       executions of the selected call.
   * - PipelineProto
     - Stage navigation, generation-settings previews, per-stage inference and
       fusion, and whole-container saving with embedded or external weights.
     - :doc:`features/opening_models`. No pipeline execution or encryption.
   * - Shape/type inference
     - Explicit **Refresh shapes**, visible update counts and refreshed labels
       without resetting the camera or selection.
     - :doc:`features/inspection`. In-memory only; unsupported operators may fail.
   * - Static metadata
     - Constants, tags, lifetimes, last reads, releasable/unused outputs and stored
       model/graph/node/tensor metadata.
     - :doc:`features/metadata`. Static analysis, not measured runtime memory usage.
   * - Memory profiling
     - Dynamic dimension inputs, estimated CPU memory curve and paginated
       per-node allocation/scratch events.
     - :doc:`features/memory_profiling`. Native estimation, not model execution
       or measured RAM/GPU usage.
   * - Diagnostics and help
     - Action history, progress/errors, native-library status, recovery after
       caught exceptions, keyboard/mouse controls and ONNX repository links.
     - :doc:`features/action_logs`, :doc:`features/navigation` and :doc:`links`.

Not yet available
-----------------

The following are not implemented features of the default branch. A capability
in onnx-light, onnx-light-cpu or ONNX Runtime does not imply a corresponding
viewer control.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Capability
     - Status
   * - Model saving and encrypted ModelProto I/O
     - In progress in `PR #115
       <https://github.com/xadupre/onnx-light-viewer/pull/115>`_, including embedded
       and plain external-weight export. Not available on ``main`` yet.
   * - Graph transformations
     - Planned: fusion/Layer patterns, weight quantization and rewriting
       scenarios/pipelines.
   * - Full tensor analytics
     - Planned: statistics, histograms, color maps and decoded payload browsing.
       Opaque/structured type and encoded layout previews are already available.
   * - Execution and benchmarking
     - Planned: whole-model/subgraph execution, CPU/CUDA engine comparison and
       input/output export with reproducible execution scripts.

Detailed guides
---------------

.. toctree::
   :maxdepth: 1

   features/opening_models
   features/exporting_models
   features/navigation
   features/inspection
   features/subgraphs
   features/metadata
   features/memory_profiling
   features/fusion_patterns
   features/backend_tests
   features/action_logs
