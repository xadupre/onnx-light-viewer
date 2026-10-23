Application design
==================

Three layers: **VS Code manages sources and panel state**, **C++ retains and
analyzes the model**, and **the webview draws lightweight snapshots**.
There is no Python runtime or model-execution service on the default branch.
Saving explicitly writes a new model copy; it does not modify the source file.
See :doc:`features` for implemented/in-progress capabilities.

Who owns what?
--------------

.. figure:: _static/viewer-architecture.svg
   :alt: The webview sends requests to the extension host. The host calls the
         native addon and returns graph, tensor and metadata snapshots.
   :width: 100%

   Files and native handles stay outside the browser; the browser owns the view.

* Local files are streamed on a native worker, omitting large tensor payloads.
  Virtual sources retain their backing byte buffer; backend cases retain their
  native owner. Keep borrowed buffers unchanged while a handle is alive.
  Small available tensor values can be included in previews.
* Hugging Face downloads enter the persistent cache only after successful native
  loading and atomic publication. Recovery uses the final local cache path.
* Graph projection and analysis are synchronous. Only the selected graph is
  projected, not every descendant.
* Inline Open/Save submit masked password fields, cleared after submission or
  closing their panel. Direct editor opening uses a host password prompt.
  Decryption passwords remain in panel-local reload closures and are never logged.

What does an action change?
---------------------------

.. figure:: _static/viewer-actions.svg
   :alt: Moving and zooming only change the browser view. Shape refresh changes
         the in-memory model and patches tensors; metadata is read-only.
         Opening or navigating replaces the graph view.
   :width: 100%

   Shape and metadata replies preserve the camera, positions and selection.
   A new graph snapshot rebuilds the view.

The protocol uses ``postMessage`` in both directions:

.. list-table::
   :header-rows: 1
   :widths: 32 36 32

   * - Request
     - Reply
     - Meaning
   * - ``open`` / ``navigate``
     - ``graph``
     - Replace the displayed graph.
   * - ``browse-files``
     - ``file-list`` or ``file-list-error``
     - List folders, model files or backend cases inside the sidebar.
   * - ``save``
     - ``save-info`` or ``save-error``
     - Write a plain or encrypted model copy, optionally with external weights.
   * - ``fusion-patterns`` / ``preview-fusion``
     - ``fusion-patterns`` / ``fusion-preview``
     - List native patterns and preview their candidates without editing.
   * - ``apply-fusion`` / ``cancel-fusion``
     - ``graph`` and ``fusion-info`` / no reply
     - Apply selected disjoint candidates or discard the preview.
   * - ``refresh-shapes``
     - ``shape-info``
     - Patch tensor descriptions.
   * - ``metadata``
     - ``metadata-info``
     - Update separate static analysis.
   * - ``ready``
     - Snapshots, history and status
     - Restore the webview without reloading the model.

Failures have dedicated error replies and action-log entries. Shape and metadata
analysis first run ``VerifyModel``; invalid connectivity can still be displayed
without running either analysis. Update counts compare successful snapshots,
not runtime measurements.

What happens after an error?
----------------------------

.. figure:: _static/native-recovery.svg
   :alt: Failed shape or metadata analysis invalidates the handle but keeps the
         display. The next native action reloads the source. A completion is
         applied only if its panel, model and graph request are still current.
   :width: 100%

   The addon stays loaded; only the model handle is replaced.

Concurrent actions can share one recovery promise. Handlers discard stale
results **and errors** after disposal, model replacement or superseding
navigation. A reload can lose prior in-memory inference; local sources must
remain available. Logs retain failures across model changes and webview
recreation. This recovers caught exceptions, not native process crashes.
After fusion, recovery clones the retained edited model instead of reloading
the original source, preserving applied rewrites.

Contracts worth keeping
------------------------

* ``NativeModel.graph(inferShapes?, path?)`` projects one graph; requesting
  inference updates the enclosing model first. ``metadata(path?)`` returns a
  separate, read-only analysis snapshot. ``serialize`` and ``serializeFiles``
  save the retained model as a new plain, encrypted or external-weight copy.
* Fusion materializes weights in an independent native snapshot. Selected
  patterns apply together against original node indices; overlapping candidates
  are rejected. Graph replacement or cancellation invalidates previews.
  Replacement previews share one trial builder per pattern without rewriting
  the matched graph. External-weight reuse indexes candidates by name, element
  type and dimensions before comparing their payloads and checking backing files.
* ``GraphPath`` uses validated indices, not potentially duplicate names.
  Functions use their own imports and formal names, not the caller's weights;
  nested graph attributes can capture enclosing values.
* Tensor dimensions remain strings. ``[]`` means scalar, ``null`` means unknown
  rank; non-tensor types can omit the shape.
* Model text arrives through messages, not HTML interpolation. Operator docs
  use bundled Markdown rendering with raw HTML disabled, HTTP(S)-only links and
  no remote image fetching.
* Large graphs attach only visible SVG details. **Fit** uses batched paths;
  search indexes all nodes, including those without a current DOM element.
  Moving nodes updates only their groups and incident edges.

Where to change things
----------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Source
     - Responsibility
   * - :download:`src/extension.ts <../src/extension.ts>`
     - Commands, sources, panel state, routing and recovery.
   * - :download:`src/graph.ts <../src/graph.ts>`
     - Typed snapshots, native model API and addon loading.
   * - :download:`src/huggingface.ts <../src/huggingface.ts>`
     - URL normalization, downloads and persistent cache.
   * - :download:`src/log.ts <../src/log.ts>`
     - Correlated action history and status.
   * - :download:`native/addon.cc <../native/addon.cc>`
     - Ownership, bounded parsing, graph projection and native algorithms.
   * - :download:`src/webview.ts <../src/webview.ts>`
     - HTML shell, local resources and content security policy.
   * - :download:`media/viewer.js <../media/viewer.js>` /
       :download:`viewer.css <../media/viewer.css>`
     - SVG layout, gestures, search, inspector and presentation.

``CMakeLists.txt`` pins and checksum-verifies the published onnx-light libraries;
only the addon is compiled locally. Its platform must match the extension host.
Tests mirror the boundaries: native graph tests, host routing tests and webview
DOM/SVG tests. Build commands and validation details are in :doc:`development`.
