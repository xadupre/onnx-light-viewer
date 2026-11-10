Changelogs
==========

Notable changes to ONNX Light Viewer, newest first.

0.3.0 (unreleased)
------------------

- Fix release packaging by pinning VS Code type definitions to the minimum
  supported VS Code version
  (`#366 <https://github.com/xadupre/onnx-light-viewer/pull/366>`_).
- Show original affected and replacement node and initializer counts in applied
  and pending rewriting steps, including initializer-only cleanup, while
  highlighting surviving results
  (`#359 <https://github.com/xadupre/onnx-light-viewer/pull/359>`_).
- Use onnx-light 0.2.7 and onnx-light-cpu 0.2.7, and regenerate quantized
  inspection examples with supported INT4 and INT8 formats
  (`#358 <https://github.com/xadupre/onnx-light-viewer/pull/358>`_).
- Route short skip connections inside the graph where possible, and show
  arrowheads in large-graph overviews
  (`#357 <https://github.com/xadupre/onnx-light-viewer/pull/357>`_).
- Move the **Patterns** tab between **Tensors** and **Run**
  (`#354 <https://github.com/xadupre/onnx-light-viewer/pull/354>`_).
- Highlight compute-heavy and custom-domain operators, including in zoomed-out
  views, with a collapsible color legend
  (`#352 <https://github.com/xadupre/onnx-light-viewer/pull/352>`_).
- Add **Global Rewriting** actions for IdentityRemoval, RepeatedLayer and
  PartitionGroup, with undo, replay and save support
  (`#348 <https://github.com/xadupre/onnx-light-viewer/pull/348>`_).
- Edit rewritings one step at a time, append matches, and create or update saved
  plans. Preserve the selected rewriting across **Unfuse**, **Fuse** and
  **Optimize** (`#347 <https://github.com/xadupre/onnx-light-viewer/pull/347>`_,
  `#344 <https://github.com/xadupre/onnx-light-viewer/pull/344>`_).
- Sort columns and navigate sections in global optimization reports
  (`#346 <https://github.com/xadupre/onnx-light-viewer/pull/346>`_).
- Simplify **Save** to one format-aware output path and a VS Code file or folder
  chooser (`#343 <https://github.com/xadupre/onnx-light-viewer/pull/343>`_).
- Select saved rewriting alternatives beside the graph name, with **None** as
  the default original graph, and inspect their steps and expandable
  before/after operator details in **Rewritings**
  (`#338 <https://github.com/xadupre/onnx-light-viewer/pull/338>`_).

0.2.4 (2026-11-06)
------------------

- Put the graph name, node/initializer counts and pending rewritings on one
  line. Add a graph-name menu for switching between functions and subgraphs,
  remove Navigation/Application headings, and keep version details in Status
  (`#336 <https://github.com/xadupre/onnx-light-viewer/pull/336>`_).
- Preserve the configured Python interpreter in source-PYTHONPATH integration
  tests (`#328 <https://github.com/xadupre/onnx-light-viewer/pull/328>`_).
- Add this changelog to the repository and documentation
  (`#335 <https://github.com/xadupre/onnx-light-viewer/pull/335>`_).
- Update Node.js/VS Code development types and the documentation theme
  (`#331 <https://github.com/xadupre/onnx-light-viewer/pull/331>`_,
  `#332 <https://github.com/xadupre/onnx-light-viewer/pull/332>`_,
  `#333 <https://github.com/xadupre/onnx-light-viewer/pull/333>`_).

- Load/save ORT FlatBuffer models and ZIP/ZIP64 `.ortpackage` archives or package
  directories.
  Infer the Open format automatically, offer separate file/folder pickers, and
  disable inapplicable Save choices. Preserve manifests, variants, shared assets,
  GenAI configuration/tokenizers and compiled EPContext assets; resolve shared
  GenAI filenames without modifying the original package. Package semantics
  remain viewer-owned; contained model parsing/serialization uses onnx-light
  (`#329 <https://github.com/xadupre/onnx-light-viewer/pull/329>`_).

0.2.3 (2026-11-04)
------------------

- First public release of the ONNX graph viewer for VS Code, powered by native
  onnx-light.
