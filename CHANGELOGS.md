Changelogs
==========

Notable changes to ONNX Light Viewer, newest first.

0.2.4 (unreleased)
------------------

- Select saved rewriting alternatives beside the graph name, with **None** as
  the default original graph. Show their steps and fusion undo controls together
  in **Fuse / Unfuse**.
- Separate original fusion impact from surviving result nodes; show initializer
  changes and expandable before/after operator details.
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
