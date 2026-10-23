---
name: visual-documentation
description: Use when creating or revising documentation in onnx-light-viewer, especially architecture and design explanations. Keep it concise, visual and grounded in implemented behavior.
---

# Short, visual documentation

This skill applies only to this repository.

1. Start with the reader's question and a short answer. Prefer a compact overview
   to a function-by-function implementation walkthrough.
2. Use real diagrams for boundaries, ownership, data flow, state transitions and
   action effects. Prefer a few focused diagrams over one dense drawing or an
   ASCII diagram inside a code block.
3. Store self-contained SVG diagrams in `docs/_static/` and embed them with Sphinx
   `figure` directives. Include a title, description, meaningful alt text and a
   short caption. Label arrows and distinguish states with text, not color alone.
4. Keep diagrams readable at documentation-page width, with generous spacing,
   short labels and light/dark contrast. No remote fonts, images, scripts or new
   rendering dependencies.
5. Preserve the important contracts: model versus display state, source ownership,
   asynchronous recovery and stale-result/error guards. Verify them against the
   code; do not simplify a diagram into an incorrect behavior claim.
6. Link to source files and focused feature guides for implementation details.
   Do not move an unreadable wall of text into a new appendix by default.
7. Clearly separate implemented, in-review and planned capabilities. Do not
   advertise a library capability as a viewer feature.
8. Add short, runnable code snippets when they help the reader complete a task.
   State prerequisites and expected results; verify the snippets.
9. Build Sphinx with warnings treated as errors, check formatting and SVG XML,
   and inspect rendered diagrams for clipped text, overlaps and unclear arrows.
   Keep the change documentation-only unless runtime changes were requested.
