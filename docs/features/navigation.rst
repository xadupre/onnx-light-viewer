Zoom, navigation and moving nodes
=================================

.. image:: ../_static/navigation.png
   :alt: Interactive graph with a selected Relu node and the minus, Fit and plus zoom controls.
   :width: 480px

The **Controls** tab groups keyboard shortcuts, mouse gestures and navigation
buttons into two-column tables: **Control** and **Action**. It is available even
before a model is loaded.

* **+**, **-** or Ctrl+wheel zoom without rearranging nodes. Ctrl+wheel zooms
  around the pointer, both in and out. **Fit** recenters the view and includes
  nodes moved outside the original layout.
* Click **Zoom area**, then drag a rectangle over any part of the graph to
  magnify it. The button stays filled, outlined and checked while the mode is
  armed, even after focus moves to the graph. Alt+drag also works, including over
  nodes and subgraph links, without opening the subgraph. The mode ends after one
  drag; Escape cancels it. Selection and node positions are preserved. Area zoom
  uses the same minimum zoom as the buttons, even when the drag ends outside the
  canvas.
* The **Zoom to readable size** icon beside **Fit** jumps directly to 100%,
  centered on the inspected node, without repeated **+** clicks.
  Large graphs omit unreadable text in overview and only paint nodes and
  connections intersecting the viewport, with a small margin for panning.
* With focus in the graph or its zoom controls, **F** fits the graph, **R**
  jumps to readable size, and **+** / **-** zoom in / out.
* Graphs with more than 1,000 nodes open at readable size. Only nearby SVG
  elements are created and attached; **Fit** uses two batched paths rather than
  thousands of interactive elements. Long connections crossing the viewport
  are batched when their labels are offscreen.
* **Find node or operator** searches the whole graph, including offscreen nodes.
  Press Enter or **Go** to select and center the next match at readable size.
* Wheel scrolling moves the view vertically; Shift+wheel moves it horizontally.
  Horizontal trackpad scrolling is supported.
* Drag an operator, input or output to move it. Connections and tensor labels
  follow, including while zoomed. Drag a selected node to move the selection.
* Ctrl, Cmd or Shift+click toggles node selection. Drag a rectangle on the
  background to select a group; hold a selection modifier to add to it.
  Click the background to clear selection.
* Ctrl+A (Cmd+A on macOS) selects all nodes when the graph has focus.
  Enter or Space selects a focused node. Escape cancels a drag/selection
  rectangle, or clears selection when no gesture is active.
* **Refresh positions** restores the top-to-bottom layout with new random
  horizontal operator positions and fits the view, preserving inferred shapes.
  Resizing the editor preserves node positions and zoom.
