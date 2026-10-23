Zoom, navigation and moving nodes
=================================

.. image:: ../_static/navigation.png
   :alt: Interactive graph with a selected Relu node and the minus, Fit and plus zoom controls.
   :width: 480px

The **Controls** tab lists the keyboard shortcuts, mouse gestures and navigation
buttons in the app, even before a model is loaded.

* **+**, **-** or Ctrl+wheel zoom without rearranging nodes. Ctrl+wheel zooms
  around the pointer, both in and out. **Fit** recenters the view and includes
  nodes moved outside the original layout.
* Click **Zoom area**, then drag a rectangle over any part of the graph to
  magnify it. Alt+drag also works, including over nodes and subgraph links,
  without opening the subgraph. The mode ends after one drag; Escape cancels
  it. Selection and node positions are preserved.
* The **Zoom to readable size** icon beside **Fit** jumps directly to 100%,
  centered on the inspected node, without repeated **+** clicks.
  Large graphs omit unreadable text in overview and only paint nodes and
  connections intersecting the viewport, with a small margin for panning.
* With focus in the graph or its zoom controls, **F** fits the graph, **R**
  jumps to readable size, and **+** / **-** zoom in / out.
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
