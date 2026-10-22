Zoom, navigation and moving nodes
=================================

.. image:: ../_static/navigation.png
   :alt: Interactive graph with a selected Relu node and the minus, Fit and plus zoom controls.
   :width: 480px

* **+**, **-** or Ctrl+wheel zoom without rearranging nodes. **Fit** recenters
  the view and includes nodes moved outside the original layout.
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
