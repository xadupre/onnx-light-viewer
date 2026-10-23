Action logs and application status
==================================

The viewer has **Model**, **Logs** and **Status** tabs. Switching tabs preserves
the graph, selection, positions and zoom. Logs record operations, not clicks:
tab switching, selecting nodes, zooming and clicking empty space or the log
panel do not add entries. **Refresh positions**, loading, navigation,
**Refresh shapes** and **Refresh metadata** show **Running**, then **OK** or
**Failed** with full error details. Cancelled file/backend pickers show
**Cancelled**.

Each viewer retains its history when another model is loaded or its webview is
recreated. Initial loading failures also appear there. The
**ONNX Light Viewer: Show Logs** command selects the viewer's **Logs** tab,
or opens an empty viewer if none exists.

Shape and metadata algorithms first call onnx-light's structural ``VerifyModel``.
Invalid graphs can still be inspected, but verification failures prevent these
algorithms from running. Errors identify the failing native algorithm.
After a failed refresh, the native handle is discarded and recreated before
the next native action: local files (including Hugging Face cache entries) are
streamed again, virtual files use retained bytes, and backend cases are rebuilt.
The previous display remains visible; inferred information may need refreshing
after this reload. The source file is never modified, but local files must remain
available for recovery.

**Status** shows the loaded native library and its version, the current graph,
native model handle, running operations and the last failed action with its
complete error. A caught C++ exception is converted into a JavaScript error;
it does not unload onnx-light. After failed analysis, the handle is marked
discarded until the next native action recreates it. The status then returns
to **Ready**, while retaining the last failure for inspection.

This recovery applies to caught exceptions, not native process crashes.
If the addon cannot be imported at all, VS Code reports an extension activation
failure; the viewer cannot open to display its Status tab.
