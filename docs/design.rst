Application design
==================

The application has three boundaries: VS Code owns files and panels, a native
Node-API addon owns ONNX models and analysis, and a webview owns the interactive
display. Python is not part of the runtime. See :doc:`features` for user controls
and :doc:`development` for building and testing.

Architecture
------------

.. code-block:: text

   Local ONNX file                 Released backend test catalog
         | file path / workspace.fs.readFile  |
         v                                   v
   VS Code extension host: src/extension.ts
         |
         | typed calls through src/graph.ts
         v
   Node-API addon: native/addon.cc <--> published onnx-light C++ libraries
         |
         | graph / tensor / metadata snapshots returned to the host
         v
   Extension host <-- postMessage --> webview
                                      |
                                      +-- src/webview.ts: HTML shell
                                      +-- media/viewer.js: SVG and interaction
                                      +-- media/viewer.css: presentation

File access and native calls happen in the extension host, not in the webview.
Local model files are parsed on a native worker so the loading progress bar stays
responsive. Graph projection and analysis methods are synchronous; there is no
model-execution service.
Only the selected graph is projected for display, rather than recursively
serializing all its descendants.

TypeScript source guide
-----------------------

``src/extension.ts``
~~~~~~~~~~~~~~~~~~~~

:download:`Source: extension.ts <../src/extension.ts>`.

``activate`` registers the demo and symbolic-model commands, plus the custom
editor for ``.onnx`` files. ``readGraph`` streams local files through the native
loader; other URI schemes use VS Code's filesystem API. Both create a retained
native model with its initial graph snapshot.

``configurePanel`` owns a panel-local ``LoadedModel``: its native handle,
currently displayed ``Graph`` and optional cached ``GraphMetadata``. It
configures media access, installs the message router and supplies the HTML.
The message subscription is disposed with the panel. A reload factory retains
the local file path, original virtual-file bytes or backend case ID independently
of the native handle.

Opening another model replaces that state only after successful loading.
Navigation changes the current graph and clears its metadata snapshot.
Asynchronous recovery results are ignored after panel disposal, model replacement
or navigation to another graph. Shape and metadata refreshes of the same graph
can share recovery without invalidating each other's results.
A subsequent ``ready`` message resends the current graph and any cached
metadata; it does not reload the source file or return to the root graph.

``isGraphPath`` checks incoming message structure. Native path resolution
also validates indices against the retained model. Native failures are
reported through VS Code notifications; navigation, shape and metadata errors
also have dedicated webview responses.

``src/log.ts``
~~~~~~~~~~~~~~

:download:`Source: log.ts <../src/log.ts>`.

Defines structured action entries, retains each panel's history and publishes
status updates. It validates incoming operation events and correlates native
operations with the webview's action IDs.

``src/huggingface.ts``
~~~~~~~~~~~~~~~~~~~~~~

:download:`Source: huggingface.ts <../src/huggingface.ts>`.

``parseHuggingFaceInput`` accepts public repository IDs, repository URLs and ONNX
file URLs. ``listHuggingFaceModels`` queries the Hub model API for ONNX files;
the host offers a variant picker when several files are present.
``parseHuggingFaceUrl`` normalizes blob file links to resolve links.
``cachedHuggingFaceModel`` keys an
extension-owned cache by the normalized URL's SHA-256 hash. Downloads follow
Hub/CDN redirects and stream to a uniquely named temporary file; the host's
native loading callback must succeed before an atomic rename publishes it.
Failed or cancelled transfers remove their temporary file. Cache hits are
loaded directly without a network request. Repository inputs still require a
network request to list files; direct cached file URLs work offline.

Cache storage is under ``context.globalStorageUri/huggingface`` in the extension
host. The original remote basename is passed to the native loader, not the
temporary cache name. After publication, the recovery factory uses the final
cache path, not the renamed temporary file. Recovery streams that local file
without another download. Authentication, external weight downloads and model
execution are outside this action's scope.

``src/graph.ts``
~~~~~~~~~~~~~~~~

:download:`Source: graph.ts <../src/graph.ts>`.

This file defines the snapshot contracts and loads
``dist/native/onnx_light_viewer.node``. Its wrappers expose ``loadModel``,
``loadModelFile``, ``decodeGraph``, ``listBackendTests`` and ``loadBackendTest``.

* ``Graph`` contains nodes, input/output names, tensor descriptions, an
  initializer count and breadcrumbs. ``GraphNode`` contains explicit inputs,
  implicit captures, attributes, initializer metadata and an optional schema.
* ``TensorInformation`` preserves dimensions as strings, including symbols.
  A shape of ``[]`` means scalar, ``null`` means unknown rank, and an omitted
  shape is used for non-tensor types. Unknown element types can be ``null``.
* ``GraphMetadata`` is a separate analysis snapshot, not a replacement graph.
  It contains stored metadata entries, constant/tag information and per-node
  lifetime lists.
* ``BackendTest.id`` is the original native collection index. Sorting names
  for the selector does not change IDs; names need not be unique.

The retained model interface is:

.. code-block:: typescript

   interface NativeModel {
     graph(inferShapes?: boolean, path?: GraphPath): Graph;
     metadata(path?: GraphPath): GraphMetadata;
   }

An omitted path selects the root graph. ``graph()`` does not infer shapes;
``graph(true, path)`` first infers the enclosing model and then returns the
selected graph. ``decodeGraph`` instead produces a one-shot snapshot, skipping
raw tensor bytes by default, without retaining a model for future refreshes.

``src/webview.ts``
~~~~~~~~~~~~~~~~~~

:download:`Source: webview.ts <../src/webview.ts>`.

``webviewHtml`` generates the Model/Logs/Status tabs, toolbar, SVG canvas, inspector,
breadcrumbs and operator-documentation dialog. Its resource callback converts
bundled media filenames to webview URIs. The content security policy permits
local images/styles and authorizes the viewer script with a per-panel nonce.
The host restricts local resource roots to ``media/``.

Model-derived strings are not interpolated into this HTML. They arrive through
messages and are inserted as DOM text by the viewer, including metadata values.
Schema documentation uses the bundled markdown-it renderer with raw HTML
disabled, HTTP(S)-only links and image labels rendered as escaped text.
Only this renderer's output is inserted as HTML in the documentation dialog;
no remote scripts or images are required.

Native C++ source guide
-----------------------

:download:`Source: native/addon.cc <../native/addon.cc>`.

``Initialize`` exports ``loadModel``, ``loadModelFile``, ``decodeGraph``, ``getSchema``,
``listBackendTests`` and ``loadBackendTest`` through Node-API. The addon
validates JavaScript arguments and converts C++ exceptions to JavaScript errors
at this boundary.

Parsing and ownership
~~~~~~~~~~~~~~~~~~~~~

``ReadModel`` parses bytes with onnx-light and checks that a model graph and
valid IR version exist. ``loadModelFile`` uses a buffered descriptor stream;
Windows paths are converted from UTF-8 before opening. Large inline raw and typed
tensor payloads are skipped, while constants with payloads smaller than
256 bytes remain available for shape inference and tiny initializer previews.
This accommodates up to 15 complex128 elements. Omitted payloads are
represented as external tensors in the retained model so structural validation
can still verify graph connectivity without requiring the values. Their storage
is shown as inline in the viewer, and external weight files are not opened. The
worker returns an owned model without retaining the file contents. Shape
inference that needs skipped weight values may fail and is reported through the
existing error handling.

Virtual-file reads still require a complete byte buffer because VS Code's
filesystem API does not expose streaming reads. The byte-buffer loading path
uses borrowed fields rather than copying tensor payloads. ``StoredModel``
therefore holds a persistent reference to the source ``Uint8Array`` alongside the
parsed model. Callers must not modify those bytes while the handle is in use.

Backend models instead retain their owning native ``TestCase``. Expected-output
generation is disabled; opening a case is not running a backend test.
``ModelHandle`` captures shared ownership in its ``graph`` and ``metadata``
method closures, so keeping just one method alive still keeps its backing
storage alive.

The display filename is only a label, separate from the local loader's file path.
Neither loading nor refreshing saves the model or opens external weight files.
Initializer snapshots contain type, dimensions and storage locations, plus
formatted values for available dense embedded tensors with fewer than 16
elements. Raw and typed payloads are decoded without converting 64-bit integers
to JavaScript numbers. Floating-point previews use float or double precision as
appropriate. Sparse previews contain stored values and indices when both components fit the
limit, independently of the logical dense shape. String previews retain at most
128 bytes per element. Attribute previews are bounded separately.

Graph projection and nested scopes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ReadPath``, ``ChildGraph`` and ``GraphScopes`` resolve an absolute
``GraphPath`` from the root. Each step contains a node index, attribute index
and graph index: zero for a ``GRAPH`` attribute, or an item index for ``GRAPHS``.
Alternatively, a function step contains a call-site node index and a model-local
function index. Resolution verifies that domain, operator name and overload match
the called definition. Paths use indices, not potentially duplicate graph or
node names.

``GraphSelection`` retains borrowed graph scopes and owns temporary graph views
of the selected function bodies. Only navigated definitions are materialized,
with shallow copies of their nodes and declared types. Entering a function resets
the capture boundary: caller tensors, initializers and constants do not leak into
its formal namespace. Nested graph attributes still inherit from that function.
Full enclosing scopes remain available for breadcrumbs.

``ModelGraph`` walks enclosing scopes to inherit tensor descriptions and
initializer metadata. Local inputs, weights and node outputs shadow enclosing
names. ``CollectNodeInputs`` identifies implicit dependencies, including
captures in nested attributes. Captured values appear as input ports, whereas
captured initializers remain parameters.

Graph-valued attributes and local function calls expose lightweight
``SubgraphLink`` objects containing a name, path and node count. Breadcrumbs carry
paths back to ancestors.
``Schema`` selects the newest matching onnx-light ``LightOpSchema`` not newer
than the model's or selected function's imported opset, supplying documentation
and attribute defaults. Local function calls do not use registered operator
schemas. Defaults do not overwrite explicit attribute values.

Shape inference versus metadata analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``InferShapes`` registers onnx-light's shape functions and computes a
``ShapesContext`` for the whole retained model. It applies root information to
the model; ``ApplySubgraphShapes`` then applies retained child contexts to
``GRAPH`` attributes recursively. This persists inferred outputs and
intermediate ``value_info`` inside the in-memory model, not just in labels.

Both shape inference and metadata analysis first call the schema-free
``VerifyModel`` validator. Snapshot projection remains available for inspecting
invalid connectivity without running analysis. Native algorithm boundaries add
the failing algorithm's name to exceptions, preserving its complete error text.

``ModelMetadata`` analyzes only the selected graph using ``InferConstants``,
``InferValueAndNodeTags`` and ``ComputeResultLifetimeInfo(graph, false)``.
Constant analysis is seeded from enclosing scopes. Results distinguish
releasable intermediates, caller-owned values reaching their last read and
unused non-output results. Metadata analysis is read-only: it does not infer
shapes or execute operators.

Host and webview protocol
-------------------------

The viewer sends requests with ``acquireVsCodeApi().postMessage``; the host
replies with ``panel.webview.postMessage``.

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Request
     - Host action
     - Response
   * - ``ready``
     - Resend current snapshots.
     - ``log-history``, ``application-status`` and ``graph`` with optional
       cached metadata, or ``load-state`` before a model is available.
   * - ``interaction`` with ``entry``
     - Retain a timestamped explicit operation or viewer error.
     - ``log-entry``.
   * - ``open``
     - Choose a local file, a public Hugging Face model or a backend test;
       load the file, download/reuse the persistent cache or search the native
       catalog.
     - ``graph`` or ``load-error``; cancellation leaves the view unchanged.
   * - ``open-backend-test``
     - Search the released native catalog and load a case.
     - ``graph`` on success; cancellation leaves the view unchanged.
   * - ``navigate`` with ``path``
     - Project the selected graph and discard cached metadata.
     - ``graph`` or ``navigation-error``.
   * - ``refresh-shapes``
     - Infer the retained model and update the current graph snapshot.
     - ``shape-info`` containing tensors, or ``shape-error``.
   * - ``metadata``
     - Recompute analysis for the current graph and cache the snapshot.
     - ``metadata-info`` or ``metadata-error``.

Opening failures produce ``load-error`` and VS Code notifications without
replacing the displayed graph. Refresh/navigation errors clear their busy state
and show a toolbar error without replacing the previous display.

The ``graph`` response rebuilds the display and resets its layout/selection.
``shape-info`` updates tensor labels and the inspector in place, preserving
positions and the camera. ``metadata-info`` updates the separate analysis
view. The metadata toggle requests analysis only if none is available;
**Refresh metadata** always requests a fresh snapshot, even with the panel closed.

Both refresh actions show per-refresh update counts in the main toolbar, including
zero when nothing changed. Shape/type updates count each added, removed or changed
tensor once. Metadata updates count each added, removed or changed model metadata
entry, graph metadata entry, node analysis or value analysis once. The first
metadata analysis counts all newly available records; subsequent refreshes compare
with the previous successful snapshot. Loading or navigating to a graph resets
these comparisons. Labels, graph markers and the current inspector are updated
in place without resetting positions or selection.

Action history and recovery
---------------------------

Each panel owns an ``ActionLog`` (``src/log.ts``). Entries have an ID, UTC
timestamp, action, status and optional error details. ``ActionLog.run`` publishes
``RUNNING`` then ``SUCCEEDED`` or ``FAILED``; cancelled pickers use
``CANCELLED``. Failures retain their full error stack and are rethrown to the UI
error handlers. The panel exists before initial loading so even loading
failures are inspectable.

Explicit model operations immediately display an entry and send
``interaction``. Native requests carry the same ``actionId``:
host ``log-entry`` updates replace that row rather than adding duplicate
start/completion rows. Refresh positions finishes in the webview. Tabs, node
selection, zoom and arbitrary clicks are not logged. ``ready`` replays
the history after webview recreation. Model replacement does not clear it.

The **Model**, **Logs** and **Status** tab panels hide/show existing DOM;
switching tabs does not render the graph again or reset browser-side state.
The **Show Logs** command reveals the viewer and sends ``show-logs`` to select
its Logs tab.

A failed shape or metadata refresh invalidates the native handle, not the
last displayed snapshots. Before the next native action, ``nativeModel``
logs and recreates that handle by streaming the local file again, using retained
virtual-file bytes, or rebuilding the backend case. Local files, including cache
entries, must still exist; source files are never modified. This discards
potentially partial native inference state; previously inferred information may need refreshing
after recovery. Reload failures are logged and surfaced like other failures.

``application-status`` is sent on action updates, model replacement, native
handle recreation and ``ready``. It reports the imported library/version,
native handle state (``none``, ``ready`` or ``invalidated``), running operations
and last failure. A caught C++ exception becomes a JavaScript error; it does
not unload the native addon. Status distinguishes the still-loaded library
from a discarded model handle and retains the last failure after recovery.
An addon import failure prevents activation, and native process crashes are
not covered by this exception-recovery mechanism.

Browser-side state and layout
-----------------------------

``media/viewer.js`` owns selection, node positions, zoom/pan, gestures, tensor
labels and metadata display state. Moving nodes, selecting, zooming and
**Refresh positions** are local operations; they do not call native methods or
change the model. ``media/viewer.css`` supplies the layout and SVG styles.

The initial layout places inputs above node-index rows and outputs below them,
with randomized operator horizontal positions. Operator dimensions account for
full labels and clickable subgraph rows. Connections attach using those
dimensions. Subgraph entries and inspector buttons send the same navigation
request; breadcrumbs navigate with the same path contract.

Positions are stable objects indexed to their incident connections.
``moveNodes`` updates only moved groups and affected edges, rather than
rebuilding the graph on every pointer move. Selection rendering also tracks
changed nodes. A gesture caches its viewport mapping; changes that resize the
canvas invalidate that mapping. Zoom and pan change the SVG ``viewBox``, not
model coordinates.

Above 1,000 nodes, view records retain lightweight SVG factories instead of
creating the full SVG tree. The initial camera uses readable size; camera
updates attach only intersecting detail groups and detach offscreen groups.
Tensor labels and metadata markers are initialized from current snapshots when
a group is first created. **Fit** draws the whole graph as two batched paths;
long edges whose labels are offscreen also use batched paths in detail view.
Node search uses the complete view index, so it can select and center nodes
which have never had a DOM element. These operations preserve model coordinates
and the current inspection state.

Build and verification boundaries
---------------------------------

``CMakeLists.txt`` downloads a pinned, checksum-verified onnx-light C++ release
for the target platform and links its operator, shape and backend-test
libraries. Only the C++20 addon is compiled locally. ``scripts/build-native.cjs``
drives CMake through ``cmake-js``; TypeScript compilation and the native output
are placed in ``dist/``. The native binary must match the extension host's OS
and architecture, including Remote WSL.

The same pinned ``ONNX_LIGHT_VERSION`` is embedded in the addon and exposed as
``onnxLightVersion`` through ``src/graph.ts``. The host passes this build
information to the HTML shell, which displays it in the toolbar independently
of the loaded model. There is no separate hard-coded UI version.

Tests follow the same boundaries: ``test/graph.test.cjs`` exercises the real
addon and retained models; ``test/extension.test.cjs`` exercises host message
routing; ``test/viewer.test.cjs`` exercises DOM/SVG interactions in jsdom.
``test/start.test.cjs`` checks the launcher. The Sphinx build checks this
documentation with warnings treated as errors.
