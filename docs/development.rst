Development
===========

Manual launch
-------------

The simplest development loop is :doc:`start.sh <getting_started>`.
To launch manually instead:

.. code-block:: bash

   npm ci
   npm run compile

Open the repository in VS Code and press **F5** to launch the Extension
Development Host. Run **ONNX Light Viewer: Open Graph** in that window's
Command Palette to open the one-node demo.

``npm run compile`` builds the native binding and TypeScript.
``npm run watch`` watches TypeScript only; compile the native binding first,
and rebuild it with ``npm run build:native`` after editing C++.

The first native build downloads the published onnx-light C++ archive pinned
in ``CMakeLists.txt`` and verifies its SHA-256 checksum. Only the Node-API binding
is compiled; onnx-light is not rebuilt. The archive is reused from
``build/native/``. The binding must match the extension host's OS and architecture,
including Linux for Remote WSL.

Tests and formatting
--------------------

Run from the repository root:

.. code-block:: bash

   npm test
   npm run format:check
   clang-format --dry-run --Werror native/addon.cc

The **Tests** GitHub Actions workflow builds the binding, runs the tests and
checks JavaScript/C++ formatting on pull requests and pushes to ``main``.
Its documentation job also builds Sphinx with warnings treated as errors.
Results appear in the PR's **Checks** tab and the repository's **Actions** tab.

Test models
-----------

``test/models/relu.onnx`` contains one Relu operator with float32 input/output
tensors of shape ``[1, 3]`` (IR 9, opset 13). It can also be opened in the viewer.
The same model is bundled as ``media/relu.onnx`` for the command's demo.
Other small models cover custom operators, dense/sparse initializers, optional
ports, an empty operator list and intentionally invalid metadata.
Shape tests also cover symbolic dimensions, scalar and unknown ranks, and
inference through Add, Reshape, Transpose and Cast with a raw-data constant.

To regenerate them with the Python ``onnx-light`` package available:

.. code-block:: bash

   python3 test/models/generate_models.py

The bundled transformer example is generated separately with
``python3 examples/generate_tiny_llm.py``. Its ONNX
file is exercised directly by the native and extension tests. Initializer
metadata tests also use external-file references whose payload files are
intentionally absent, and more than eight connected initializers.

Python is not required to run the extension tests.

Building this documentation
---------------------------

With Python 3.11 or later, run from the repository root:

.. code-block:: bash

   python3 -m pip install -r docs/requirements.txt
   python3 -m sphinx -n -W --keep-going -b html docs build/docs/html

Open ``build/docs/html/index.html`` in a browser. Generated HTML is ignored by
Git. Documentation dependencies are separate from the extension's dependencies.

The documentation uses PyData Sphinx Theme, like onnx-light, with a light/dark
theme switcher in the navigation bar. By default, it follows the system theme.

Source layout
-------------

* ``src/extension.ts``: VS Code integration and model opening.
* ``src/graph.ts``: graph types and the typed native entry point.
* ``native/addon.cc``: native model decoding.
* ``media/viewer.js`` and ``media/viewer.css``: graph rendering and layout.
* ``media/logo.svg``: extension logo.
