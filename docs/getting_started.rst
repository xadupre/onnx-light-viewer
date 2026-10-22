Getting started
===============

Requirements
------------

* Node.js 22+, npm and VS Code 1.96+ with the desktop ``code`` command on ``PATH``.
* CMake 3.24+, a C++20 compiler and OpenSSL development packages.
* Internet access for the first build's npm dependencies and published
  onnx-light library. Python and ONNX Runtime are not required.

Launch
------

From a Bash terminal in the repository:

.. code-block:: bash

   ./start.sh

This builds the extension and opens a VS Code development window with a small
Relu model. Choose **Open ONNX model** to open your own ``.onnx`` file.
After changing the application, close that window and run ``./start.sh`` again.

In WSL, use the desktop VS Code ``bin`` directory on ``PATH``, not the remote
terminal CLI. The native binding must match the extension host platform.

See :doc:`features` for the viewer controls and :doc:`development` for tests
and manual builds.
