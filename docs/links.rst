Links
=====

Related projects and export tools in the ONNX ecosystem.
The viewer's **ONNX** tab groups project and **Exporters** cards in the same
design, with the packages or ecosystems each project specializes in.

onnx-light projects
-------------------

.. figure:: _static/onnx-light-projects.svg
   :alt: onnx-light decodes models for onnx-light-viewer, onnx-light-cpu provides
         CPU kernels and onnx-light-kernel-images provides ImageDecoder kernels
         for onnx-light, and onnx-light-ort bridges onnx-light and ONNX Runtime
         in both directions.
   :width: 100%

   Local model inspection, CPU execution, image decoding, and ONNX Runtime
   interoperability.

* `onnx-light-viewer <https://github.com/xadupre/onnx-light-viewer>`_: this VS Code
  extension for inspecting ONNX graphs locally without executing models.
  **Specialized for:** VS Code model inspection with ``onnx-light``.
* `onnx-light <https://github.com/xadupre/onnx-light>`_: ONNX without protobuf,
  used by this viewer to decode models locally.
  **Specialized for:** ONNX model I/O and graph analysis with ``onnx-light``.
* `onnx-light-cpu <https://github.com/xadupre/onnx-light-cpu>`_: optimized CPU
  kernels for onnx-light.
  **Specialized for:** ``onnx-light`` CPU execution.
* `onnx-light-kernel-images <https://github.com/xadupre/onnx-light-kernel-images>`_:
  ImageDecoder kernel extension for onnx-light.
  **Specialized for:** image decoding with ``onnx-light``.
* `onnx-light-ort <https://github.com/xadupre/onnx-light-ort>`_: bidirectional
  interoperability between onnx-light and ONNX Runtime custom operators and
  execution providers.
  **Specialized for:** ``onnx-light`` and ``onnxruntime`` interoperability.

ONNX and ONNX Runtime
---------------------

.. figure:: _static/onnx-ecosystem.svg
   :alt: ONNX defines the model format consumed by ONNX Runtime;
         ONNX Runtime powers generative AI inference in onnxruntime-genai.
   :width: 100%

   From a shared model format to general inference and generative AI.
   onnx-light reads the same ONNX format; onnx-light-ort connects it to
   ONNX Runtime.

* `onnx <https://github.com/onnx/onnx>`_: the open standard for machine learning
  model interoperability.
  **Specialized for:** framework-independent model interchange with ``onnx``.
* `onnxruntime <https://github.com/microsoft/onnxruntime>`_: a cross-platform
  runtime for machine learning models.
  **Specialized for:** ONNX model execution with ``onnxruntime``.
* `onnxruntime-extensions <https://github.com/microsoft/onnxruntime-extensions>`_:
  custom operators and tools for preprocessing and postprocessing.
  **Specialized for:** ``onnxruntime`` preprocessing and postprocessing with
  ``onnxruntime-extensions``.
* `onnxruntime-genai <https://github.com/microsoft/onnxruntime-genai>`_:
  generative AI model inference with ONNX Runtime.
  **Specialized for:** generative model execution with ``onnxruntime-genai``.

Exporters
----------

Create an ONNX model with an external exporter, then open its ``.onnx`` file
in the viewer. Supported models, operators and target runtimes vary by tool.

.. figure:: _static/onnx-exporters.svg
   :alt: External exporters write an ONNX model and optional external weight
         files. ONNX Light Viewer opens the resulting model for inspection.
   :width: 100%

   Export happens outside the viewer; opening a model does not execute it.

* `Mobius <https://github.com/onnxruntime/mobius>`_:
  builds ONNX graphs with ONNX Script and pretrained weights rather than
  tracing a PyTorch model.
  **Specialized for:** Hugging Face Transformers and Diffusers models,
  including ONNX Runtime GenAI workflows.
* `modelbuilder (ONNX Runtime GenAI)
  <https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models>`_:
  builds optimized and quantized ONNX graphs for supported generative AI
  architectures.
  **Specialized for:** Hugging Face Transformers models targeting
  ``onnxruntime-genai``.
* `mbext <https://github.com/xadupre/mbext>`_:
  extends the GenAI model builder with additional architectures and fast,
  offline export-validation tests.
  **Specialized for:** Hugging Face Transformers models targeting
  ``onnxruntime-genai``, through the ``modelbuilder`` package.
* `yet-another-onnx-builder
  <https://github.com/xadupre/yet-another-onnx-builder>`_:
  a shared ONNX graph builder and the ``yobx.to_onnx`` conversion API.
  **Specialized for:** PyTorch, TensorFlow/LiteRT, scikit-learn, XGBoost and
  LightGBM models, using ``yobx`` and ``onnx-light``.
* `sklearn-onnx <https://github.com/onnx/sklearn-onnx>`_:
  converts supported estimators and pipelines through ``skl2onnx``.
  **Specialized for:** scikit-learn models and pipelines.
* `tensorflow-onnx <https://github.com/onnx/tensorflow-onnx>`_:
  converts models through ``tf2onnx``; TensorFlow.js support is experimental.
  **Specialized for:** TensorFlow, Keras, TFLite and TensorFlow.js.
* `onnxmltools <https://github.com/onnx/onnxmltools>`_:
  converts models from multiple machine-learning toolkits, including
  boosted-tree libraries.
  **Specialized for:** XGBoost, LightGBM, CatBoost, Core ML and other ML toolkits.

These links do not imply built-in export commands. Viewer-side exporter
integration is planned in `issue #131
<https://github.com/xadupre/onnx-light-viewer/issues/131>`_.
