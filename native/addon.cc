/** @file Node-API entry points for the onnx-light viewer. */
#include "addon.h"
#include "graph.h"
#include <algorithm>
#include <onnx_core/builder/pattern_registry.h>
#include <onnx_extensions/kernels/kernel_dispatch_table.h>
#include <onnx_extensions/shapes/dispatch_table.h>
#include <onnx_light_cpu/kernels/register_kernels.h>
#include <onnx_light_cpu/patterns/com_microsoft/patterns.h>
#include <onnx_light_cpu/shapes/ai_onnx/shape_inference.h>
#include <onnx_light_cpu/shapes/com_microsoft/shape_inference.h>

namespace {

Napi::Object Initialize(Napi::Env env, Napi::Object exports) {
  static const bool registered = [] {
    ONNX_LIGHT_NAMESPACE::onnx_shapes::RegisterShapeFunctions();
    ONNX_LIGHT_NAMESPACE::onnx_shapes::RegisterPeakMemoryFunctions();
    ONNX_LIGHT_NAMESPACE::onnx_kernels::RegisterKernelFunctions();
    onnx_light_cpu::RegisterMicrosoftShapeAndMemoryFunctions();
    onnx_light_cpu::RegisterExperimentalShapeAndMemoryFunctions();
    const auto existing_patterns = ONNX_LIGHT_NAMESPACE::core::builder::RegisteredPatternNames();
    onnx_light_cpu::RegisterCustomOperatorPatterns();
    auto cpu_patterns = ONNX_LIGHT_NAMESPACE::core::builder::RegisteredPatternNames();
    std::erase_if(cpu_patterns, [&](const auto &name) {
      return std::find(existing_patterns.begin(), existing_patterns.end(), name) !=
             existing_patterns.end();
    });
    ONNX_LIGHT_NAMESPACE::core::builder::RegisterPatternSet("onnx-light-cpu", cpu_patterns);
    onnx_light_cpu::RegisterAllKernels();
    return true;
  }();
  (void)registered;
  exports.Set("onnxLightVersion", ONNX_LIGHT_VERSION);
  exports.Set("decodeGraph", Napi::Function::New(env, onnx_light_viewer::DecodeGraph));
  exports.Set("getSchema", Napi::Function::New(env, onnx_light_viewer::GetSchema));
  exports.Set("loadModel", Napi::Function::New(env, onnx_light_viewer::LoadModel));
  exports.Set("loadModelFile", Napi::Function::New(env, onnx_light_viewer::LoadModelFile));
  exports.Set("loadPipeline", Napi::Function::New(env, onnx_light_viewer::LoadPipeline));
  exports.Set("loadPipelineFile", Napi::Function::New(env, onnx_light_viewer::LoadPipelineFile));
  exports.Set("listBackendTests", Napi::Function::New(env, onnx_light_viewer::ListBackendTests));
  exports.Set("loadBackendTest", Napi::Function::New(env, onnx_light_viewer::LoadBackendTest));
  return exports;
}

} // namespace

NODE_API_MODULE(onnx_light_viewer, Initialize)
