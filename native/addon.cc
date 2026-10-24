/** @file Node-API entry points for the onnx-light viewer. */
#include "addon.h"

namespace {

Napi::Object Initialize(Napi::Env env, Napi::Object exports) {
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
