#pragma once

#include <napi.h>

namespace onnx_light_viewer {

Napi::Value DecodeGraph(const Napi::CallbackInfo &info);
Napi::Value GetSchema(const Napi::CallbackInfo &info);
Napi::Value LoadModel(const Napi::CallbackInfo &info);
Napi::Value LoadModelFile(const Napi::CallbackInfo &info);
Napi::Value LoadPipeline(const Napi::CallbackInfo &info);
Napi::Value LoadPipelineFile(const Napi::CallbackInfo &info);
Napi::Value ListBackendTests(const Napi::CallbackInfo &info);
Napi::Value LoadBackendTest(const Napi::CallbackInfo &info);
Napi::Value RunNativePrompt(const Napi::CallbackInfo &info);
Napi::Value ReleaseNativePrompt(const Napi::CallbackInfo &info);

} // namespace onnx_light_viewer
