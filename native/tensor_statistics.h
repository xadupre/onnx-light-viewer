#pragma once

#include <napi.h>

namespace onnx_light_viewer {

Napi::Value TensorStatisticsFromProto(const Napi::CallbackInfo &call);

} // namespace onnx_light_viewer
