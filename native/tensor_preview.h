#pragma once

#include <napi.h>
#include <onnx_proto/onnx.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace onnx_light_viewer {
namespace ol = ONNX_LIGHT_NAMESPACE;

Napi::String Text(Napi::Env env, std::string_view value);

template <typename Proto> std::string Preview(const Proto &value, size_t limit = 4096) {
  ol::utils::PrintOptions options;
  options.skip_raw_data = true;
  options.max_short_repr_length = limit;
  std::stringstream stream;
  value.PrintToStringStream(stream, options);
  ol::utils::enforce_short_repr_length(stream, options);
  return stream.str();
}

template <typename Values> Napi::Array Ports(Napi::Env env, const Values &values) {
  auto result = Napi::Array::New(env);
  uint32_t index = 0;
  for (const auto &value : values) {
    if (!value.empty()) {
      result.Set(index++, Text(env, value));
    }
  }
  return result;
}

bool IsPayloadOmitted(const ol::TensorProto &tensor);
std::string AttributePreview(const ol::AttributeProto &attribute);
std::optional<std::string> TinyTensorValues(const ol::TensorProto &tensor);

} // namespace onnx_light_viewer
