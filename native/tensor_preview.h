#pragma once

#include <napi.h>
#include <onnx_proto/onnx.h>

#include <algorithm>
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
std::string IntegerAttributePreview(std::string_view name, int64_t value, std::string_view domain);
std::string AttributePreview(const ol::AttributeProto &attribute, std::string_view domain);
template <typename Dims>
std::optional<size_t> TinyElementCount(const Dims &dims, size_t limit = 16) {
  if (std::find(dims.begin(), dims.end(), 0) != dims.end())
    return 0;
  size_t count = 1;
  for (const auto dim : dims) {
    if (dim < 0 || static_cast<size_t>(dim) >= limit ||
        count > (limit - 1) / static_cast<size_t>(dim))
      return std::nullopt;
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::optional<std::string> TinyTensorValues(const ol::TensorProto &tensor, size_t limit = 16,
                                            bool truncate_strings = true);

} // namespace onnx_light_viewer
