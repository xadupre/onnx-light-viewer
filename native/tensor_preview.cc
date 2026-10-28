/** @file Bounded attribute previews and small inline tensor values. */
#include "tensor_preview.h"
#include "model.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <onnx_core/runtime/kernels/cast_float8.h>
#include <onnx_core/runtime/kernels/cast_helper.h>
#include <onnx_core/runtime/memory/simple_tensor.h>
#include <onnx_op/operator_sets.h>
#include <stdexcept>
#include <type_traits>

namespace onnx_light_viewer {
namespace {

template <typename Values> size_t PreviewValues(const Values &values, Values &preview) {
  for (size_t index = 0; index < values.size() && index < 8; ++index) {
    preview.push_back(values[index]);
  }
  return values.size() > 8 ? values.size() : 0;
}

std::string PreviewCount(size_t count) {
  return count ? " ... (" + std::to_string(count) + " elements)" : "";
}

template <typename T, typename Values> std::string AttributeListPreview(const Values &values) {
  std::vector<T> preview;
  for (size_t index = 0; index < values.size() && index < 8; ++index) {
    if constexpr (std::is_same_v<T, std::string>)
      preview.emplace_back(values[index].substr(0, 512));
    else
      preview.push_back(values[index]);
  }
  auto result = ol::core::schema::AttributeDefaultRepr(preview);
  if (result.size() > 512)
    result = result.substr(0, 509) + "...";
  return result + PreviewCount(values.size() > 8 ? values.size() : 0);
}

std::string TensorPreview(const ol::TensorProto &tensor) {
  ol::TensorProto preview;
  preview.set_name(tensor.name());
  preview.set_data_type(tensor.data_type());
  for (const auto dimension : tensor.dims()) {
    preview.add_dims(dimension);
  }
  const bool payload_omitted = IsPayloadOmitted(tensor);
  if (!payload_omitted) {
    for (const auto &entry : tensor.external_data()) {
      *preview.add_external_data() = entry;
    }
  }
  if (!payload_omitted && tensor.has_data_location()) {
    preview.set_data_location(tensor.data_location());
  }
  size_t count = 0;
  count = std::max(count, PreviewValues(tensor.float_data(), preview.ref_float_data()));
  count = std::max(count, PreviewValues(tensor.double_data(), preview.ref_double_data()));
  count = std::max(count, PreviewValues(tensor.int32_data(), preview.ref_int32_data()));
  count = std::max(count, PreviewValues(tensor.int64_data(), preview.ref_int64_data()));
  count = std::max(count, PreviewValues(tensor.uint64_data(), preview.ref_uint64_data()));
  count = std::max(count, PreviewValues(tensor.string_data(), preview.ref_string_data()));
  return Preview(preview, 512) + PreviewCount(count);
}

std::string SparsePreview(const ol::SparseTensorProto &tensor) {
  std::stringstream result;
  result << "{ dims: [";
  for (int index = 0; index < tensor.dims_size(); ++index) {
    result << (index ? ", " : "") << tensor.dims(index);
  }
  result << "] values: " << TensorPreview(tensor.values())
         << " indices: " << TensorPreview(tensor.indices()) << " }";
  return result.str();
}

template <typename T> T TensorElement(const ol::core::runtime::Tensor &tensor, size_t index) {
  std::array<uint8_t, sizeof(T)> bytes;
  std::copy_n(tensor.bytes() + index * sizeof(T), sizeof(T), bytes.begin());
  if constexpr (std::endian::native == std::endian::big)
    std::reverse(bytes.begin(), bytes.end());
  T value;
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

} // namespace

Napi::String Text(Napi::Env env, std::string_view value) {
  return Napi::String::New(env, value.empty() ? "" : value.data(), value.size());
}

bool IsPayloadOmitted(const ol::TensorProto &tensor) {
  return std::any_of(tensor.external_data().begin(), tensor.external_data().end(),
                     [](const auto &entry) {
                       return entry.key() == "location" && entry.value() == skippedPayloadLocation;
                     });
}

std::string AttributePreview(const ol::AttributeProto &attribute) {
  if (!attribute.ref_attr_name().empty())
    return Preview(attribute, 512);
  switch (attribute.type()) {
  case ol::AttributeProto::FLOAT:
    return ol::core::schema::AttributeDefaultRepr(static_cast<double>(attribute.f()));
  case ol::AttributeProto::INT:
    return std::to_string(attribute.i());
  case ol::AttributeProto::STRING:
    return std::string(attribute.s().substr(0, 512)) + (attribute.s().size() > 512 ? "..." : "");
  case ol::AttributeProto::GRAPH:
    return std::to_string(attribute.g().node_size()) + " nodes";
  case ol::AttributeProto::GRAPHS:
    return std::to_string(attribute.graphs_size()) + " subgraphs";
  case ol::AttributeProto::FLOATS:
    return AttributeListPreview<double>(attribute.floats());
  case ol::AttributeProto::INTS:
    return AttributeListPreview<int64_t>(attribute.ints());
  case ol::AttributeProto::STRINGS:
    return AttributeListPreview<std::string>(attribute.strings());
  case ol::AttributeProto::TENSOR:
    return TensorPreview(attribute.t());
  case ol::AttributeProto::SPARSE_TENSOR:
    return SparsePreview(attribute.sparse_tensor());
  case ol::AttributeProto::TENSORS: {
    std::string result = "[";
    for (int index = 0; index < attribute.tensors_size() && index < 8; ++index) {
      result += (index ? ", " : "") + TensorPreview(attribute.tensors(index));
    }
    return result + "]" + PreviewCount(attribute.tensors_size() > 8 ? attribute.tensors_size() : 0);
  }
  default:
    return Preview(attribute, 512);
  }
}

std::optional<std::string> TinyTensorValues(const ol::TensorProto &tensor, size_t limit,
                                            bool truncate_strings) {
  using namespace ol::core::runtime;
  const auto count = TinyElementCount(tensor.dims(), limit);
  if (!count || tensor.data_location() == ol::TensorProto::EXTERNAL ||
      !tensor.external_data().empty())
    return std::nullopt;
  if (*count == 0)
    return "[]";
  if (!tensor.has_raw_data() && tensor.float_data().empty() && tensor.double_data().empty() &&
      tensor.int32_data().empty() && tensor.int64_data().empty() && tensor.uint64_data().empty() &&
      tensor.string_data().empty())
    return std::nullopt;
  std::stringstream values;
  values.imbue(std::locale::classic());
  values << std::setprecision(tensor.data_type() == ol::TensorProto::DOUBLE ||
                                      tensor.data_type() == ol::TensorProto::COMPLEX128
                                  ? std::numeric_limits<double>::max_digits10
                                  : std::numeric_limits<float>::max_digits10)
         << "[";
  if (tensor.data_type() == ol::TensorProto::STRING) {
    if (tensor.string_data().size() != *count)
      throw std::runtime_error("Initializer '" + std::string(tensor.name()) +
                               "' string values do not match its shape: expected " +
                               std::to_string(*count) + " elements, got " +
                               std::to_string(tensor.string_data().size()) + ".");
    for (size_t index = 0; index < *count; ++index) {
      const auto &text = tensor.string_data(index);
      values << (index ? ", " : "")
             << std::quoted(truncate_strings ? text.substr(0, 128) : std::string(text));
      if (truncate_strings && text.size() > 128)
        values << "...";
    }
  } else {
    const auto data = TensorFromProto(tensor);
    for (size_t index = 0; index < *count; ++index) {
      values << (index ? ", " : "");
      const auto bits = [&] { return TensorElement<uint8_t>(data, index); };
      switch (tensor.data_type()) {
      case ol::TensorProto::FLOAT:
        values << TensorElement<float>(data, index);
        break;
      case ol::TensorProto::DOUBLE:
        values << TensorElement<double>(data, index);
        break;
      case ol::TensorProto::INT8:
        values << +TensorElement<int8_t>(data, index);
        break;
      case ol::TensorProto::UINT8:
        values << +bits();
        break;
      case ol::TensorProto::INT16:
        values << TensorElement<int16_t>(data, index);
        break;
      case ol::TensorProto::UINT16:
        values << TensorElement<uint16_t>(data, index);
        break;
      case ol::TensorProto::INT32:
        values << TensorElement<int32_t>(data, index);
        break;
      case ol::TensorProto::UINT32:
        values << TensorElement<uint32_t>(data, index);
        break;
      case ol::TensorProto::INT64:
        values << TensorElement<int64_t>(data, index);
        break;
      case ol::TensorProto::UINT64:
        values << TensorElement<uint64_t>(data, index);
        break;
      case ol::TensorProto::BOOL:
        values << (bits() ? "true" : "false");
        break;
      case ol::TensorProto::FLOAT16:
        values << Float16BitsToFloat(TensorElement<uint16_t>(data, index));
        break;
      case ol::TensorProto::BFLOAT16:
        values << Bfloat16BitsToFloat(TensorElement<uint16_t>(data, index));
        break;
      case ol::TensorProto::FLOAT8E4M3FN:
        values << Float8E4M3FNBitsToFloat(bits());
        break;
      case ol::TensorProto::FLOAT8E4M3FNUZ:
        values << Float8E4M3FNUZBitsToFloat(bits());
        break;
      case ol::TensorProto::FLOAT8E5M2:
        values << Float8E5M2BitsToFloat(bits());
        break;
      case ol::TensorProto::FLOAT8E5M2FNUZ:
        values << Float8E5M2FNUZBitsToFloat(bits());
        break;
      case ol::TensorProto::FLOAT8E8M0:
        values << Float8E8M0BitsToFloat(bits());
        break;
      case ol::TensorProto::FLOAT6E2M3:
      case ol::TensorProto::FLOAT6E3M2:
        values << Float6BitsToFloat(Read6BitElement(data.bytes(), static_cast<int64_t>(index)),
                                    static_cast<DataType>(tensor.data_type()));
        break;
      case ol::TensorProto::UINT4:
      case ol::TensorProto::INT4:
      case ol::TensorProto::FLOAT4E2M1:
      case ol::TensorProto::UINT2:
      case ol::TensorProto::INT2: {
        const bool two_bits = tensor.data_type() == ol::TensorProto::UINT2 ||
                              tensor.data_type() == ol::TensorProto::INT2;
        const int width = two_bits ? 2 : 4;
        const int value = (TensorElement<uint8_t>(data, index * width / 8) >> (index * width % 8)) &
                          ((1 << width) - 1);
        if (tensor.data_type() == ol::TensorProto::FLOAT4E2M1)
          values << Float4E2M1NibbleToFloat(static_cast<uint8_t>(value));
        else if (tensor.data_type() == ol::TensorProto::INT4 ||
                 tensor.data_type() == ol::TensorProto::INT2)
          values << (value >= (1 << (width - 1)) ? value - (1 << width) : value);
        else
          values << value;
        break;
      }
      case ol::TensorProto::COMPLEX64:
        values << "(" << TensorElement<float>(data, index * 2) << ", "
               << TensorElement<float>(data, index * 2 + 1) << ")";
        break;
      case ol::TensorProto::COMPLEX128:
        values << "(" << TensorElement<double>(data, index * 2) << ", "
               << TensorElement<double>(data, index * 2 + 1) << ")";
        break;
      default:
        throw std::runtime_error("Unsupported initializer element type '" +
                                 std::string(ol::TensorProto::DataType_Name(tensor.data_type())) +
                                 "' for initializer '" + std::string(tensor.name()) + "'.");
      }
    }
  }
  values << "]";
  return values.str();
}

} // namespace onnx_light_viewer
