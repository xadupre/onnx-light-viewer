/** @file Graph, initializer and operator-schema snapshots for JavaScript. */
#include "addon.h"
#include "graph.h"
#include "tensor_preview.h"
#include <algorithm>
#include <limits>
#include <map>
#include <onnx_core/graph/graph_manipulations.h>
#include <onnx_core/runtime/memory/simple_tensor.h>
#include <onnx_op/operator_sets.h>
#include <onnx_proto/onnx_helper.h>
#include <onnx_proto/onnx_verify.h>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace onnx_light_viewer {
namespace {

Napi::Array OpsetImports(Napi::Env env, const auto &imports) {
  auto result = Napi::Array::New(env);
  for (const auto &opset : imports) {
    auto entry = Napi::Object::New(env);
    entry.Set("domain", Domain(opset.domain()));
    entry.Set("version", Text(env, std::to_string(opset.version())));
    result.Set(result.Length(), entry);
  }
  return result;
}

Napi::Array Dimensions(Napi::Env env, const ol::TensorShapeProto &shape) {
  auto result = Napi::Array::New(env);
  uint32_t index = 0;
  for (const auto &dimension : shape.dim()) {
    result.Set(index++, dimension.has_dim_value()
                            ? Text(env, std::to_string(dimension.dim_value()))
                            : Text(env, dimension.dim_param().empty()
                                            ? std::string_view("?")
                                            : std::string_view(dimension.dim_param())));
  }
  return result;
}

using StructDeclarations = std::unordered_map<uint64_t, const ol::StructTypeProto *>;

struct EncodedPreviewCatalogue {
  ol::StructTypeCatalogue catalogue;
  bool initialized = false;
  std::string error;

  void Initialize(const ol::ModelProto &model) {
    if (initialized)
      return;
    initialized = true;
    try {
      catalogue.Build(model);
    } catch (const std::invalid_argument &failure) {
      error = failure.what();
    }
  }
};

class TypeTable {
public:
  TypeTable(Napi::Env env, const StructDeclarations &declarations)
      : rows_(Napi::Array::New(env)), declarations_(declarations) {}

  Napi::Object Logical(const ol::TypeProto &type) {
    AddType("", type, 0);
    return Finish(TypeName(type));
  }

  Napi::Object Encoding(const ol::EncodedValueProto &value, const ol::ModelProto &model,
                        EncodedPreviewCatalogue &catalogue) {
    payload_ = &value;
    if (value.data_location() == ol::TensorProto::EXTERNAL || !value.external_data().empty())
      preview_error_ = "External payload is not read for inspection.";
    else if (!value.has_raw_data())
      preview_error_ = "No inline payload.";
    if (value.has_struct_type()) {
      if (preview_error_.empty()) {
        catalogue.Initialize(model);
        catalogue_ = &catalogue.catalogue;
        uint64_t bits = 0;
        if (!catalogue.error.empty())
          preview_error_ = catalogue.error;
        else if (!catalogue_->FixedBitSize(value.struct_type(), bits, &preview_error_)) {
          if (preview_error_.empty())
            preview_error_ = "Layout has no fixed physical size.";
        } else if (bits == 0 || bits % 8 != 0) {
          preview_error_ = "Encoded records must have a positive whole-byte size.";
        } else if (value.raw_data().size() % (bits / 8) != 0) {
          preview_error_ = "Payload size is not a multiple of the record size.";
        } else if (value.raw_data().empty()) {
          preview_error_ = "Inline payload is empty.";
        } else {
          position_ = 0;
          Add("Payload preview", "First record of " +
                                     std::to_string(value.raw_data().size() / (bits / 8)) +
                                     "; stored fields, not dequantized weights.");
        }
      }
      if (!preview_error_.empty())
        Add("Payload preview", "Unavailable: " + preview_error_);
      AddStruct("", value.struct_type(), 0);
      return Finish("STRUCT");
    }
    const auto &layout = value.affine();
    if (layout.has_storage_type())
      Add("Storage type", ol::TensorProto::DataType_Name(layout.storage_type()));
    if (layout.has_signed_storage())
      Add("Signed storage", std::to_string(layout.signed_storage()));
    if (layout.has_axis())
      Add("Axis", std::to_string(layout.axis()));
    if (layout.has_block_size())
      Add("Block size", std::to_string(layout.block_size()));
    if (layout.has_scale())
      AddConstant("Scale", layout.scale());
    if (layout.has_zero_point())
      AddConstant("Zero point", layout.zero_point());
    const auto codes = Add("Codes", layout.has_storage_type()
                                        ? ol::TensorProto::DataType_Name(layout.storage_type())
                                        : "Packed integers");
    if (!preview_error_.empty())
      SetData(codes, "Unavailable: " + preview_error_);
    else if (layout.has_storage_type() && value.has_logical_type() &&
             value.logical_type().has_tensor_type()) {
      uint64_t bits = 0;
      if (TensorBits(value.logical_type().tensor_type(), bits)) {
        const auto width = ol::FixedBitWidth(value.logical_type().tensor_type().elem_type());
        SetData(codes, TensorData(layout.storage_type(), bits / width, 0));
      } else
        SetData(codes, "Unavailable: " + preview_error_);
    } else
      SetData(codes, "Unavailable: code width or logical element count is unknown.");
    return Finish("AFFINE");
  }

private:
  class RowScope {
  public:
    RowScope(TypeTable &table, uint32_t row) : table_(table), previous_(table.parent_) {
      table_.parent_ = row;
    }
    ~RowScope() { table_.parent_ = previous_; }

  private:
    TypeTable &table_;
    std::optional<uint32_t> previous_;
  };

  void SetData(uint32_t row, const std::string &data) {
    if (row < 256 && !data.empty())
      rows_.Get(row).As<Napi::Object>().Set("data", Text(rows_.Env(), Short(data)));
  }

  void SetShape(uint32_t row, const std::string &shape) {
    if (row < 256)
      rows_.Get(row).As<Napi::Object>().Set("shape", Text(rows_.Env(), Short(shape)));
  }

  uint32_t Group(const std::string &path, std::string_view description) {
    if (parent_ && !path.empty()) {
      auto entry = rows_.Get(*parent_).As<Napi::Object>();
      if (entry.Get("key").As<Napi::String>().Utf8Value() == path) {
        entry.Set("value", Text(rows_.Env(), description));
        return *parent_;
      }
    }
    return Add(path.empty() ? "Layout" : path, description);
  }

  bool Multiply(uint64_t left, uint64_t right, uint64_t &result) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) {
      preview_error_ = "Physical size exceeds the supported range.";
      return false;
    }
    result = left * right;
    return true;
  }

  bool TensorBits(const auto &type, uint64_t &bits) {
    const auto width = ol::FixedBitWidth(type.elem_type());
    if (!width || !type.has_shape()) {
      preview_error_ = "Tensor has no fixed numeric shape or element width.";
      return false;
    }
    bool empty = false;
    for (const auto &dim : type.shape().dim()) {
      if (!dim.has_dim_value() || dim.dim_value() < 0) {
        preview_error_ = "Tensor has non-concrete dimensions.";
        return false;
      }
      empty |= dim.dim_value() == 0;
    }
    bits = empty ? 0 : width;
    for (const auto &dim : type.shape().dim())
      if (!Multiply(bits, dim.dim_value(), bits))
        return false;
    return true;
  }

  bool TypeBits(const ol::TypeProto &type, uint64_t &bits) {
    if (type.has_tensor_type())
      return TensorBits(type.tensor_type(), bits);
    if (type.has_struct_type() && catalogue_)
      return catalogue_->FixedBitSize(type.struct_type(), bits, &preview_error_);
    preview_error_ = "Field has no fixed inline representation.";
    return false;
  }

  void Advance(uint64_t bits) {
    if (!position_)
      return;
    if (bits > std::numeric_limits<uint64_t>::max() - *position_) {
      position_.reset();
      preview_error_ = "Physical offset exceeds the supported range.";
    } else
      *position_ += bits;
  }

  bool Slice(uint64_t offset, uint64_t bits) const {
    return bits == 0 || (bits - 1 <= std::numeric_limits<uint64_t>::max() - offset &&
                         (offset + bits - 1) / 8 < payload_->raw_data().size());
  }

  uint64_t ReadBits(uint64_t offset, uint32_t width) const {
    uint64_t result = 0;
    for (uint32_t index = 0; index < width; ++index) {
      const auto bit = offset + index;
      result |= uint64_t((payload_->raw_data()[bit / 8] >> (bit % 8)) & 1) << index;
    }
    return result;
  }

  std::string TensorData(ol::TensorProto::DataType type, uint64_t count, uint64_t offset) {
    if (count == 0)
      return "";
    const uint64_t width = ol::FixedBitWidth(type);
    if (!width)
      return "Unavailable: unsupported numeric element type.";
    uint64_t total_bits = 0;
    if (!Multiply(count, width, total_bits))
      return "Unavailable: " + preview_error_;
    if (!Slice(offset, total_bits))
      return "Unavailable: inline payload is truncated.";
    const auto shown = std::min<uint64_t>(count, 15);
    const auto bits = shown * width;
    std::string bytes((bits + 7) / 8, '\0');
    for (uint64_t bit = 0; bit < bits; ++bit) {
      const auto source = offset + bit;
      bytes[bit / 8] |= ((payload_->raw_data()[source / 8] >> (source % 8)) & 1) << (bit % 8);
    }
    ol::TensorProto preview;
    preview.set_data_type(type);
    preview.add_dims(shown);
    preview.set_raw_data(bytes);
    const auto values = TinyTensorValues(preview);
    if (!values)
      return "Unavailable: numeric preview is unsupported.";
    return *values + (shown < count ? " ... (first " + std::to_string(shown) + " of " +
                                          std::to_string(count) + " elements)"
                                    : "");
  }

  static std::string Short(std::string_view text) {
    return text.size() <= 512 ? std::string(text) : std::string(text.substr(0, 509)) + "...";
  }

  static std::string Key(const std::string &path, std::string_view name) {
    return Short(path.empty() ? Short(name) : path + "." + Short(name));
  }

  bool Full() {
    if (rows_.Length() < 256)
      return false;
    if (rows_.Length() == 256) {
      auto entry = Napi::Object::New(rows_.Env());
      entry.Set("key", "...");
      entry.Set("value", "Additional type fields omitted.");
      if (parent_)
        entry.Set("parent", *parent_);
      rows_.Set(rows_.Length(), entry);
    }
    return true;
  }

  uint32_t Add(const std::string &key, std::string_view value) {
    if (Full())
      return rows_.Length() - 1;
    auto entry = Napi::Object::New(rows_.Env());
    entry.Set("key", Text(rows_.Env(), Short(key)));
    entry.Set("value", Text(rows_.Env(), Short(value)));
    if (parent_)
      entry.Set("parent", *parent_);
    const auto index = rows_.Length();
    rows_.Set(index, entry);
    return index;
  }

  bool Stop(const std::string &path, size_t depth) {
    if (Full())
      return true;
    if (depth >= 16) {
      Add(Key(path, "..."), "Maximum type nesting depth reached.");
      return true;
    }
    return false;
  }

  static std::string TypeName(const ol::TypeProto &type) {
    if (type.has_tensor_type())
      return ol::TensorProto::DataType_Name(type.tensor_type().elem_type());
    if (type.has_sparse_tensor_type())
      return std::string("SPARSE ") +
             ol::TensorProto::DataType_Name(type.sparse_tensor_type().elem_type());
    if (type.has_struct_type())
      return "STRUCT";
    if (type.has_opaque_type())
      return "OPAQUE";
    if (type.has_sequence_type())
      return "SEQUENCE";
    if (type.has_map_type())
      return "MAP";
    if (type.has_optional_type())
      return "OPTIONAL";
    return "Unknown";
  }

  void AddType(const std::string &path, const ol::TypeProto &type, size_t depth) {
    if (Stop(path, depth)) {
      uint64_t bits = 0;
      if (payload_ && position_) {
        if (TypeBits(type, bits))
          Advance(bits);
        else
          position_.reset();
      }
      return;
    }
    const auto row = Add(path.empty() ? "Type" : path, TypeName(type));
    RowScope scope(*this, row);
    if (type.has_tensor_type() || type.has_sparse_tensor_type()) {
      const bool sparse = type.has_sparse_tensor_type();
      const bool known =
          sparse ? type.sparse_tensor_type().has_shape() : type.tensor_type().has_shape();
      std::string shape = known ? "[" : "?";
      bool empty = false;
      if (known) {
        const auto &dimensions =
            sparse ? type.sparse_tensor_type().shape() : type.tensor_type().shape();
        empty = std::any_of(dimensions.dim().begin(), dimensions.dim().end(),
                            [](const auto &dimension) {
                              return dimension.has_dim_value() && dimension.dim_value() == 0;
                            });
        for (const auto &dimension : dimensions.dim()) {
          if (shape.size() > 512) {
            shape += "...";
            break;
          }
          if (shape.size() > 1)
            shape += ", ";
          shape += dimension.has_dim_value() ? std::to_string(dimension.dim_value())
                                             : Short(dimension.dim_param().empty()
                                                         ? std::string_view("?")
                                                         : std::string_view(dimension.dim_param()));
        }
        shape += "]";
      }
      SetShape(row, shape);
      if (payload_ && !empty) {
        uint64_t bits = 0;
        if (position_ && type.has_tensor_type() && TensorBits(type.tensor_type(), bits)) {
          const auto width = ol::FixedBitWidth(type.tensor_type().elem_type());
          SetData(row, TensorData(type.tensor_type().elem_type(), bits / width, *position_));
          Advance(bits);
        } else {
          SetData(row, "Unavailable: " + preview_error_);
          position_.reset();
        }
      }
    } else if (type.has_struct_type()) {
      AddStruct(path, type.struct_type(), depth + 1);
    } else if (type.has_opaque_type()) {
      Add(Key(path, "Domain"), type.opaque_type().domain());
      Add(Key(path, "Name"), type.opaque_type().name());
    } else if (type.has_sequence_type() && type.sequence_type().has_elem_type()) {
      AddType(Key(path, "Element"), type.sequence_type().elem_type(), depth + 1);
    } else if (type.has_optional_type() && type.optional_type().has_elem_type()) {
      AddType(Key(path, "Element"), type.optional_type().elem_type(), depth + 1);
    } else if (type.has_map_type()) {
      Add(Key(path, "Key type"), ol::TensorProto::DataType_Name(type.map_type().key_type()));
      if (type.map_type().has_value_type())
        AddType(Key(path, "Value"), type.map_type().value_type(), depth + 1);
    }
    if (payload_ && !type.has_tensor_type() && !type.has_struct_type()) {
      SetData(row, "Unavailable: field has no fixed inline representation.");
      position_.reset();
    }
    if (type.has_denotation())
      Add(Key(path, "Denotation"), type.denotation());
  }

  void AddConstant(const std::string &path, const ol::TensorProto &tensor) {
    const auto row =
        Add(path, std::string("Constant ") + ol::TensorProto::DataType_Name(tensor.data_type()));
    std::string shape = "[";
    for (const auto dimension : tensor.dims()) {
      if (shape.size() > 512) {
        shape += "...";
        break;
      }
      if (shape.size() > 1)
        shape += ", ";
      shape += std::to_string(dimension);
    }
    SetShape(row, shape + "]");
    if (Full() || std::find(tensor.dims().begin(), tensor.dims().end(), 0) != tensor.dims().end())
      return;
    try {
      if (const auto values = TinyTensorValues(tensor))
        SetData(row, *values);
      else
        SetData(row, "Not previewed (external, omitted or 16+ elements).");
    } catch (const std::exception &error) {
      SetData(row, std::string("Unavailable: ") + error.what());
    }
  }

  void AddStruct(const std::string &path, const ol::StructTypeProto &type, size_t depth) {
    if (Stop(path, depth)) {
      uint64_t bits = 0;
      if (payload_ && position_) {
        if (catalogue_ && catalogue_->FixedBitSize(type, bits, &preview_error_))
          Advance(bits);
        else
          position_.reset();
      }
      return;
    }
    if (!active_.emplace(&type).second) {
      Add(Key(path, "Reference"), "Recursive reference (not expanded).");
      return;
    }
    if (type.has_type_ref()) {
      Add(Key(path, "Type reference"), std::to_string(type.type_ref()));
      const auto declaration = declarations_.find(type.type_ref());
      if (declaration != declarations_.end())
        AddStruct(path, *declaration->second, depth + 1);
      else
        Add(Key(path, "Reference"), "Declaration not found.");
    } else {
      if (type.has_name())
        Add(Key(path, "Name"), type.name());
      if (type.has_type_id())
        Add(Key(path, "Type ID"), std::to_string(type.type_id()));
      if (type.has_doc_string())
        Add(Key(path, "Description"), type.doc_string());
      if (type.has_structure()) {
        RowScope scope(*this, Group(path, "Structure"));
        for (const auto &field : type.structure().field()) {
          if (Full())
            break;
          const auto key = Key(path, field.name());
          if (field.has_type())
            AddType(key, field.type(), depth + 1);
          else if (field.has_constant())
            AddConstant(key, field.constant());
          else
            Add(key, "Unknown");
          if (field.has_doc_string())
            Add(Key(key, "Description"), field.doc_string());
        }
      } else if (type.has_array()) {
        RowScope scope(*this, Group(path, "Array"));
        Add(Key(path, "Length"), std::to_string(type.array().dimension()));
        if (type.array().has_element_type()) {
          const auto start = position_;
          uint64_t element_bits = 0, total_bits = 0;
          if (payload_ && position_ && TypeBits(type.array().element_type(), element_bits) &&
              Multiply(element_bits, type.array().dimension(), total_bits)) {
            const auto shown = std::min<uint64_t>(type.array().dimension(), 8);
            for (uint64_t index = 0; index < shown && !Full(); ++index)
              AddType(Key(path, "Element [" + std::to_string(index) + "]"),
                      type.array().element_type(), depth + 1);
            if (shown < type.array().dimension())
              Add(Key(path, "..."), "Showing first " + std::to_string(shown) + " of " +
                                        std::to_string(type.array().dimension()) +
                                        " array elements.");
            position_ = start;
            Advance(total_bits);
          } else {
            position_.reset();
            AddType(Key(path, "Element"), type.array().element_type(), depth + 1);
            if (!payload_)
              position_ = start;
          }
        }
      } else if (type.has_bit_packing()) {
        RowScope scope(*this, Group(path, "Bit packing"));
        Add(Key(path, "Groups"), std::to_string(type.bit_packing().dimension()));
        uint64_t group_bits = 0;
        for (const auto &component : type.bit_packing().component())
          group_bits += component.bit_width();
        uint64_t component_offset = 0;
        for (const auto &component : type.bit_packing().component()) {
          if (Full())
            break;
          const auto row =
              Add(Key(path, component.name()), std::to_string(component.bit_width()) + " bits");
          if (payload_ && type.bit_packing().dimension() != 0) {
            std::string values = "[";
            const auto shown = std::min<uint64_t>(type.bit_packing().dimension(), 15);
            bool available = position_.has_value() && component.bit_width() <= 64;
            for (uint64_t index = 0; available && index < shown; ++index) {
              const auto offset = *position_ + index * group_bits + component_offset;
              if (!Slice(offset, component.bit_width())) {
                available = false;
                break;
              }
              values +=
                  (index ? ", " : "") + std::to_string(ReadBits(offset, component.bit_width()));
            }
            values += "]";
            if (shown < type.bit_packing().dimension())
              values += " ... (first " + std::to_string(shown) + " of " +
                        std::to_string(type.bit_packing().dimension()) + " groups)";
            SetData(row, available
                             ? values
                             : "Unavailable: " +
                                   (preview_error_.empty()
                                        ? "payload is truncated or bit field is wider than 64 bits."
                                        : preview_error_));
          }
          component_offset += component.bit_width();
        }
        uint64_t total_bits = 0;
        if (Multiply(group_bits, type.bit_packing().dimension(), total_bits))
          Advance(total_bits);
        else
          position_.reset();
      } else {
        Add(path.empty() ? "Layout" : path, "Unspecified");
      }
      if (type.has_decoder())
        Add(Key(path, "Decoder"), FunctionName(type.decoder()));
      if (type.has_encoder())
        Add(Key(path, "Encoder"), FunctionName(type.encoder()));
      for (const auto &entry : type.metadata_props()) {
        if (Full())
          break;
        Add(Key(Key(path, "Metadata"), entry.key()), entry.value());
      }
    }
    active_.erase(&type);
  }

  Napi::Object Finish(const std::string &kind) {
    auto result = Napi::Object::New(rows_.Env());
    result.Set("kind", kind);
    result.Set("entries", rows_);
    return result;
  }

  Napi::Array rows_;
  const StructDeclarations &declarations_;
  std::unordered_set<const ol::StructTypeProto *> active_;
  std::optional<uint32_t> parent_;
  const ol::EncodedValueProto *payload_ = nullptr;
  const ol::StructTypeCatalogue *catalogue_ = nullptr;
  std::optional<uint64_t> position_;
  std::string preview_error_;
};

std::string StructPreview(const ol::StructTypeProto &type, const StructDeclarations *declarations) {
  if (declarations && type.has_type_ref()) {
    const auto declaration = declarations->find(type.type_ref());
    if (declaration != declarations->end())
      return Preview(*declaration->second);
  }
  return Preview(type);
}

Napi::Object TensorInformation(Napi::Env env, std::string_view name,
                               const ol::TypeProto *type = nullptr,
                               const StructDeclarations *declarations = nullptr) {
  auto result = Napi::Object::New(env);
  result.Set("name", Text(env, name));
  if (type && (type->has_tensor_type() || type->has_sparse_tensor_type())) {
    const bool sparse = type->has_sparse_tensor_type();
    const auto data_type =
        sparse ? type->sparse_tensor_type().elem_type() : type->tensor_type().elem_type();
    result.Set("type", data_type == ol::TensorProto::UNDEFINED
                           ? Napi::Value(env.Null())
                           : Napi::Value(Text(env, std::string(sparse ? "SPARSE " : "") +
                                                       ol::TensorProto::DataType_Name(data_type))));
    const bool has_shape =
        sparse ? type->sparse_tensor_type().has_shape() : type->tensor_type().has_shape();
    result.Set("shape",
               has_shape ? Napi::Value(Dimensions(env, sparse ? type->sparse_tensor_type().shape()
                                                              : type->tensor_type().shape()))
                         : Napi::Value(env.Null()));
  } else if (type && type->has_opaque_type()) {
    result.Set("type", "OPAQUE " + Preview(type->opaque_type()));
  } else if (type && type->has_struct_type()) {
    result.Set("type", "STRUCT " + StructPreview(type->struct_type(), declarations));
  } else if (type && type->has_type()) {
    result.Set("type", Preview(*type));
  } else {
    result.Set("type", env.Null());
    result.Set("shape", env.Null());
  }
  return result;
}

template <typename Dims>
Napi::Object TensorDistribution(Napi::Env env, ol::TensorProto::DataType data_type,
                                const Dims &dims, const char *kind) {
  ol::core::runtime::Tensor metadata;
  for (const auto dimension : dims)
    metadata.shape.push_back(dimension);
  const auto count = metadata.element_count();
  auto result = Napi::Object::New(env);
  result.Set("kind", kind);
  result.Set("elementCount", Text(env, std::to_string(count)));
  result.Set("logicalSizeBytes",
             ol::FixedBitWidth(data_type)
                 ? Napi::Value(Text(
                       env, std::to_string(ol::core::runtime::PackedByteSize(data_type, count))))
                 : Napi::Value(env.Null()));
  return result;
}

template <typename Dims>
Napi::Object InitializerInformation(Napi::Env env, std::string_view name,
                                    ol::TensorProto::DataType data_type, const Dims &dims,
                                    bool sparse = false) {
  auto result = Napi::Object::New(env);
  result.Set("name", Text(env, name));
  result.Set("type",
             std::string(sparse ? "SPARSE " : "") + ol::TensorProto::DataType_Name(data_type));
  auto shape = Napi::Array::New(env);
  uint32_t index = 0;
  for (const auto dimension : dims) {
    shape.Set(index++, Text(env, std::to_string(dimension)));
  }
  result.Set("shape", shape);
  return result;
}

Napi::Array ConstantOutputs(Napi::Env env, const ol::NodeProto &node) {
  auto result = Napi::Array::New(env);
  if (node.output_size() != 1 || node.output(0).empty())
    return result;
  const auto add = [&](ol::TensorProto::DataType type, const auto &dims, bool sparse = false) {
    auto information = InitializerInformation(env, node.output(0), type, dims, sparse);
    information.Set("distribution",
                    TensorDistribution(env, type, dims, sparse ? "sparse" : "dense"));
    result.Set(0u, information);
  };
  for (const auto &attribute : node.attribute()) {
    if (!attribute.ref_attr_name().empty())
      continue;
    if (attribute.name() == "value" && attribute.has_t())
      add(attribute.t().data_type(), attribute.t().dims());
    else if (attribute.name() == "sparse_value" && attribute.has_sparse_tensor())
      add(attribute.sparse_tensor().values().data_type(), attribute.sparse_tensor().dims(), true);
    else if (attribute.name() == "value_float" && attribute.has_f())
      add(ol::TensorProto::FLOAT, std::vector<int64_t>{});
    else if (attribute.name() == "value_int" && attribute.has_i())
      add(ol::TensorProto::INT64, std::vector<int64_t>{});
    else if (attribute.name() == "value_string" && attribute.has_s())
      add(ol::TensorProto::STRING, std::vector<int64_t>{});
    else if (attribute.name() == "value_floats")
      add(ol::TensorProto::FLOAT, std::vector<int64_t>{attribute.floats_size()});
    else if (attribute.name() == "value_ints")
      add(ol::TensorProto::INT64, std::vector<int64_t>{attribute.ints_size()});
    else if (attribute.name() == "value_strings")
      add(ol::TensorProto::STRING, std::vector<int64_t>{attribute.strings_size()});
  }
  return result;
}

template <typename Values>
Napi::Array ValueNames(Napi::Env env, const Values &values,
                       const std::unordered_set<std::string> &excluded = {}) {
  auto result = Napi::Array::New(env);
  uint32_t index = 0;
  for (const auto &value : values) {
    const std::string_view name = value.name();
    if (!name.empty() && !excluded.contains(std::string(name))) {
      result.Set(index++, Text(env, name));
    }
  }
  return result;
}

template <typename Proto> Napi::Object StorageInformation(Napi::Env env, const Proto &tensor) {
  auto result = Napi::Object::New(env);
  const bool payload_omitted = [&] {
    if constexpr (std::is_same_v<Proto, ol::TensorProto>)
      return IsPayloadOmitted(tensor);
    return false;
  }();
  result.Set("external", !payload_omitted && (tensor.data_location() == ol::TensorProto::EXTERNAL ||
                                              !tensor.external_data().empty()));
  result.Set("location", env.Null());
  for (const auto &entry : tensor.external_data()) {
    if (payload_omitted)
      break;
    if (entry.key() == "location") {
      result.Set("location", Text(env, entry.value()));
    }
  }
  return result;
}

} // namespace

Napi::Value Schema(Napi::Env env, const std::string &op_type, const std::string &domain,
                   int64_t opset) {
  const auto schemas = ol::onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, true);
  const ol::core::schema::LightOpSchema *selected = nullptr;
  for (const auto &schema : schemas) {
    if (Domain(schema.domain()) == domain && schema.since_version() <= opset &&
        (!selected || schema.since_version() > selected->since_version())) {
      selected = &schema;
    }
  }
  if (!selected) {
    return env.Null();
  }
  auto result = Napi::Object::New(env);
  result.Set("sinceVersion", selected->since_version());
  std::stringstream documentation;
  documentation << selected->doc();
  for (const auto &[heading, parameters] :
       {std::pair{"Inputs", &selected->inputs()}, std::pair{"Outputs", &selected->outputs()}}) {
    documentation << "\n\n## " << heading << "\n";
    for (const auto &parameter : *parameters) {
      documentation << "\n### " << parameter.name << " (" << parameter.type << ")\n\n"
                    << parameter.description << "\n";
    }
  }
  auto defaults = Napi::Array::New(env);
  uint32_t index = 0;
  documentation << "\n\n## Attributes\n";
  for (const auto &attribute : selected->attributes()) {
    const auto type = ol::core::schema::AttributeType_Name(attribute.type);
    documentation << "\n"
                  << attribute.name << " (" << type << ")"
                  << (attribute.required ? " [required]" : "") << "\n"
                  << attribute.description << "\n";
    if (!std::holds_alternative<std::monostate>(attribute.default_value)) {
      const auto value = ol::core::schema::AttributeDefaultRepr(attribute.default_value);
      documentation << "Default: " << value << "\n";
      auto item = Napi::Object::New(env);
      item.Set("name", attribute.name);
      item.Set("type", type);
      item.Set("value", value);
      item.Set("default", true);
      defaults.Set(index++, item);
    }
  }
  documentation << "\n\n## Type constraints\n";
  for (const auto &constraint : selected->type_constraints()) {
    documentation << "\n" << constraint.type_param_str << ": ";
    for (const auto type : constraint.allowed_type_strs) {
      documentation << ol::core::schema::ToTypeString(type) << " ";
    }
    documentation << "\n" << constraint.description << "\n";
  }
  result.Set("doc", selected->doc().empty() ? "" : documentation.str());
  result.Set("defaults", defaults);
  return result;
}

Napi::Value GetSchema(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if (info.Length() != 3 || !info[0].IsString() || !info[1].IsString() || !info[2].IsNumber()) {
    Napi::TypeError::New(env, "Expected an operator name, domain and opset version.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  try {
    return Schema(env, info[0].As<Napi::String>().Utf8Value(),
                  Domain(info[1].As<Napi::String>().Utf8Value()),
                  info[2].As<Napi::Number>().Int64Value());
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Object ModelSummary(Napi::Env env, const ol::ModelProto &model,
                          const std::string &file_name) {
  const auto &graph = model.graph();
  StructDeclarations declarations;
  for (const auto &declaration : model.struct_types())
    declarations.emplace(declaration.type_id(), &declaration);
  std::unordered_set<std::string> initializers;
  size_t dense_initializer_count = 0;
  size_t sparse_initializer_count = 0;
  size_t encoded_initializer_count = 0;
  std::map<std::string, Napi::Object> ports;
  for (const auto *values : {&graph.input(), &graph.output()})
    for (const auto &value : *values)
      if (!value.name().empty())
        ports.insert_or_assign(std::string(value.name()),
                               TensorInformation(env, value.name(),
                                                 value.has_type() ? &value.type() : nullptr,
                                                 &declarations));
  for (const auto &value : graph.value_info())
    if (ports.contains(std::string(value.name())))
      ports.insert_or_assign(std::string(value.name()),
                             TensorInformation(env, value.name(),
                                               value.has_type() ? &value.type() : nullptr,
                                               &declarations));
  for (const auto &tensor : graph.initializer()) {
    if (tensor.name().empty())
      continue;
    ++dense_initializer_count;
    initializers.emplace(tensor.name());
    if (ports.contains(std::string(tensor.name())))
      ports.insert_or_assign(
          std::string(tensor.name()),
          InitializerInformation(env, tensor.name(), tensor.data_type(), tensor.dims()));
  }
  for (const auto &tensor : graph.sparse_initializer()) {
    if (!tensor.has_values() || tensor.values().name().empty())
      continue;
    ++sparse_initializer_count;
    const std::string name(tensor.values().name());
    initializers.emplace(name);
    if (ports.contains(name))
      ports.insert_or_assign(name, InitializerInformation(env, name, tensor.values().data_type(),
                                                          tensor.dims(), true));
  }
  for (const auto &tensor : graph.extended_initializer()) {
    if (!tensor.has_encoded_value() || tensor.encoded_value().name().empty())
      continue;
    ++encoded_initializer_count;
    const auto &value = tensor.encoded_value();
    const std::string name(value.name());
    initializers.emplace(name);
    if (ports.contains(name)) {
      auto information = TensorInformation(
          env, name, value.has_logical_type() ? &value.logical_type() : nullptr, &declarations);
      if (information.Get("type").IsNull())
        information.Set("type", "ENCODED");
      ports.insert_or_assign(name, information);
    }
  }
  auto tensors = Napi::Array::New(env);
  for (const auto &[name, tensor] : ports)
    if (!initializers.contains(name) ||
        std::any_of(graph.output().begin(), graph.output().end(),
                    [&](const auto &output) { return output.name() == name; }))
      tensors.Set(tensors.Length(), tensor);
  auto result = Napi::Object::New(env);
  result.Set("name", graph.name().empty() ? Text(env, file_name) : Text(env, graph.name()));
  result.Set("nodeCount", graph.node_size());
  result.Set("opsets", OpsetImports(env, model.opset_import()));
  result.Set("initializerCount", Napi::Number::New(env, initializers.size()));
  result.Set("denseInitializerCount", Napi::Number::New(env, dense_initializer_count));
  result.Set("sparseInitializerCount", Napi::Number::New(env, sparse_initializer_count));
  result.Set("encodedInitializerCount", Napi::Number::New(env, encoded_initializer_count));
  result.Set("functionCount", model.functions_size());
  using OperatorKey = std::pair<std::string, std::string>;
  const auto operator_hash = [](const OperatorKey &key) {
    const auto domain_hash = std::hash<std::string>{}(key.first);
    const auto type_hash = std::hash<std::string>{}(key.second);
    return domain_hash ^ (type_hash + 0x9e3779b9 + (domain_hash << 6) + (domain_hash >> 2));
  };
  std::unordered_map<OperatorKey, size_t, decltype(operator_hash)> operator_counts(0,
                                                                                   operator_hash);
  for (const auto &node : graph.node())
    ++operator_counts[{Domain(node.domain()), std::string(node.op_type())}];
  auto operators = Napi::Array::New(env);
  for (const auto &[key, count] : operator_counts) {
    auto entry = Napi::Object::New(env);
    entry.Set("domain", Text(env, key.first));
    entry.Set("opType", Text(env, key.second));
    entry.Set("count", Napi::Number::New(env, count));
    operators.Set(operators.Length(), entry);
  }
  result.Set("operatorTypes", operators);
  result.Set("statisticsAvailable", true);
  result.Set("inputs", ValueNames(env, graph.input(), initializers));
  result.Set("outputs", ValueNames(env, graph.output()));
  result.Set("tensors", tensors);
  return result;
}

/**
 * Projects only the selected graph, inheriting outer tensor/initializer metadata
 * unless shadowed locally. Child graphs become links, not recursive snapshots.
 */
Napi::Object ModelGraph(Napi::Env env, const ol::ModelProto &model, const std::string &file_name,
                        const GraphPath &path) {
  const auto selection = GraphScopes(model, path);
  const auto &scopes = selection.scopes;
  const auto &graph = *scopes.back();
  StructDeclarations struct_declarations;
  EncodedPreviewCatalogue preview_catalogue;
  struct_declarations.reserve(model.struct_types().size());
  for (const auto &declaration : model.struct_types())
    struct_declarations.emplace(declaration.type_id(), &declaration);
  std::map<std::string, Napi::Object> tensor_information;
  std::map<std::string, int64_t> opsets;
  for (const auto &opset :
       selection.function ? selection.function->opset_import() : model.opset_import()) {
    opsets[Domain(opset.domain())] = opset.version();
  }
  std::map<std::pair<std::string, std::string>, Napi::Value> schemas;
  std::map<std::tuple<std::string, std::string, std::string>, int> functions;
  for (int index = 0; index < model.functions_size(); ++index) {
    const auto &function = model.functions(index);
    functions.emplace(std::tuple{Domain(function.domain()), std::string(function.name()),
                                 std::string(function.overload())},
                      index);
  }
  std::unordered_set<std::string> initializers;
  std::map<std::string, Napi::Object> parameters;
  auto dense_initializers = Napi::Array::New(env);
  auto sparse_initializers = Napi::Array::New(env);
  auto encoded_initializers = Napi::Array::New(env);
  std::unordered_set<std::string> local_names;
  for (size_t scope_index = selection.capture_start; scope_index < scopes.size(); ++scope_index) {
    const auto *scope = scopes[scope_index];
    local_names.clear();
    for (const auto &value : scope->input())
      local_names.emplace(value.name());
    for (const auto &value : scope->initializer())
      local_names.emplace(value.name());
    for (const auto &value : scope->sparse_initializer())
      local_names.emplace(value.values().name());
    for (const auto &value : scope->extended_initializer())
      if (value.has_encoded_value())
        local_names.emplace(value.encoded_value().name());
    for (const auto &node : scope->node())
      for (const auto &name : node.output())
        local_names.emplace(name);
    for (const auto &name : local_names) {
      tensor_information.erase(name);
      parameters.erase(name);
      initializers.erase(name);
    }
    for (const auto *values : {&scope->input(), &scope->output(), &scope->value_info()})
      for (const auto &value : *values)
        if (!value.name().empty())
          tensor_information.insert_or_assign(
              std::string(value.name()),
              TensorInformation(env, value.name(), value.has_type() ? &value.type() : nullptr,
                                &struct_declarations));
    const auto &graph = *scope;
    for (const auto &tensor : graph.initializer()) {
      if (!tensor.name().empty()) {
        const std::string name(tensor.name());
        initializers.emplace(name);
        tensor_information.insert_or_assign(
            name, InitializerInformation(env, name, tensor.data_type(), tensor.dims()));
        auto parameter = InitializerInformation(env, name, tensor.data_type(), tensor.dims());
        parameter.Set("distribution",
                      TensorDistribution(env, tensor.data_type(), tensor.dims(), "dense"));
        parameter.Set("storage", StorageInformation(env, tensor));
        if (const auto value = TinyTensorValues(tensor))
          parameter.Set("value", *value);
        parameters.emplace(name, parameter);
        if (scope_index == scopes.size() - 1)
          dense_initializers.Set(dense_initializers.Length(), parameter);
      }
    }
    for (const auto &tensor : graph.sparse_initializer()) {
      if (tensor.has_values() && !tensor.values().name().empty()) {
        initializers.emplace(std::string_view(tensor.values().name()));
        const std::string name(tensor.values().name());
        tensor_information.insert_or_assign(
            name,
            InitializerInformation(env, name, tensor.values().data_type(), tensor.dims(), true));
        auto parameter =
            InitializerInformation(env, name, tensor.values().data_type(), tensor.dims(), true);
        parameter.Set("distribution", TensorDistribution(env, tensor.values().data_type(),
                                                         tensor.dims(), "sparse"));
        parameter.Set("storage", StorageInformation(env, tensor.values()));
        parameter.Set("indicesStorage", StorageInformation(env, tensor.indices()));
        const auto values = TinyTensorValues(tensor.values());
        const auto indices = TinyTensorValues(tensor.indices());
        if (values && indices)
          parameter.Set("value", "values: " + *values + "; indices: " + *indices);
        parameters.emplace(name, parameter);
        if (scope_index == scopes.size() - 1)
          sparse_initializers.Set(sparse_initializers.Length(), parameter);
      }
    }
    for (const auto &initializer : graph.extended_initializer()) {
      if (!initializer.has_encoded_value() || initializer.encoded_value().name().empty())
        continue;
      const auto &value = initializer.encoded_value();
      const std::string name(value.name());
      initializers.emplace(name);
      auto information =
          TensorInformation(env, name, value.has_logical_type() ? &value.logical_type() : nullptr,
                            &struct_declarations);
      if (information.Get("type").IsNull())
        information.Set("type", "ENCODED");
      tensor_information.insert_or_assign(name, information);
      auto parameter = Napi::Object::New(env);
      const auto keys = information.GetPropertyNames();
      for (uint32_t index = 0; index < keys.Length(); ++index) {
        const auto key = keys.Get(index);
        parameter.Set(key, information.Get(key));
      }
      parameter.Set("storage", StorageInformation(env, value));
      auto distribution = Napi::Object::New(env);
      distribution.Set("kind", "encoded");
      distribution.Set("elementCount", env.Null());
      distribution.Set("logicalSizeBytes", env.Null());
      if (value.has_logical_type() && value.logical_type().has_tensor_type()) {
        const auto &type = value.logical_type().tensor_type();
        std::vector<int64_t> dims;
        bool concrete = type.has_shape();
        for (const auto &dimension : type.shape().dim()) {
          concrete &= dimension.has_dim_value() && dimension.dim_value() >= 0;
          dims.push_back(dimension.dim_value());
        }
        if (concrete)
          distribution = TensorDistribution(env, type.elem_type(), dims, "encoded");
      }
      parameter.Set("distribution", distribution);
      if (value.has_logical_type())
        parameter.Set("typeDetails",
                      TypeTable(env, struct_declarations).Logical(value.logical_type()));
      if (value.has_affine() || value.has_struct_type())
        parameter.Set(
            "encodingDetails",
            TypeTable(env, struct_declarations).Encoding(value, model, preview_catalogue));
      if (value.has_affine())
        parameter.Set("encoding", "AFFINE " + Preview(value.affine()));
      else if (value.has_struct_type())
        parameter.Set("encoding",
                      "STRUCT " + StructPreview(value.struct_type(), &struct_declarations));
      parameters.emplace(name, parameter);
      if (scope_index == scopes.size() - 1)
        encoded_initializers.Set(encoded_initializers.Length(), parameter);
    }
  }

  auto nodes = Napi::Array::New(env);
  auto inputs = ValueNames(env, graph.input(), initializers);
  std::unordered_set<std::string> visible_names = local_names;
  std::unordered_set<std::string> captured_names;
  for (const auto *values : {&graph.output(), &graph.value_info()})
    for (const auto &value : *values)
      visible_names.emplace(value.name());
  const auto add_capture = [&](const std::string &name) {
    visible_names.emplace(name);
    if (name.empty() || local_names.contains(name) || initializers.contains(name) ||
        !captured_names.emplace(name).second)
      return;
    inputs.Set(inputs.Length(), Text(env, name));
  };
  for (const auto &value : graph.output())
    if (!local_names.contains(std::string(value.name())))
      add_capture(std::string(value.name()));
  uint32_t index = 0;
  for (const auto &node : graph.node()) {
    if (node.op_type().empty()) {
      throw std::runtime_error("Node " + std::to_string(index + 1) + " has no ONNX operator type.");
    }
    auto result = Napi::Object::New(env);
    result.Set("id", "node-" + std::to_string(index));
    result.Set("name", node.name().empty() ? Text(env, std::string(node.op_type()) + "_" +
                                                           std::to_string(index + 1))
                                           : Text(env, node.name()));
    result.Set("opType", Text(env, node.op_type()));
    const auto domain = Domain(node.domain());
    result.Set("domain", domain);
    const auto opset = opsets.find(domain);
    result.Set("opset", opset == opsets.end()
                            ? Napi::Value(env.Null())
                            : Napi::Value(Text(env, std::to_string(opset->second))));
    const auto key = std::pair{domain, std::string(node.op_type())};
    if (!schemas.contains(key)) {
      schemas.emplace(key, opset == opsets.end() ? env.Null()
                                                 : Schema(env, key.second, domain, opset->second));
    }
    const auto function =
        functions.find({domain, std::string(node.op_type()), std::string(node.overload())});
    result.Set("schema", function == functions.end() ? schemas.at(key) : Napi::Value(env.Null()));
    if (function != functions.end()) {
      const auto &definition = model.functions(function->second);
      auto link = Napi::Object::New(env);
      auto function_path = path;
      function_path.push_back({static_cast<int>(index), 0, 0, function->second});
      link.Set("name", FunctionName(definition));
      link.Set("path", PathValue(env, function_path));
      link.Set("nodeCount", definition.node_size());
      result.Set("function", link);
    }
    if (node.op_type() == "Constant" && domain == "ai.onnx" && function == functions.end())
      result.Set("storedOutputs", ConstantOutputs(env, node));
    auto attributes = Napi::Array::New(env);
    uint32_t attribute_index = 0;
    for (const auto &attribute : node.attribute()) {
      auto item = Napi::Object::New(env);
      item.Set("name", Text(env, attribute.name()));
      item.Set("type", ol::AttributeProto::AttributeType_Name(attribute.type()));
      item.Set("value", AttributePreview(attribute));
      item.Set("default", false);
      if (attribute.type() == ol::AttributeProto::GRAPH ||
          attribute.type() == ol::AttributeProto::GRAPHS) {
        auto children = Napi::Array::New(env);
        const int count = attribute.type() == ol::AttributeProto::GRAPH
                              ? (attribute.has_g() ? 1 : 0)
                              : attribute.graphs_size();
        for (int child_index = 0; child_index < count; ++child_index) {
          const auto &child = attribute.type() == ol::AttributeProto::GRAPH
                                  ? attribute.g()
                                  : attribute.graphs(child_index);
          auto link = Napi::Object::New(env);
          auto child_path = path;
          child_path.push_back(
              {static_cast<int>(index), static_cast<int>(attribute_index), child_index});
          link.Set("name",
                   child.name().empty() ? Text(env, attribute.name()) : Text(env, child.name()));
          link.Set("path", PathValue(env, child_path));
          link.Set("nodeCount", child.node_size());
          children.Set(child_index, link);
        }
        item.Set("graphs", children);
      }
      attributes.Set(attribute_index++, item);
    }
    result.Set("attributes", attributes);
    auto weights = Napi::Array::New(env);
    uint32_t weight_index = 0;
    std::vector<std::string> captures;
    const auto all_inputs = ol::core::graph::CollectNodeInputs(node);
    for (const auto &input : all_inputs) {
      add_capture(input);
      if (!tensor_information.contains(input))
        tensor_information.emplace(input, TensorInformation(env, input));
      if (std::find(node.input().begin(), node.input().end(), input) == node.input().end())
        captures.push_back(input);
    }
    const auto add_weight = [&](std::string_view input) {
      const auto parameter = parameters.find(std::string(input));
      if (parameter != parameters.end()) {
        weights.Set(weight_index++, parameter->second);
      }
    };
    for (const auto &input : node.input())
      add_weight(input);
    for (const auto &input : captures)
      add_weight(input);
    result.Set("parameters", weights);
    for (const auto *ports : {&node.input(), &node.output()}) {
      for (const auto &port : *ports) {
        if (!port.empty() && !tensor_information.contains(std::string(port))) {
          tensor_information.emplace(std::string(port), TensorInformation(env, port));
        }
      }
    }
    result.Set("inputs", Ports(env, node.input()));
    result.Set("captures", Ports(env, captures));
    result.Set("outputs", Ports(env, node.output()));
    nodes.Set(index++, result);
  }
  auto result = Napi::Object::New(env);
  result.Set("name", graph.name().empty() ? Text(env, file_name) : Text(env, graph.name()));
  auto breadcrumbs = Napi::Array::New(env);
  GraphPath prefix;
  for (uint32_t depth = 0; depth < scopes.size(); ++depth) {
    auto crumb = Napi::Object::New(env);
    crumb.Set("name", scopes[depth]->name().empty() ? Text(env, depth ? "Subgraph" : file_name)
                                                    : Text(env, scopes[depth]->name()));
    crumb.Set("path", PathValue(env, prefix));
    breadcrumbs.Set(depth, crumb);
    if (depth < path.size())
      prefix.push_back(path[depth]);
  }
  result.Set("breadcrumbs", breadcrumbs);
  result.Set("opsets", OpsetImports(env, selection.function ? selection.function->opset_import()
                                                            : model.opset_import()));
  result.Set("nodes", nodes);
  auto local_functions = Napi::Array::New(env);
  if (path.empty()) {
    for (int index = 0; index < model.functions_size(); ++index) {
      const auto &function = model.functions(index);
      auto link = Napi::Object::New(env);
      link.Set("name", FunctionName(function));
      link.Set("path", PathValue(env, {{-1, 0, 0, index}}));
      link.Set("nodeCount", function.node_size());
      local_functions.Set(local_functions.Length(), link);
    }
  }
  result.Set("localFunctions", local_functions);
  result.Set("inputs", inputs);
  result.Set("outputs", ValueNames(env, graph.output()));
  result.Set("initializers", dense_initializers);
  result.Set("sparseInitializers", sparse_initializers);
  result.Set("encodedInitializers", encoded_initializers);
  result.Set("initializerCount",
             Napi::Number::New(env, std::count_if(initializers.begin(), initializers.end(),
                                                  [&](const auto &name) {
                                                    return visible_names.contains(name);
                                                  })));
  auto tensors = Napi::Array::New(env);
  uint32_t tensor_index = 0;
  for (const auto &[name, tensor] : tensor_information) {
    if (visible_names.contains(name))
      tensors.Set(tensor_index++, tensor);
  }
  result.Set("tensors", tensors);
  return result;
}

} // namespace onnx_light_viewer
