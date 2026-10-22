#include <napi.h>
#include <onnx_core/shapes/shape_inference.h>
#include <onnx_extensions/shapes/dispatch_table.h>
#include <onnx_op/operator_sets.h>
#include <onnx_proto/onnx.h>

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace ol = ONNX_LIGHT_NAMESPACE;

namespace {

Napi::String Text(Napi::Env env, std::string_view value) {
  return Napi::String::New(env, value.empty() ? "" : value.data(), value.size());
}

std::string Domain(std::string_view domain) {
  return domain.empty() ? "ai.onnx" : std::string(domain);
}

template <typename Proto> std::string Preview(const Proto &value, size_t limit = 4096) {
  ol::utils::PrintOptions options;
  options.skip_raw_data = true;
  options.max_short_repr_length = limit;
  std::stringstream stream;
  value.PrintToStringStream(stream, options);
  ol::utils::enforce_short_repr_length(stream, options);
  return stream.str();
}

template <typename Values> size_t PreviewValues(const Values &values, Values &preview) {
  for (size_t index = 0; index < values.size() && index < 8; ++index) {
    preview.push_back(values[index]);
  }
  return values.size() > 8 ? values.size() : 0;
}

std::string PreviewCount(size_t count) {
  return count ? " ... (" + std::to_string(count) + " elements)" : "";
}

std::string TensorPreview(const ol::TensorProto &tensor) {
  ol::TensorProto preview;
  preview.set_name(tensor.name());
  preview.set_data_type(tensor.data_type());
  for (const auto dimension : tensor.dims()) {
    preview.add_dims(dimension);
  }
  for (const auto &entry : tensor.external_data()) {
    *preview.add_external_data() = entry;
  }
  if (tensor.has_data_location()) {
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

std::string AttributePreview(const ol::AttributeProto &attribute) {
  if (!attribute.ref_attr_name().empty())
    return Preview(attribute, 512);
  ol::AttributeProto preview;
  preview.set_name(attribute.name());
  preview.set_type(attribute.type());
  size_t count = 0;
  switch (attribute.type()) {
  case ol::AttributeProto::FLOATS:
    count = PreviewValues(attribute.floats(), preview.ref_floats());
    break;
  case ol::AttributeProto::INTS:
    count = PreviewValues(attribute.ints(), preview.ref_ints());
    break;
  case ol::AttributeProto::STRINGS:
    count = PreviewValues(attribute.strings(), preview.ref_strings());
    break;
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
  return Preview(preview, 512) + PreviewCount(count);
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

Napi::Object TensorInformation(Napi::Env env, std::string_view name,
                               const ol::TypeProto *type = nullptr) {
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
  } else if (type && type->has_type()) {
    result.Set("type", Preview(*type));
  } else {
    result.Set("type", env.Null());
    result.Set("shape", env.Null());
  }
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
    documentation << "\n\n" << heading << "\n";
    for (const auto &parameter : *parameters) {
      documentation << "\n"
                    << parameter.name << " (" << parameter.type << ")\n"
                    << parameter.description << "\n";
    }
  }
  auto defaults = Napi::Array::New(env);
  uint32_t index = 0;
  documentation << "\n\nAttributes\n";
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
  documentation << "\n\nType constraints\n";
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

Napi::Object StorageInformation(Napi::Env env, const ol::TensorProto &tensor) {
  auto result = Napi::Object::New(env);
  result.Set("external", tensor.data_location() == ol::TensorProto::EXTERNAL ||
                             !tensor.external_data().empty());
  result.Set("location", env.Null());
  for (const auto &entry : tensor.external_data()) {
    if (entry.key() == "location") {
      result.Set("location", Text(env, entry.value()));
    }
  }
  return result;
}

Napi::Object ModelGraph(Napi::Env env, const ol::ModelProto &model, const std::string &file_name) {
  const auto &graph = model.graph();
  std::map<std::string, Napi::Object> tensor_information;
  for (const auto *values : {&graph.input(), &graph.output(), &graph.value_info()}) {
    for (const auto &value : *values) {
      if (!value.name().empty()) {
        tensor_information.insert_or_assign(
            std::string(value.name()),
            TensorInformation(env, value.name(), value.has_type() ? &value.type() : nullptr));
      }
    }
  }
  std::map<std::string, int64_t> opsets;
  for (const auto &opset : model.opset_import()) {
    opsets[Domain(opset.domain())] = opset.version();
  }
  std::map<std::pair<std::string, std::string>, Napi::Value> schemas;
  std::unordered_set<std::string> initializers;
  std::map<std::string, Napi::Object> parameters;
  for (const auto &tensor : graph.initializer()) {
    if (!tensor.name().empty()) {
      const std::string name(tensor.name());
      initializers.emplace(name);
      tensor_information.insert_or_assign(
          name, InitializerInformation(env, name, tensor.data_type(), tensor.dims()));
      auto parameter = InitializerInformation(env, name, tensor.data_type(), tensor.dims());
      parameter.Set("storage", StorageInformation(env, tensor));
      parameters.emplace(name, parameter);
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
      parameter.Set("storage", StorageInformation(env, tensor.values()));
      parameter.Set("indicesStorage", StorageInformation(env, tensor.indices()));
      parameters.emplace(name, parameter);
    }
  }

  auto nodes = Napi::Array::New(env);
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
    result.Set("schema", schemas.at(key));
    auto attributes = Napi::Array::New(env);
    uint32_t attribute_index = 0;
    for (const auto &attribute : node.attribute()) {
      auto item = Napi::Object::New(env);
      item.Set("name", Text(env, attribute.name()));
      item.Set("type", ol::AttributeProto::AttributeType_Name(attribute.type()));
      item.Set("value", AttributePreview(attribute));
      item.Set("default", false);
      attributes.Set(attribute_index++, item);
    }
    result.Set("attributes", attributes);
    auto weights = Napi::Array::New(env);
    uint32_t weight_index = 0;
    for (const auto &input : node.input()) {
      const auto parameter = parameters.find(std::string(input));
      if (parameter != parameters.end()) {
        weights.Set(weight_index++, parameter->second);
      }
    }
    result.Set("parameters", weights);
    for (const auto *ports : {&node.input(), &node.output()}) {
      for (const auto &port : *ports) {
        if (!port.empty() && !tensor_information.contains(std::string(port))) {
          tensor_information.emplace(std::string(port), TensorInformation(env, port));
        }
      }
    }
    result.Set("inputs", Ports(env, node.input()));
    result.Set("outputs", Ports(env, node.output()));
    nodes.Set(index++, result);
  }
  auto result = Napi::Object::New(env);
  result.Set("name", graph.name().empty() ? Text(env, file_name) : Text(env, graph.name()));
  result.Set("nodes", nodes);
  result.Set("inputs", ValueNames(env, graph.input(), initializers));
  result.Set("outputs", ValueNames(env, graph.output()));
  result.Set("initializerCount", Napi::Number::New(env, initializers.size()));
  auto tensors = Napi::Array::New(env);
  uint32_t tensor_index = 0;
  for (const auto &[name, tensor] : tensor_information) {
    tensors.Set(tensor_index++, tensor);
  }
  result.Set("tensors", tensors);
  return result;
}

void ReadModel(ol::ModelProto &model, const Napi::Uint8Array &bytes, bool skip_raw_data) {
  ol::utils::StringStream stream(bytes.Data(), bytes.ByteLength());
  ol::ParseOptions options;
  options.skip_raw_data = skip_raw_data;
  options.no_copy = !skip_raw_data;
  if (!model.ParseFromStream(stream, options)) {
    throw std::runtime_error("Could not decode the ONNX model.");
  }
  if (!model.has_graph() || !model.has_ir_version() || model.ir_version() <= 0) {
    throw std::runtime_error("This file does not contain an ONNX model graph.");
  }
}

void InferShapes(ol::ModelProto &model) {
  ol::onnx_shapes::RegisterShapeFunctions();
  ol::core::shapes::InferShapesModel(model);
}

Napi::Value DecodeGraph(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if ((info.Length() != 2 && info.Length() != 3) || !info[0].IsTypedArray() ||
      info[0].As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array || !info[1].IsString() ||
      (info.Length() == 3 && !info[2].IsBoolean())) {
    Napi::TypeError::New(env, "Expected a Uint8Array, a file name and an optional inference flag.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  // C++ exceptions must not unwind through the Node-API boundary.
  try {
    const bool infer_shapes = info.Length() == 3 && info[2].As<Napi::Boolean>().Value();
    ol::ModelProto model;
    ReadModel(model, info[0].As<Napi::Uint8Array>(), !infer_shapes);
    if (infer_shapes)
      InferShapes(model);
    return ModelGraph(env, model, info[1].As<Napi::String>().Utf8Value());
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

struct StoredModel {
  // Tensor buffers borrow this array for the lifetime of the native model.
  Napi::Reference<Napi::Uint8Array> source;
  ol::ModelProto model;
  std::string file_name;
};

Napi::Value LoadModel(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if (info.Length() != 2 || !info[0].IsTypedArray() ||
      info[0].As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array || !info[1].IsString()) {
    Napi::TypeError::New(env, "Expected a Uint8Array and a file name.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  try {
    auto model = std::make_shared<StoredModel>();
    model->source = Napi::Persistent(info[0].As<Napi::Uint8Array>());
    model->file_name = info[1].As<Napi::String>().Utf8Value();
    ReadModel(model->model, model->source.Value(), false);
    auto result = Napi::Object::New(env);
    result.Set(
        "graph", Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
          const auto env = call.Env();
          if (call.Length() > 1 ||
              (call.Length() == 1 && !call[0].IsUndefined() && !call[0].IsBoolean())) {
            Napi::TypeError::New(env, "Expected an optional inference flag.")
                .ThrowAsJavaScriptException();
            return env.Undefined();
          }
          try {
            if (call.Length() == 1 && call[0].IsBoolean() && call[0].As<Napi::Boolean>().Value()) {
              InferShapes(model->model);
            }
            return ModelGraph(env, model->model, model->file_name);
          } catch (const std::exception &error) {
            Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
            return env.Undefined();
          }
        }));
    return result;
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Object Initialize(Napi::Env env, Napi::Object exports) {
  exports.Set("decodeGraph", Napi::Function::New(env, DecodeGraph));
  exports.Set("getSchema", Napi::Function::New(env, GetSchema));
  exports.Set("loadModel", Napi::Function::New(env, LoadModel));
  return exports;
}

} // namespace

NODE_API_MODULE(onnx_light_viewer, Initialize)
