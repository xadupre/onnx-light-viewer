/** @file Graph, initializer and operator-schema snapshots for JavaScript. */
#include "addon.h"
#include "graph.h"
#include "tensor_preview.h"
#include <algorithm>
#include <map>
#include <onnx_core/graph/graph_manipulations.h>
#include <onnx_op/operator_sets.h>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace onnx_light_viewer {
namespace {

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
        parameter.Set("storage", StorageInformation(env, tensor));
        if (const auto value = TinyTensorValues(tensor))
          parameter.Set("value", *value);
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
        const auto values = TinyTensorValues(tensor.values());
        const auto indices = TinyTensorValues(tensor.indices());
        if (values && indices)
          parameter.Set("value", "values: " + *values + "; indices: " + *indices);
        parameters.emplace(name, parameter);
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
      if (value.has_affine())
        parameter.Set("encoding", "AFFINE " + Preview(value.affine()));
      else if (value.has_struct_type())
        parameter.Set("encoding",
                      "STRUCT " + StructPreview(value.struct_type(), &struct_declarations));
      parameters.emplace(name, parameter);
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
  result.Set("nodes", nodes);
  result.Set("inputs", inputs);
  result.Set("outputs", ValueNames(env, graph.output()));
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
