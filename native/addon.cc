/**
 * @file
 * Node-API bridge to onnx-light. Retained models keep their backing buffers
 * alive; JavaScript receives graph and analysis snapshots.
 * See docs/design.rst for ownership, graph paths and the message flow.
 */
#include <napi.h>
#include <onnx_core/backend_test/test_case.h>
#include <onnx_core/compute/constant_info.h>
#include <onnx_core/compute/result_lifetime.h>
#include <onnx_core/compute/value_tags.h>
#include <onnx_core/graph/graph_manipulations.h>
#include <onnx_core/shapes/shape_inference.h>
#include <onnx_extensions/shapes/dispatch_table.h>
#include <onnx_op/operator_sets.h>
#include <onnx_proto/onnx.h>
#include <onnx_proto/onnx_verify.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace ol = ONNX_LIGHT_NAMESPACE;

namespace {

template <typename Action> auto RunAlgorithm(const char *name, Action action) {
  try {
    return action();
  } catch (const std::exception &error) {
    throw std::runtime_error(std::string(name) + ": " + error.what());
  }
}

struct GraphStep {
  int node;
  int attribute;
  int graph;
};
using GraphPath = std::vector<GraphStep>;

GraphPath ReadPath(const Napi::Value &value) {
  GraphPath path;
  if (value.IsUndefined())
    return path;
  if (!value.IsArray())
    throw std::runtime_error("Expected an array for the subgraph path.");
  const auto steps = value.As<Napi::Array>();
  for (uint32_t index = 0; index < steps.Length(); ++index) {
    const auto entry = steps.Get(index);
    if (!entry.IsObject() || entry.IsNull())
      throw std::runtime_error("Invalid subgraph path step.");
    const auto object = entry.As<Napi::Object>();
    const auto read = [&](const char *key) {
      const auto item = object.Get(key);
      if (!item.IsNumber())
        throw std::runtime_error("Invalid subgraph path index.");
      const double number = item.As<Napi::Number>().DoubleValue();
      if (!std::isfinite(number) || number < 0 || std::floor(number) != number ||
          number > std::numeric_limits<int>::max())
        throw std::runtime_error("Invalid subgraph path index.");
      return static_cast<int>(number);
    };
    path.push_back({read("node"), read("attribute"), read("graph")});
  }
  return path;
}

Napi::Array PathValue(Napi::Env env, const GraphPath &path) {
  auto result = Napi::Array::New(env);
  for (uint32_t index = 0; index < path.size(); ++index) {
    auto step = Napi::Object::New(env);
    step.Set("node", path[index].node);
    step.Set("attribute", path[index].attribute);
    step.Set("graph", path[index].graph);
    result.Set(index, step);
  }
  return result;
}

const ol::GraphProto &ChildGraph(const ol::GraphProto &parent, const GraphStep &step) {
  if (step.node >= parent.node_size() || step.attribute >= parent.node(step.node).attribute_size())
    throw std::runtime_error("Unknown subgraph path.");
  const auto &attribute = parent.node(step.node).attribute(step.attribute);
  if (attribute.type() == ol::AttributeProto::GRAPH && attribute.has_g() && step.graph == 0)
    return attribute.g();
  if (attribute.type() == ol::AttributeProto::GRAPHS && step.graph < attribute.graphs_size())
    return attribute.graphs(step.graph);
  throw std::runtime_error("Unknown subgraph path.");
}

/** Returns borrowed graph pointers, ordered from the root to the selected scope. */
std::vector<const ol::GraphProto *> GraphScopes(const ol::ModelProto &model,
                                                const GraphPath &path) {
  std::vector<const ol::GraphProto *> scopes{&model.graph()};
  for (const auto &step : path)
    scopes.push_back(&ChildGraph(*scopes.back(), step));
  return scopes;
}

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
  case ol::AttributeProto::GRAPH:
    return std::to_string(attribute.g().node_size()) + " nodes";
  case ol::AttributeProto::GRAPHS:
    return std::to_string(attribute.graphs_size()) + " subgraphs";
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

template <typename Proto> Napi::Array MetadataEntries(Napi::Env env, const Proto &proto) {
  auto result = Napi::Array::New(env);
  uint32_t index = 0;
  for (const auto &entry : proto.metadata_props()) {
    auto item = Napi::Object::New(env);
    item.Set("key", Text(env, entry.key()));
    item.Set("value", Text(env, entry.value()));
    result.Set(index++, item);
  }
  return result;
}

/** Seeds captured constants from enclosing scopes without modifying the model. */
Napi::Object ModelMetadata(Napi::Env env, const ol::ModelProto &model, const GraphPath &path) {
  const auto scopes = GraphScopes(model, path);
  RunAlgorithm("VerifyModel", [&] { ol::VerifyModel(model); });
  const auto &graph = *scopes.back();
  std::unordered_set<std::string> outer_constants;
  for (size_t index = 0; index + 1 < scopes.size(); ++index)
    outer_constants = RunAlgorithm("InferConstants (enclosing graph)", [&] {
                        return ol::core::compute::InferConstants(*scopes[index], outer_constants);
                      }).first;
  const auto [constants, constant_nodes] = RunAlgorithm(
      "InferConstants", [&] { return ol::core::compute::InferConstants(graph, outer_constants); });
  const auto [tags, node_tags] = RunAlgorithm(
      "InferValueAndNodeTags", [&] { return ol::core::compute::InferValueAndNodeTags(graph); });
  const auto lifetimes = RunAlgorithm("ComputeResultLifetimeInfo", [&] {
    return ol::core::compute::ComputeResultLifetimeInfo(graph, false);
  });
  std::map<std::string, Napi::Object> values;
  const auto add_value = [&](std::string_view name) {
    const std::string key(name);
    if (key.empty() || values.contains(key))
      return;
    auto item = Napi::Object::New(env);
    item.Set("name", Text(env, name));
    item.Set("entries", Napi::Array::New(env));
    item.Set("constant", constants.contains(key));
    const auto tag = tags.find(key);
    item.Set("tag", tag == tags.end() || tag->second.empty() ? Napi::Value(env.Null())
                                                             : Napi::Value(Text(env, tag->second)));
    values.emplace(key, item);
  };
  const auto add_entries = [&](const auto &proto) {
    add_value(proto.name());
    if (proto.name().empty())
      return;
    auto entries = values.at(std::string(proto.name())).Get("entries").As<Napi::Array>();
    const auto additional = MetadataEntries(env, proto);
    for (uint32_t i = 0; i < additional.Length(); ++i)
      entries.Set(entries.Length(), additional.Get(i));
  };
  for (const auto *declarations : {&graph.input(), &graph.output(), &graph.value_info()})
    for (const auto &value : *declarations)
      add_entries(value);
  for (const auto &value : graph.initializer())
    add_entries(value);
  for (const auto &value : graph.sparse_initializer())
    add_entries(value.values());
  auto nodes = Napi::Array::New(env);
  uint32_t index = 0;
  for (const auto &node : graph.node()) {
    for (const auto &name : ol::core::graph::CollectNodeInputs(node))
      add_value(name);
    std::vector<std::string> unused;
    for (const auto &name : node.output()) {
      add_value(name);
      if (!name.empty() && !lifetimes.keep.contains(std::string(name)) &&
          !lifetimes.last_use.contains(std::string(name)))
        unused.emplace_back(name);
    }
    auto item = Napi::Object::New(env);
    item.Set("id", "node-" + std::to_string(index));
    item.Set("entries", MetadataEntries(env, node));
    item.Set("constant", constant_nodes.at(index) == ol::core::compute::ConstantInfo::kConstant);
    item.Set("tag", node_tags.at(index).empty() ? Napi::Value(env.Null())
                                                : Napi::Value(Text(env, node_tags.at(index))));
    item.Set("releaseAfter", Ports(env, lifetimes[index].release_after));
    item.Set("lastUse", Ports(env, lifetimes[index].not_used_after));
    item.Set("unusedOutputs", Ports(env, unused));
    nodes.Set(index++, item);
  }
  auto value_list = Napi::Array::New(env);
  index = 0;
  for (const auto &[name, value] : values)
    value_list.Set(index++, value);
  auto result = Napi::Object::New(env);
  result.Set("model", MetadataEntries(env, model));
  result.Set("graph", MetadataEntries(env, graph));
  result.Set("nodes", nodes);
  result.Set("values", value_list);
  return result;
}

/**
 * Projects only the selected graph, inheriting outer tensor/initializer metadata
 * unless shadowed locally. Child graphs become links, not recursive snapshots.
 */
Napi::Object ModelGraph(Napi::Env env, const ol::ModelProto &model, const std::string &file_name,
                        const GraphPath &path = {}) {
  const auto scopes = GraphScopes(model, path);
  const auto &graph = *scopes.back();
  std::map<std::string, Napi::Object> tensor_information;
  std::map<std::string, int64_t> opsets;
  for (const auto &opset : model.opset_import()) {
    opsets[Domain(opset.domain())] = opset.version();
  }
  std::map<std::pair<std::string, std::string>, Napi::Value> schemas;
  std::unordered_set<std::string> initializers;
  std::map<std::string, Napi::Object> parameters;
  std::unordered_set<std::string> local_names;
  for (const auto *scope : scopes) {
    local_names.clear();
    for (const auto &value : scope->input())
      local_names.emplace(value.name());
    for (const auto &value : scope->initializer())
      local_names.emplace(value.name());
    for (const auto &value : scope->sparse_initializer())
      local_names.emplace(value.values().name());
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
              TensorInformation(env, value.name(), value.has_type() ? &value.type() : nullptr));
    const auto &graph = *scope;
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
    result.Set("schema", schemas.at(key));
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

void ReadModel(ol::ModelProto &model, ol::utils::BinaryStream &stream, ol::ParseOptions options) {
  if (!model.ParseFromStream(stream, options)) {
    throw std::runtime_error("Could not decode the ONNX model.");
  }
  if (!model.has_graph() || !model.has_ir_version() || model.ir_version() <= 0) {
    throw std::runtime_error("This file does not contain an ONNX model graph.");
  }
}

/** With raw data retained, the caller must keep bytes alive for borrowed fields. */
void ReadModel(ol::ModelProto &model, const Napi::Uint8Array &bytes, bool skip_raw_data) {
  ol::utils::StringStream stream(bytes.Data(), bytes.ByteLength());
  ol::ParseOptions options;
  options.skip_raw_data = skip_raw_data;
  options.no_copy = !skip_raw_data;
  ReadModel(model, stream, options);
}

/**
 * Root shape application does not persist nested value_info. Apply the retained
 * GRAPH contexts before inference state is discarded.
 */
void ApplySubgraphShapes(ol::GraphProto &graph, const ol::core::shapes::ShapesContext &context) {
  for (int index = 0; index < graph.node_size(); ++index) {
    for (auto &attribute : graph.ref_node()[index].ref_attribute()) {
      if (attribute.type() != ol::AttributeProto::GRAPH || !attribute.has_g() ||
          !context.HasSubgraphContext(index, std::string(attribute.name())))
        continue;
      const auto &child = context.GetSubgraphContext(index, std::string(attribute.name()));
      child.ApplyInferredShapesToGraph(attribute.ref_g());
      ApplySubgraphShapes(attribute.ref_g(), child);
    }
  }
}

void InferShapes(ol::ModelProto &model) {
  RunAlgorithm("VerifyModel", [&] { ol::VerifyModel(model); });
  RunAlgorithm("RegisterShapeFunctions", [] { ol::onnx_shapes::RegisterShapeFunctions(); });
  ol::core::shapes::ShapesContext context;
  RunAlgorithm("ComputeShapeModel", [&] { context.ComputeShapeModel(model); });
  RunAlgorithm("ApplyInferredShapesToModel", [&] { context.ApplyInferredShapesToModel(model); });
  RunAlgorithm("ApplySubgraphShapes", [&] { ApplySubgraphShapes(model.ref_graph(), context); });
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

/**
 * Owns the parsed model and its backing storage. Declaration order keeps the
 * source buffer or backend case alive until after the model is destroyed.
 */
struct StoredModel {
  Napi::Reference<Napi::Uint8Array> source;
  std::shared_ptr<ol::core::backend_test::TestCase> backend_case;
  std::shared_ptr<ol::ModelProto> model;
  std::string file_name;
};

/** Method closures share ownership, even if JavaScript retains only one method. */
Napi::Object ModelHandle(Napi::Env env, std::shared_ptr<StoredModel> model) {
  auto result = Napi::Object::New(env);
  result.Set("metadata",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               const auto env = call.Env();
               if (call.Length() > 1 ||
                   (call.Length() == 1 && !call[0].IsUndefined() && !call[0].IsArray())) {
                 Napi::TypeError::New(env, "Expected an optional subgraph path.")
                     .ThrowAsJavaScriptException();
                 return env.Undefined();
               }
               try {
                 return ModelMetadata(env, *model->model, ReadPath(call[0]));
               } catch (const std::exception &error) {
                 Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
                 return env.Undefined();
               }
             }));
  result.Set(
      "graph", Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
        const auto env = call.Env();
        if (call.Length() > 2 ||
            (call.Length() >= 1 && !call[0].IsUndefined() && !call[0].IsBoolean()) ||
            (call.Length() == 2 && !call[1].IsUndefined() && !call[1].IsArray())) {
          Napi::TypeError::New(env, "Expected an optional inference flag and subgraph path.")
              .ThrowAsJavaScriptException();
          return env.Undefined();
        }
        try {
          const auto path = ReadPath(call[1]);
          GraphScopes(*model->model, path);
          if (call.Length() >= 1 && call[0].IsBoolean() && call[0].As<Napi::Boolean>().Value()) {
            InferShapes(*model->model);
          }
          return ModelGraph(env, *model->model, model->file_name, path);
        } catch (const std::exception &error) {
          Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
          return env.Undefined();
        }
      }));
  return result;
}

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
    model->model = std::make_shared<ol::ModelProto>();
    ReadModel(*model->model, model->source.Value(), false);
    return ModelHandle(env, model);
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

class LoadModelWorker : public Napi::AsyncWorker {
public:
  LoadModelWorker(Napi::Env env, std::string path, std::string file_name)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        path_(std::move(path)), model_(std::make_shared<StoredModel>()) {
    model_->file_name = std::move(file_name);
    model_->model = std::make_shared<ol::ModelProto>();
  }

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      ol::utils::FileStream stream(path_);
      ol::ParseOptions options;
      options.skip_raw_data = true;
      ReadModel(*model_->model, stream, options);
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override { deferred_.Resolve(ModelHandle(Env(), model_)); }
  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  Napi::Promise::Deferred deferred_;
  std::string path_;
  std::shared_ptr<StoredModel> model_;
};

Napi::Value LoadModelFile(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if (info.Length() != 2 || !info[0].IsString() || !info[1].IsString()) {
    Napi::TypeError::New(env, "Expected a file path and a file name.").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  try {
    const auto path = info[0].As<Napi::String>().Utf8Value();
    if (path.find('\0') != std::string::npos)
      throw std::runtime_error("Invalid model file path.");
    auto worker =
        std::make_unique<LoadModelWorker>(env, path, info[1].As<Napi::String>().Utf8Value());
    const auto promise = worker->Promise();
    worker->Queue();
    worker.release();
    return promise;
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value ListBackendTests(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  try {
    const auto cases = ol::core::backend_test::CollectTestCases("", true);
    auto result = Napi::Array::New(env);
    uint32_t index = 0;
    for (const auto &test : cases) {
      auto item = Napi::Object::New(env);
      item.Set("id", index);
      item.Set("name", test.name);
      item.Set("modelName", test.model_name);
      item.Set("kind", Text(env, ol::core::backend_test::TestCaseKindName(test.kind)));
      item.Set("tag", Text(env, ol::core::backend_test::TestCaseTagName(test.tag)));
      result.Set(index++, item);
    }
    return result;
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value LoadBackendTest(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if (info.Length() != 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected a backend test ID.").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  const auto id = info[0].As<Napi::Number>().DoubleValue();
  if (!std::isfinite(id) || id < 0 || std::floor(id) != id) {
    Napi::RangeError::New(env, "Invalid onnx-light backend test ID.").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  try {
    auto cases = ol::core::backend_test::CollectTestCases("", true);
    if (id >= cases.size()) {
      throw std::runtime_error("Unknown onnx-light backend test ID: " + std::to_string(id));
    }
    auto model = std::make_shared<StoredModel>();
    model->backend_case = std::make_shared<ol::core::backend_test::TestCase>(
        std::move(cases[static_cast<size_t>(id)]));
    model->backend_case->set_expected_outputs_generated(false);
    model->model = model->backend_case->model_handle();
    model->file_name = model->backend_case->name;
    return ModelHandle(env, model);
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Object Initialize(Napi::Env env, Napi::Object exports) {
  exports.Set("onnxLightVersion", ONNX_LIGHT_VERSION);
  exports.Set("decodeGraph", Napi::Function::New(env, DecodeGraph));
  exports.Set("getSchema", Napi::Function::New(env, GetSchema));
  exports.Set("loadModel", Napi::Function::New(env, LoadModel));
  exports.Set("loadModelFile", Napi::Function::New(env, LoadModelFile));
  exports.Set("listBackendTests", Napi::Function::New(env, ListBackendTests));
  exports.Set("loadBackendTest", Napi::Function::New(env, LoadBackendTest));
  return exports;
}

} // namespace

NODE_API_MODULE(onnx_light_viewer, Initialize)
