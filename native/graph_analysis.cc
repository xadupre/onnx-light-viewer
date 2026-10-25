/** @file Graph-path scopes, metadata analysis and shape inference. */
#include "graph.h"
#include "tensor_preview.h"
#include <cmath>
#include <limits>
#include <map>
#include <onnx_core/compute/constant_info.h>
#include <onnx_core/compute/result_lifetime.h>
#include <onnx_core/compute/value_tags.h>
#include <onnx_core/graph/graph_manipulations.h>
#include <onnx_core/shapes/shape_inference.h>
#include <onnx_extensions/shapes/dispatch_table.h>
#include <onnx_proto/onnx_verify.h>
#include <stdexcept>
#include <unordered_set>

namespace onnx_light_viewer {
namespace {

template <typename Action> auto RunAlgorithm(const char *name, Action action) {
  try {
    return action();
  } catch (const std::exception &error) {
    throw std::runtime_error(std::string(name) + ": " + error.what());
  }
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

bool CallsFunction(const ol::NodeProto &node, const ol::FunctionProto &function) {
  return Domain(node.domain()) == Domain(function.domain()) && node.op_type() == function.name() &&
         node.overload() == function.overload();
}

ol::GraphProto FunctionGraph(const ol::FunctionProto &function) {
  ol::GraphProto graph;
  graph.set_name(FunctionName(function));
  const auto copy_port = [&](ol::ValueInfoProto *port, std::string_view name) {
    port->set_name(std::string(name));
    for (const auto &info : function.value_info())
      if (info.name() == name) {
        *port = info.SharedCopy();
        break;
      }
  };
  for (const auto &name : function.input())
    copy_port(graph.add_input(), name);
  for (const auto &name : function.output())
    copy_port(graph.add_output(), name);
  for (const auto &info : function.value_info())
    *graph.add_value_info() = info.SharedCopy();
  for (const auto &node : function.node())
    *graph.add_node() = node.SharedCopy();
  for (const auto &entry : function.metadata_props())
    *graph.add_metadata_props() = entry;
  return graph;
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

} // namespace

std::string Domain(std::string_view domain) {
  return domain.empty() ? "ai.onnx" : std::string(domain);
}

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
    if (object.Has("function"))
      path.push_back({object.Has("node") ? read("node") : -1, 0, 0, read("function")});
    else
      path.push_back({read("node"), read("attribute"), read("graph")});
  }
  return path;
}

Napi::Array PathValue(Napi::Env env, const GraphPath &path) {
  auto result = Napi::Array::New(env);
  for (uint32_t index = 0; index < path.size(); ++index) {
    auto step = Napi::Object::New(env);
    if (path[index].node >= 0)
      step.Set("node", path[index].node);
    if (path[index].function >= 0)
      step.Set("function", path[index].function);
    else {
      step.Set("attribute", path[index].attribute);
      step.Set("graph", path[index].graph);
    }
    result.Set(index, step);
  }
  return result;
}

std::string FunctionName(const ol::FunctionProto &function) {
  return Domain(function.domain()) + "::" + std::string(function.name()) +
         (function.overload().empty() ? "" : " [" + std::string(function.overload()) + "]");
}

/** Function bodies own their namespace; only graph attributes capture enclosing values. */
GraphSelection GraphScopes(const ol::ModelProto &model, const GraphPath &path) {
  GraphSelection selection;
  selection.scopes.push_back(&model.graph());
  for (const auto &step : path) {
    const auto &parent = *selection.scopes.back();
    if (step.function < 0) {
      selection.scopes.push_back(&ChildGraph(parent, step));
      continue;
    }
    if (step.function >= model.functions_size() ||
        (step.node >= 0 &&
         (step.node >= parent.node_size() ||
          !CallsFunction(parent.node(step.node), model.functions(step.function)))))
      throw std::runtime_error("Unknown function in subgraph path.");
    selection.function = &model.functions(step.function);
    selection.functions.push_back(
        std::make_unique<ol::GraphProto>(FunctionGraph(*selection.function)));
    selection.capture_start = selection.scopes.size();
    selection.scopes.push_back(selection.functions.back().get());
  }
  return selection;
}

/** Seeds captured constants from enclosing scopes without modifying the model. */
Napi::Object ModelMetadata(Napi::Env env, const ol::ModelProto &model, const GraphPath &path) {
  const auto selection = GraphScopes(model, path);
  const auto &scopes = selection.scopes;
  RunAlgorithm("VerifyModel", [&] { ol::VerifyModel(model); });
  const auto &graph = *scopes.back();
  std::unordered_set<std::string> outer_constants;
  for (size_t index = selection.capture_start; index + 1 < scopes.size(); ++index)
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
  for (const auto &value : graph.extended_initializer())
    if (value.has_encoded_value())
      add_value(value.encoded_value().name());
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

void InferShapes(ol::ModelProto &model) {
  RunAlgorithm("VerifyModel", [&] { ol::VerifyModel(model); });
  RunAlgorithm("RegisterShapeFunctions", [] { ol::onnx_shapes::RegisterShapeFunctions(); });
  ol::core::shapes::ShapesContext context;
  RunAlgorithm("ComputeShapeModel", [&] { context.ComputeShapeModel(model); });
  RunAlgorithm("ApplyInferredShapesToModel", [&] { context.ApplyInferredShapesToModel(model); });
  RunAlgorithm("ApplySubgraphShapes", [&] { ApplySubgraphShapes(model.ref_graph(), context); });
}

} // namespace onnx_light_viewer
