/** @file Application-owned edits of symbolic dimension labels. */
#include "model.h"
#include <stdexcept>
#include <unordered_map>

namespace onnx_light_viewer {
namespace {

class DimensionNames {
public:
  std::vector<ol::utils::OptionalString *> values;

  void Shape(ol::TensorShapeProto &shape) {
    for (auto &dim : shape.ref_dim())
      if (!dim.dim_param().empty())
        values.push_back(&dim.ref_dim_param());
  }

  void Type(ol::TypeProto &type) {
    if (type.has_tensor_type() && type.tensor_type().has_shape())
      Shape(type.ref_tensor_type().ref_shape());
    if (type.has_sparse_tensor_type() && type.sparse_tensor_type().has_shape())
      Shape(type.ref_sparse_tensor_type().ref_shape());
    if (type.has_sequence_type() && type.sequence_type().has_elem_type())
      Type(type.ref_sequence_type().ref_elem_type());
    if (type.has_optional_type() && type.optional_type().has_elem_type())
      Type(type.ref_optional_type().ref_elem_type());
    if (type.has_map_type() && type.map_type().has_value_type())
      Type(type.ref_map_type().ref_value_type());
    if (type.has_struct_type())
      Structure(type.ref_struct_type());
  }

  void Structure(ol::StructTypeProto &type) {
    if (type.has_array() && type.array().has_element_type())
      Type(type.ref_array().ref_element_type());
    if (type.has_structure())
      for (auto &field : type.ref_structure().ref_field())
        if (field.has_type())
          Type(field.ref_type());
    if (type.has_encoder())
      Function(type.ref_encoder());
    if (type.has_decoder())
      Function(type.ref_decoder());
  }

  void Attribute(ol::AttributeProto &attribute) {
    if (attribute.has_tp())
      Type(attribute.ref_tp());
    for (auto &type : attribute.ref_type_protos())
      Type(type);
    if (attribute.has_g())
      Graph(attribute.ref_g());
    for (auto &graph : attribute.ref_graphs())
      Graph(graph);
  }

  void Nodes(ol::utils::RepeatedProtoField<ol::NodeProto> &nodes) {
    for (auto &node : nodes) {
      for (auto &attribute : node.ref_attribute())
        Attribute(attribute);
      for (auto &configuration : node.ref_device_configurations())
        for (auto &spec : configuration.ref_sharding_spec())
          for (auto &dim : spec.ref_sharded_dim())
            for (auto &sharding : dim.ref_simple_sharding())
              if (!sharding.dim_param().empty())
                values.push_back(&sharding.ref_dim_param());
    }
  }

  void Values(ol::utils::RepeatedProtoField<ol::ValueInfoProto> &values) {
    for (auto &value : values)
      if (value.has_type())
        Type(value.ref_type());
  }

  void Graph(ol::GraphProto &graph) {
    Values(graph.ref_input());
    Values(graph.ref_output());
    Values(graph.ref_value_info());
    Nodes(graph.ref_node());
    for (auto &initializer : graph.ref_extended_initializer())
      if (initializer.has_encoded_value()) {
        auto &value = initializer.ref_encoded_value();
        if (value.has_logical_type())
          Type(value.ref_logical_type());
        if (value.has_struct_type())
          Structure(value.ref_struct_type());
      }
  }

  void Function(ol::FunctionProto &function) {
    Values(function.ref_value_info());
    Nodes(function.ref_node());
    for (auto &attribute : function.ref_attribute_proto())
      Attribute(attribute);
  }

  void Plan(ol::RewritingProto &plan) {
    for (auto &function : plan.ref_added_functions())
      Function(function);
    for (auto &edit : plan.ref_rewrites()) {
      Nodes(edit.ref_added_nodes());
      Nodes(edit.ref_initializer_calls());
    }
  }

  void Model(ol::ModelProto &model) {
    if (model.has_graph())
      Graph(model.ref_graph());
    for (auto &function : model.ref_functions())
      Function(function);
    for (auto &type : model.ref_struct_types())
      Structure(type);
    for (auto &[name, plan] : model.ref_rewritings())
      Plan(plan);
  }

  std::string Prefix() const {
    if (values.empty())
      return "";
    std::string prefix(values.front()->sv());
    for (const auto *value : values) {
      const auto name = value->sv();
      size_t length = 0;
      while (length < prefix.size() && length < name.size() && prefix[length] == name[length])
        ++length;
      // Never remove an entire name, even if it ends in a namespace separator.
      prefix.resize(length == name.size() && length ? length - 1 : length);
      if (prefix.empty())
        return "";
    }
    const auto boundary = prefix.find_last_of("./:_");
    return boundary == std::string::npos ? "" : prefix.substr(0, boundary + 1);
  }
};

struct DimensionEdit {
  DimensionNames names;
  std::unordered_map<const StoredModel *, std::shared_ptr<StoredModel>> copies;

  std::shared_ptr<StoredModel> Copy(const std::shared_ptr<StoredModel> &source) {
    if (!source)
      return nullptr;
    if (const auto found = copies.find(source.get()); found != copies.end())
      return found->second;
    auto copy = CloneStoredModel(source);
    copies.emplace(source.get(), copy);
    names.Model(*copy->model);
    for (auto &plan : copy->fusion_plans)
      names.Plan(plan);
    copy->fusion_base = Copy(source->fusion_base);
    copy->fusion_previous = Copy(source->fusion_previous);
    copy->rewriting_base = Copy(source->rewriting_base);
    return copy;
  }
};

} // namespace

void AddDimensionMethods(Napi::Env env, Napi::Object result, std::shared_ptr<StoredModel> model) {
  result.Set("dimensionPrefix",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 DimensionEdit edit;
                 edit.Copy(model);
                 return edit.names.values.empty()
                            ? Napi::Value(call.Env().Null())
                            : Napi::Value(Napi::String::New(call.Env(), edit.names.Prefix()));
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  result.Set(
      "shortenDimensionNames",
      Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
        try {
          if (call.Length() > 1 ||
              (call.Length() == 1 && !call[0].IsUndefined() && !call[0].IsString()))
            throw std::runtime_error("Expected an optional dimension prefix.");
          DimensionEdit edit;
          auto copy = edit.Copy(model);
          const std::string prefix = call.Length() && call[0].IsString()
                                         ? call[0].As<Napi::String>().Utf8Value()
                                         : edit.names.Prefix();
          if (!prefix.empty())
            for (const auto *value : edit.names.values)
              if (!value->sv().starts_with(prefix) || value->sv().size() == prefix.size())
                throw std::runtime_error(
                    "The prefix must apply to every symbolic dimension without emptying a name.");
          const size_t count = prefix.empty() ? 0 : edit.names.values.size();
          if (count)
            for (auto *value : edit.names.values)
              *value = std::string(value->sv().substr(prefix.size()));
          auto result = Napi::Object::New(call.Env());
          result.Set("model", ModelHandle(call.Env(), count ? copy : model));
          result.Set("prefix", prefix);
          result.Set("count", Napi::Number::New(call.Env(), count));
          return result;
        } catch (const std::exception &error) {
          Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
          return call.Env().Undefined();
        }
      }));
}

} // namespace onnx_light_viewer
