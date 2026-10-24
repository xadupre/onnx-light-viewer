/** @file Model tensor traversal for saving. */
#include "model_storage.h"
#include <functional>
#include <unordered_set>

namespace onnx_light_viewer {

/** Includes sparse components, which the native generic tensor iterator omits. */
std::vector<ol::TensorProto *> ModelSaveTensors(ol::ModelProto &model,
                                                std::vector<ol::EncodedValueProto *> *encoded,
                                                std::set<const ol::TensorProto *> *layout_tensors) {
  std::vector<ol::TensorProto *> tensors;
  std::unordered_set<const ol::TensorProto *> visited;
  std::unordered_set<const ol::EncodedValueProto *> visited_encoded;
  const auto tensor = [&](ol::TensorProto &value) {
    if (visited.insert(&value).second)
      tensors.push_back(&value);
  };
  const auto layout_tensor = [&](ol::TensorProto &value) {
    tensor(value);
    if (layout_tensors)
      layout_tensors->insert(&value);
  };
  const auto sparse = [&](ol::SparseTensorProto &value) {
    if (value.has_values())
      tensor(value.ref_values());
    if (value.has_indices())
      tensor(value.ref_indices());
  };
  std::function<void(ol::AttributeProto &)> attribute;
  std::function<void(ol::TypeProto &)> type;
  std::function<void(ol::StructTypeProto &)> structure;
  const auto nodes = [&](auto &values) {
    for (auto &node : values)
      for (auto &value : node.ref_attribute())
        attribute(value);
  };
  const auto graph = [&](auto &value) {
    for (auto *values : {&value.ref_input(), &value.ref_output(), &value.ref_value_info()})
      for (auto &item : *values)
        if (item.has_type())
          type(item.ref_type());
    for (auto &item : value.ref_initializer())
      tensor(item);
    for (auto &item : value.ref_sparse_initializer())
      sparse(item);
    for (auto &initializer : value.ref_extended_initializer())
      if (initializer.has_encoded_value()) {
        auto &item = initializer.ref_encoded_value();
        if (encoded && visited_encoded.insert(&item).second)
          encoded->push_back(&item);
        if (item.has_affine()) {
          if (item.affine().has_scale())
            layout_tensor(item.ref_affine().ref_scale());
          if (item.affine().has_zero_point())
            layout_tensor(item.ref_affine().ref_zero_point());
        }
        if (item.has_struct_type())
          structure(item.ref_struct_type());
      }
    nodes(value.ref_node());
  };
  attribute = [&](ol::AttributeProto &value) {
    if (value.has_tp())
      type(value.ref_tp());
    for (auto &item : value.ref_type_protos())
      type(item);
    if (value.has_t())
      tensor(value.ref_t());
    for (auto &item : value.ref_tensors())
      tensor(item);
    if (value.has_sparse_tensor())
      sparse(value.ref_sparse_tensor());
    for (auto &tensor : value.ref_sparse_tensors())
      sparse(tensor);
    if (value.has_g())
      graph(value.ref_g());
    for (auto &child : value.ref_graphs())
      graph(child);
  };
  const auto functions = [&](auto &values) {
    for (auto &function : values) {
      nodes(function.ref_node());
      for (auto &value : function.ref_attribute_proto())
        attribute(value);
    }
  };
  type = [&](ol::TypeProto &value) {
    if (value.has_struct_type())
      structure(value.ref_struct_type());
    if (value.has_sequence_type() && value.sequence_type().has_elem_type())
      type(value.ref_sequence_type().ref_elem_type());
    if (value.has_optional_type() && value.optional_type().has_elem_type())
      type(value.ref_optional_type().ref_elem_type());
    if (value.has_map_type() && value.map_type().has_value_type())
      type(value.ref_map_type().ref_value_type());
  };
  structure = [&](ol::StructTypeProto &value) {
    if (value.has_array() && value.array().has_element_type())
      type(value.ref_array().ref_element_type());
    if (value.has_structure())
      for (auto &field : value.ref_structure().ref_field()) {
        if (field.has_constant())
          layout_tensor(field.ref_constant());
        if (field.has_type())
          type(field.ref_type());
      }
    for (auto *codec : {value.has_encoder() ? &value.ref_encoder() : nullptr,
                        value.has_decoder() ? &value.ref_decoder() : nullptr})
      if (codec) {
        nodes(codec->ref_node());
        for (auto &item : codec->ref_attribute_proto())
          attribute(item);
      }
  };
  for (auto &value : model.ref_struct_types())
    structure(value);
  if (model.has_graph())
    graph(model.ref_graph());
  functions(model.ref_functions());
  for (auto &[name, plan] : model.ref_rewritings()) {
    functions(plan.ref_added_functions());
    for (auto &edit : plan.ref_rewrites()) {
      nodes(edit.ref_added_nodes());
      nodes(edit.ref_initializer_calls());
      for (auto &item : edit.ref_added_initializers())
        tensor(item);
    }
  }
  return tensors;
}

} // namespace onnx_light_viewer
