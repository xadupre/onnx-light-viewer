/** @file External payload materialization and model tensor traversal. */
#include "model_storage.h"
#include "model_io.h"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <functional>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace onnx_light_viewer {
namespace {

template <typename Proto>
void LoadExternalTensorImpl(Proto &tensor, const std::filesystem::path &directory) {
  const auto location = ExternalDataLocation(tensor);
  int64_t offset = 0;
  int64_t length = -1;
  const auto number = [](std::string_view text) {
    int64_t value;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
      throw std::runtime_error("Invalid external weight offset or length.");
    return value;
  };
  for (const auto &entry : tensor.external_data()) {
    if (entry.key() == "offset")
      offset = number(std::string_view(entry.value()));
    else if (entry.key() == "length" || entry.key() == "size")
      length = number(std::string_view(entry.value()));
  }
  const auto relative = ModelFilePath(location);
  const auto normalized = relative.lexically_normal();
  size_t components = 0;
  bool unsafe = location.empty() || location.find('\0') != std::string::npos ||
                normalized.empty() || normalized.has_root_path() || normalized.filename().empty() ||
                normalized == std::filesystem::path(".");
  for (const auto &part : relative)
    unsafe |= part == std::filesystem::path("..");
  for (const auto &part : normalized)
    if (part != std::filesystem::path("."))
      unsafe |= ++components > 2;
  if (unsafe)
    throw std::runtime_error("External weight location must be a safe relative path.");
  const auto base_directory = directory.empty() ? std::filesystem::current_path() : directory;
  const auto path = base_directory / normalized;
  if (std::filesystem::is_symlink(path))
    throw std::runtime_error("External weight files must not be symbolic links.");
  const auto base = std::filesystem::weakly_canonical(base_directory);
  const auto target = std::filesystem::weakly_canonical(path);
  auto prefix = base.native();
  if (!prefix.empty() && prefix.back() != std::filesystem::path::preferred_separator)
    prefix += std::filesystem::path::preferred_separator;
  if (target != base && target.native().find(prefix) != 0)
    throw std::runtime_error("External weight file resolves outside the model directory.");
  if (std::filesystem::hard_link_count(path) > 1)
    throw std::runtime_error("External weight files must not have multiple hard links.");
  const auto utf8_path = path.u8string();
  FileDescriptor file(OpenModelFile(std::string(utf8_path.begin(), utf8_path.end())));
  if (file.get() < 0)
    throw std::system_error(errno, std::generic_category(), "Could not open external weights");
  ol::utils::FdReadStream stream(file.get());
  offset = std::max<int64_t>(0, offset);
  if (offset > stream.size())
    throw std::runtime_error("External weight offset is past the end of the file.");
  if (length < 0)
    length = stream.size() - offset;
  if (length > stream.size() - offset ||
      static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max())
    throw std::runtime_error("External weight length exceeds the available data.");
  stream.skip_bytes(offset);
  tensor.ref_raw_data().resize(static_cast<size_t>(length));
  if (length)
    stream.read_bytes(length, tensor.ref_raw_data().data());
  tensor.set_data_location(ol::TensorProto::DEFAULT);
  tensor.clear_external_data();
}

} // namespace

void LoadExternalTensor(ol::TensorProto &tensor, const std::filesystem::path &directory) {
  LoadExternalTensorImpl(tensor, directory);
}

void LoadExternalTensor(ol::EncodedValueProto &tensor, const std::filesystem::path &directory) {
  LoadExternalTensorImpl(tensor, directory);
}

/** Includes sparse components, which the native generic tensor iterator omits. */
std::vector<ol::TensorProto *> ModelSaveTensors(ol::ModelProto &model,
                                                std::vector<ol::EncodedValueProto *> *encoded,
                                                std::set<const ol::TensorProto *> *layout_tensors) {
  std::vector<ol::TensorProto *> tensors;
  const auto layout_tensor = [&](ol::TensorProto &value) {
    tensors.push_back(&value);
    if (layout_tensors)
      layout_tensors->insert(&value);
  };
  const auto sparse = [&](ol::SparseTensorProto &value) {
    if (value.has_values())
      tensors.push_back(&value.ref_values());
    if (value.has_indices())
      tensors.push_back(&value.ref_indices());
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
    for (auto &tensor : value.ref_initializer())
      tensors.push_back(&tensor);
    for (auto &tensor : value.ref_sparse_initializer())
      sparse(tensor);
    for (auto &initializer : value.ref_extended_initializer())
      if (initializer.has_encoded_value()) {
        auto &item = initializer.ref_encoded_value();
        if (encoded)
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
      tensors.push_back(&value.ref_t());
    for (auto &tensor : value.ref_tensors())
      tensors.push_back(&tensor);
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
      for (auto &tensor : edit.ref_added_initializers())
        tensors.push_back(&tensor);
    }
  }
  return tensors;
}

} // namespace onnx_light_viewer
