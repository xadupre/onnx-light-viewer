#include "../native/model_storage.h"

#include <map>
#include <onnx_core/builder/graph_builder.h>
#include <onnx_extensions/kernels/kernel_dispatch_table.h>
#include <onnx_extensions/shapes/dispatch_table.h>
#include <onnx_proto/onnx_verify.h>
#include <stdexcept>

namespace {

namespace viewer = onnx_light_viewer;
namespace ol = viewer::ol;

Napi::Value TensorVisits(const Napi::CallbackInfo &info) {
  ol::ModelProto model;
  auto &layout = *model.add_struct_types();
  layout.set_name("quantization.v1/shared");
  layout.set_type_id(1);
  auto &attributes = layout.ref_encoder().ref_attribute_proto();
  attributes.resize(2);
  auto &fields = attributes[1].ref_tp().ref_struct_type().ref_structure().ref_field();
  fields.resize(2);
  auto &constant = fields[0].ref_constant();
  constant.set_name("scale");
  constant.set_data_type(ol::TensorProto::FLOAT);
  constant.add_dims(64);
  for (int i = 0; i < 64; ++i)
    constant.add_float_data(0.5f);
  fields[1].ref_constant() = constant.SharedCopy();

  // The ordinary visit precedes classification as a layout constant.
  auto &ordinary = attributes[0].ref_tensors();
  ordinary.resize(1);
  ordinary.get(0) = std::shared_ptr<ol::TensorProto>(fields.shared_at(0), &constant);

  // Repeated type/codec paths share message objects, not merely tensor payloads.
  auto &decoder_fields = layout.ref_decoder()
                             .add_attribute_proto()
                             ->ref_tp()
                             .ref_struct_type()
                             .ref_structure()
                             .ref_field();
  decoder_fields.resize(2);
  for (size_t i = 0; i < fields.size(); ++i)
    decoder_fields.get(i) = fields.shared_at(i);

  std::set<const ol::TensorProto *> layout_tensors;
  const auto tensors = viewer::ModelSaveTensors(model, nullptr, &layout_tensors);
  const std::map<const ol::TensorProto *, int> identities{{&constant, 0},
                                                          {&fields[1].ref_constant(), 1}};
  auto visits = Napi::Array::New(info.Env(), tensors.size());
  auto classified = Napi::Array::New(info.Env(), tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    visits.Set(i, identities.at(tensors[i]));
    classified.Set(i, layout_tensors.contains(tensors[i]));
  }
  auto result = Napi::Object::New(info.Env());
  result.Set("visits", visits);
  result.Set("layout", classified);
  result.Set("layoutCount", layout_tensors.size());
  return result;
}

Napi::Value LayoutTensors(const Napi::CallbackInfo &info) {
  const auto bytes = info[0].As<Napi::Buffer<uint8_t>>();
  ol::ModelProto model;
  if (!model.ParseFromString(
          std::string(reinterpret_cast<const char *>(bytes.Data()), bytes.Length()))) {
    Napi::Error::New(info.Env(), "Could not parse test model").ThrowAsJavaScriptException();
    return info.Env().Undefined();
  }
  std::set<const ol::TensorProto *> layout;
  const auto tensors = viewer::ModelSaveTensors(model, nullptr, &layout);
  auto result = Napi::Array::New(info.Env());
  uint32_t index = 0;
  for (const auto *tensor : tensors) {
    if (!layout.contains(tensor))
      continue;
    std::string serialized;
    tensor->SerializeToString(serialized);
    auto value = Napi::Object::New(info.Env());
    value.Set("bytes", Napi::Buffer<char>::Copy(info.Env(), serialized.data(), serialized.size()));
    value.Set("hasDataLocation", tensor->has_data_location());
    value.Set("hasRawData", tensor->has_raw_data());
    value.Set("hasName", tensor->has_name());
    value.Set("hasDataType", tensor->has_data_type());
    result.Set(index++, value);
  }
  return result;
}

Napi::Value EncodedVisits(const Napi::CallbackInfo &info) {
  ol::ModelProto model;
  auto &graphs = model.ref_graph().add_node()->add_attribute()->ref_graphs();
  graphs.resize(2);
  auto &initializers = graphs[0].ref_extended_initializer();
  initializers.resize(2);
  auto &first = initializers[0].ref_encoded_value();
  first.set_name("weight");
  first.set_raw_data("codes");
  first.ref_affine().ref_scale().add_float_data(0.5f);
  initializers[1].ref_encoded_value() = first.SharedCopy();
  graphs.get(1) = graphs.shared_at(0);

  std::vector<ol::EncodedValueProto *> encoded;
  const auto tensors = viewer::ModelSaveTensors(model, &encoded);
  auto result = Napi::Object::New(info.Env());
  result.Set("encodedCount", encoded.size());
  result.Set("tensorCount", tensors.size());
  result.Set("firstVisit", !encoded.empty() && encoded[0] == &first);
  result.Set("secondVisit",
             encoded.size() > 1 && encoded[1] == &initializers[1].ref_encoded_value());
  return result;
}

Napi::Value RewritingPlans(const Napi::CallbackInfo &info) {
  try {
    const auto bytes = info[0].As<Napi::Buffer<uint8_t>>();
    ol::ModelProto model;
    if (!model.ParseFromString(
            std::string(reinterpret_cast<const char *>(bytes.Data()), bytes.Length())))
      throw std::runtime_error("Could not parse test model.");
    ol::onnx_shapes::RegisterShapeFunctions();
    ol::onnx_kernels::RegisterKernelFunctions();
    auto result = Napi::Array::New(info.Env());
    for (const auto &[name, plan] : model.rewritings()) {
      auto replayed =
          ol::core::builder::GraphBuilder(model).ApplyRewritings(plan).ToModel(model.ir_version());
      ol::VerifyModel(replayed);
      std::string serialized;
      if (!replayed.SerializeToString(serialized))
        throw std::runtime_error("Could not serialize replayed rewritings.");
      auto item = Napi::Object::New(info.Env());
      item.Set("name", name);
      item.Set("count", plan.rewrites_size());
      item.Set("bytes", Napi::Buffer<char>::Copy(info.Env(), serialized.data(), serialized.size()));
      result.Set(result.Length(), item);
    }
    return result;
  } catch (const std::exception &error) {
    Napi::Error::New(info.Env(), error.what()).ThrowAsJavaScriptException();
    return info.Env().Undefined();
  }
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("rewritingPlans", Napi::Function::New(env, RewritingPlans));
  exports.Set("tensorVisits", Napi::Function::New(env, TensorVisits));
  exports.Set("layoutTensors", Napi::Function::New(env, LayoutTensors));
  exports.Set("encodedVisits", Napi::Function::New(env, EncodedVisits));
  return exports;
}

} // namespace

NODE_API_MODULE(test_model_storage, Init)
