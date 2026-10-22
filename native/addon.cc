#include <napi.h>
#include <onnx_proto/onnx.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace ol = ONNX_LIGHT_NAMESPACE;

namespace {

Napi::String Text(Napi::Env env, std::string_view value) {
  return Napi::String::New(env, value.empty() ? "" : value.data(), value.size());
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

Napi::Value DecodeGraph(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if (info.Length() != 2 || !info[0].IsTypedArray() ||
      info[0].As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array || !info[1].IsString()) {
    Napi::TypeError::New(env, "Expected a Uint8Array and a file name.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  // C++ exceptions must not unwind through the Node-API boundary.
  try {
    const auto bytes = info[0].As<Napi::Uint8Array>();
    ol::utils::StringStream stream(bytes.Data(), bytes.ByteLength());
    ol::ParseOptions options;
    options.skip_raw_data = true;
    ol::ModelProto model;
    if (!model.ParseFromStream(stream, options)) {
      throw std::runtime_error("Could not decode the ONNX model.");
    }
    if (!model.has_graph() || !model.has_ir_version() || model.ir_version() <= 0) {
      throw std::runtime_error("This file does not contain an ONNX model graph.");
    }

    const auto &graph = model.graph();
    std::unordered_set<std::string> initializers;
    for (const auto &tensor : graph.initializer()) {
      if (!tensor.name().empty()) {
        initializers.emplace(std::string_view(tensor.name()));
      }
    }
    for (const auto &tensor : graph.sparse_initializer()) {
      if (tensor.has_values() && !tensor.values().name().empty()) {
        initializers.emplace(std::string_view(tensor.values().name()));
      }
    }

    auto nodes = Napi::Array::New(env);
    uint32_t index = 0;
    for (const auto &node : graph.node()) {
      if (node.op_type().empty()) {
        throw std::runtime_error("Node " + std::to_string(index + 1) +
                                 " has no ONNX operator type.");
      }
      auto result = Napi::Object::New(env);
      result.Set("id", "node-" + std::to_string(index));
      result.Set("name", node.name().empty() ? Text(env, std::string(node.op_type()) + "_" +
                                                             std::to_string(index + 1))
                                             : Text(env, node.name()));
      result.Set("opType", Text(env, node.op_type()));
      result.Set("domain", Text(env, node.domain().empty() ? std::string_view("ai.onnx")
                                                           : std::string_view(node.domain())));
      result.Set("inputs", Ports(env, node.input()));
      result.Set("outputs", Ports(env, node.output()));
      nodes.Set(index++, result);
    }
    auto result = Napi::Object::New(env);
    result.Set("name", graph.name().empty() ? info[1].As<Napi::String>() : Text(env, graph.name()));
    result.Set("nodes", nodes);
    result.Set("inputs", ValueNames(env, graph.input(), initializers));
    result.Set("outputs", ValueNames(env, graph.output()));
    result.Set("initializerCount", Napi::Number::New(env, initializers.size()));
    return result;
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Object Initialize(Napi::Env env, Napi::Object exports) {
  exports.Set("decodeGraph", Napi::Function::New(env, DecodeGraph));
  return exports;
}

} // namespace

NODE_API_MODULE(onnx_light_viewer, Initialize)
