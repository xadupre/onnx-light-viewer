/** @file One-shot execution of retained models through the native SDK. */
#include "model_io.h"
#include "model_storage.h"
#include "tensor_preview.h"
#include <chrono>
#include <cmath>
#include <map>
#include <onnx_core/runtime/kernels/random.h>
#include <onnx_core/runtime/runtime_session.h>
#include <onnx_manipulations/printer.h>
#include <set>
#include <stdexcept>

namespace onnx_light_viewer {
namespace {
namespace rt = ol::core::runtime;

void RootPath(const Napi::CallbackInfo &call, size_t index) {
  if (call.Length() > index + (index == 1 ? 2 : 1) ||
      (call.Length() > index && !call[index].IsUndefined() &&
       (!call[index].IsArray() || call[index].As<Napi::Array>().Length() != 0)))
    throw std::runtime_error("Run covers the whole model; choose a stage root for pipelines.");
}

struct InputSource {
  std::string name;
  std::string text;
  std::string protobuf;
  bool random = false;
};

int64_t Dimension(const Napi::Value &value) {
  if (!value.IsNumber())
    throw std::runtime_error("Expected a non-negative safe integer dimension.");
  const auto number = value.As<Napi::Number>().DoubleValue();
  if (!std::isfinite(number) || number < 0 || number > 9007199254740991.0 ||
      std::floor(number) != number)
    throw std::runtime_error("Expected a non-negative safe integer dimension.");
  return static_cast<int64_t>(number);
}

class RunWorker : public Napi::AsyncWorker {
public:
  RunWorker(Napi::Env env, std::shared_ptr<StoredModel> source, std::vector<InputSource> inputs,
            std::map<std::string, int64_t> dimensions, int verbose, bool prepare)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        source_(std::move(source)), snapshot_(source_->model->SharedCopy()),
        inputs_(std::move(inputs)), dimensions_(std::move(dimensions)), verbose_(verbose),
        prepare_(prepare) {}

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      std::vector<ol::TensorProto> input_storage;
      input_storage.reserve(inputs_.size());
      rt::RuntimeContext context;
      std::set<std::string> names;
      std::set<std::string> input_names;
      for (const auto &input : snapshot_.graph().input())
        input_names.emplace(input.name());
      for (const auto &input : inputs_) {
        auto &tensor = input_storage.emplace_back();
        if (!input.text.empty()) {
          ol::OnnxParser parser(input.text);
          const auto status = parser.Parse(tensor);
          if (!status.IsOK())
            throw std::runtime_error(status.ErrorMessage());
          if (!parser.EndOfInput())
            throw std::runtime_error("Unexpected text after the input tensor.");
        } else {
          if (!input_names.contains(input.name))
            throw std::runtime_error("Unknown model input: " + input.name);
          const ol::ValueInfoProto *metadata = nullptr;
          for (const auto &value : snapshot_.graph().input())
            if (value.name() == input.name)
              metadata = &value;
          if (input.random) {
            if (!metadata->type().has_tensor_type() || !metadata->type().tensor_type().has_shape())
              throw std::runtime_error("Random inputs require a tensor type and known rank.");
            rt::Shape shape;
            size_t axis = 0;
            for (const auto &dim : metadata->type().tensor_type().shape().dim()) {
              const auto key = dim.has_dim_param() && !dim.dim_param().empty()
                                   ? std::string(dim.dim_param())
                                   : input.name + ":" + std::to_string(axis);
              if (dim.has_dim_value())
                shape.push_back(dim.dim_value());
              else {
                const auto found = dimensions_.find(key);
                if (found == dimensions_.end())
                  throw std::runtime_error("Specify dynamic dimension: " + key);
                shape.push_back(found->second);
              }
              ++axis;
            }
            const auto seed =
                static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
            auto generated =
                rt::RandnTensor(metadata->type().tensor_type().elem_type(), shape, seed);
            tensor.set_data_type(generated.data_type);
            for (const auto size : generated.shape)
              tensor.add_dims(size);
            tensor.set_raw_data(generated.bytes(), generated.size_bytes());
          } else if (!tensor.ParseFromString(input.protobuf)) {
            throw std::runtime_error("Could not parse input TensorProto.");
          }
          tensor.set_name(input.name);
        }
        if (tensor.name().empty() || !names.emplace(tensor.name()).second)
          throw std::runtime_error("Expected unique, named input tensors.");
        if (!input_names.contains(std::string(tensor.name())))
          throw std::runtime_error("Unknown model input: " + std::string(tensor.name()));
        if (!tensor.external_data().empty() || tensor.data_location() == ol::TensorProto::EXTERNAL)
          throw std::runtime_error("Input tensors must contain inline values.");
        for (const auto &value : snapshot_.graph().input()) {
          if (value.name() != tensor.name() || !value.type().has_tensor_type())
            continue;
          size_t axis = 0;
          for (const auto &dim : value.type().tensor_type().shape().dim()) {
            const auto key = dim.has_dim_param() && !dim.dim_param().empty()
                                 ? std::string(dim.dim_param())
                                 : std::string(value.name()) + ":" + std::to_string(axis);
            const auto found = dimensions_.find(key);
            if (!dim.has_dim_value() &&
                (input.random || !input.protobuf.empty() || !dimensions_.empty())) {
              if (found == dimensions_.end())
                throw std::runtime_error("Specify dynamic dimension: " + key);
              if (axis >= tensor.dims().size() || tensor.dims()[axis] != found->second)
                throw std::runtime_error("Input shape does not match dimension: " + key);
            }
            ++axis;
          }
        }
        if (prepare_) {
          std::string bytes;
          if (!tensor.SerializeToString(bytes))
            throw std::runtime_error("Could not serialize input tensor.");
          prepared_inputs_.push_back(std::move(bytes));
        } else
          context.Set(std::string(tensor.name()), rt::TensorFromProto(tensor));
      }
      RestoreModelPayloads(snapshot_, *source_, source_->file_path);
      std::vector<ol::EncodedValueProto *> encoded;
      const auto tensors = ModelSaveTensors(snapshot_, &encoded);
      const auto directory = ModelFilePath(source_->file_path).parent_path();
      for (auto *tensor : tensors)
        if (tensor->data_location() == ol::TensorProto::EXTERNAL)
          LoadExternalTensor(*tensor, directory);
      for (auto *tensor : encoded)
        if (tensor->data_location() == ol::TensorProto::EXTERNAL)
          LoadExternalTensor(*tensor, directory);
      if (prepare_) {
        if (!snapshot_.SerializeToString(model_bytes_))
          throw std::runtime_error("Could not serialize execution model.");
        return;
      }
      rt::RuntimeSession session(snapshot_, verbose_);
      const auto start = std::chrono::steady_clock::now();
      session.Run(context);
      time_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count();
      for (const auto &output : snapshot_.graph().output()) {
        const std::string name = output.name();
        if (!context.Has(name)) {
          outputs_.emplace_back(name, "Non-tensor output (no tensor preview).");
          continue;
        }
        const auto &tensor = context.Get(name);
        ol::TensorProto proto;
        proto.set_name(name);
        proto.set_data_type(tensor.data_type);
        for (const auto dimension : tensor.shape)
          proto.add_dims(dimension);
        if (tensor.data_type == ol::TensorProto::STRING) {
          if (tensor.AsStrings().size() <= 32)
            for (const auto &value : tensor.AsStrings())
              proto.add_string_data(value);
        } else {
          proto.ref_raw_data().assign_borrowed(tensor.bytes(), tensor.size_bytes());
        }
        const auto values = TinyTensorValues(proto, 33);
        ol::ValueInfoProto metadata;
        rt::FillValueInfo(tensor, metadata);
        metadata.set_name(name);
        outputs_.emplace_back(name, ol::ProtoToString(metadata) + "\nValues: " +
                                        values.value_or("Omitted (preview limit: 32 elements)."));
      }
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override {
    auto result = Napi::Object::New(Env());
    if (prepare_) {
      result.Set("model",
                 Napi::Buffer<char>::Copy(Env(), model_bytes_.data(), model_bytes_.size()));
      auto inputs = Napi::Array::New(Env(), prepared_inputs_.size());
      for (size_t index = 0; index < prepared_inputs_.size(); ++index)
        inputs.Set(index, Napi::Buffer<char>::Copy(Env(), prepared_inputs_[index].data(),
                                                   prepared_inputs_[index].size()));
      result.Set("inputs", inputs);
      deferred_.Resolve(result);
      return;
    }
    result.Set("timeMs", time_ms_);
    auto outputs = Napi::Array::New(Env(), outputs_.size());
    for (size_t index = 0; index < outputs_.size(); ++index) {
      auto output = Napi::Object::New(Env());
      output.Set("name", Text(Env(), outputs_[index].first));
      output.Set("text", Text(Env(), outputs_[index].second));
      outputs.Set(index, output);
    }
    result.Set("outputs", outputs);
    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  Napi::Promise::Deferred deferred_;
  std::shared_ptr<StoredModel> source_;
  ol::ModelProto snapshot_;
  std::vector<InputSource> inputs_;
  std::map<std::string, int64_t> dimensions_;
  int verbose_;
  bool prepare_;
  std::string model_bytes_;
  std::vector<std::string> prepared_inputs_;
  std::vector<std::pair<std::string, std::string>> outputs_;
  double time_ms_ = 0;
};
} // namespace

void AddRunMethods(Napi::Env env, Napi::Object result, std::shared_ptr<StoredModel> model) {
  result.Set("runInputs",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 RootPath(call, 0);
                 auto inputs = Napi::Array::New(call.Env());
                 std::set<std::string> initializers;
                 for (const auto &value : model->model->graph().initializer())
                   initializers.emplace(value.name());
                 for (const auto &value : model->model->graph().sparse_initializer())
                   initializers.emplace(value.values().name());
                 for (const auto &value : model->model->graph().extended_initializer())
                   if (value.has_encoded_value())
                     initializers.emplace(value.encoded_value().name());
                 uint32_t index = 0;
                 for (const auto &value : model->model->graph().input()) {
                   if (initializers.contains(std::string(value.name())))
                     continue;
                   auto input = Napi::Object::New(call.Env());
                   input.Set("name", Text(call.Env(), value.name()));
                   input.Set("text", Text(call.Env(), ol::ProtoToString(value) + " = {}"));
                   if (value.type().has_tensor_type() && value.type().tensor_type().has_shape()) {
                     auto shape = Napi::Array::New(call.Env());
                     uint32_t axis = 0;
                     for (const auto &dim : value.type().tensor_type().shape().dim())
                       shape.Set(axis++, dim.has_dim_value()
                                             ? Text(call.Env(), std::to_string(dim.dim_value()))
                                         : dim.has_dim_param() && !dim.dim_param().empty()
                                             ? Text(call.Env(), dim.dim_param())
                                             : Napi::Value(call.Env().Null()));
                     input.Set("shape", shape);
                   }
                   inputs.Set(index++, input);
                 }
                 return inputs;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  for (const bool prepare : {false, true})
    result.Set(
        prepare ? "prepareRun" : "run",
        Napi::Function::New(env, [model, prepare](const Napi::CallbackInfo &call) -> Napi::Value {
          try {
            RootPath(call, 1);
            if (call.Length() < 1 || !call[0].IsArray())
              throw std::runtime_error("Expected input tensor text.");
            const auto values = call[0].As<Napi::Array>();
            std::vector<InputSource> inputs;
            for (uint32_t index = 0; index < values.Length(); ++index) {
              const auto value = values.Get(index);
              if (value.IsString()) {
                inputs.push_back({{}, value.As<Napi::String>().Utf8Value(), {}, false});
                continue;
              }
              if (!value.IsObject() || value.IsArray())
                throw std::runtime_error("Expected input tensor text.");
              const auto input = value.As<Napi::Object>();
              if (!input.Get("name").IsString() || !input.Get("source").IsString())
                throw std::runtime_error("Expected a named input source.");
              InputSource source;
              source.name = input.Get("name").As<Napi::String>().Utf8Value();
              const auto kind = input.Get("source").As<Napi::String>().Utf8Value();
              source.random = kind == "random";
              if (!source.random) {
                if (kind != "protobuf" || !input.Get("data").IsTypedArray() ||
                    input.Get("data").As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array)
                  throw std::runtime_error("Expected a TensorProto buffer or random input.");
                const auto bytes = input.Get("data").As<Napi::Uint8Array>();
                source.protobuf.assign(reinterpret_cast<const char *>(bytes.Data()),
                                       bytes.ByteLength());
              }
              inputs.push_back(std::move(source));
            }
            int verbose = 0;
            std::map<std::string, int64_t> dimensions;
            if (call.Length() > 2 && !call[2].IsUndefined()) {
              if (!call[2].IsObject() || call[2].IsArray())
                throw std::runtime_error("Expected run options.");
              const auto options = call[2].As<Napi::Object>();
              if (options.Has("verbose") && !options.Get("verbose").IsUndefined()) {
                const auto number = Dimension(options.Get("verbose"));
                if (number > 10)
                  throw std::runtime_error("Verbosity must be between 0 and 10.");
                verbose = static_cast<int>(number);
              }
              if (options.Has("dimensions")) {
                const auto value = options.Get("dimensions");
                if (!value.IsObject() || value.IsArray())
                  throw std::runtime_error("Expected dynamic dimensions.");
                const auto values = value.As<Napi::Object>();
                const auto keys = values.GetPropertyNames();
                for (uint32_t index = 0; index < keys.Length(); ++index) {
                  const auto key = keys.Get(index).As<Napi::String>();
                  dimensions.emplace(key.Utf8Value(), Dimension(values.Get(key)));
                }
              }
            }
            auto *worker = new RunWorker(call.Env(), model, std::move(inputs),
                                         std::move(dimensions), verbose, prepare);
            const auto promise = worker->Promise();
            worker->Queue();
            return promise;
          } catch (const std::exception &error) {
            Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
            return call.Env().Undefined();
          }
        }));
}
} // namespace onnx_light_viewer
