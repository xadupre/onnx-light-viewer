/** @file One-shot execution of retained models through the native SDK. */
#include "model_io.h"
#include "model_storage.h"
#include "tensor_preview.h"
#include <chrono>
#include <onnx_core/runtime/runtime_session.h>
#include <onnx_manipulations/printer.h>
#include <set>
#include <stdexcept>

namespace onnx_light_viewer {
namespace {
namespace rt = ol::core::runtime;

void RootPath(const Napi::CallbackInfo &call, size_t index) {
  if (call.Length() > index + 1 ||
      (call.Length() > index && !call[index].IsUndefined() &&
       (!call[index].IsArray() || call[index].As<Napi::Array>().Length() != 0)))
    throw std::runtime_error("Run covers the whole model; choose a stage root for pipelines.");
}

class RunWorker : public Napi::AsyncWorker {
public:
  RunWorker(Napi::Env env, std::shared_ptr<StoredModel> source, std::vector<std::string> inputs)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        source_(std::move(source)), snapshot_(source_->model->SharedCopy()),
        inputs_(std::move(inputs)) {}

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      rt::RuntimeContext context;
      std::set<std::string> names;
      std::set<std::string> input_names;
      for (const auto &input : snapshot_.graph().input())
        input_names.emplace(input.name());
      for (const auto &text : inputs_) {
        ol::TensorProto tensor;
        ol::OnnxParser parser(text);
        const auto status = parser.Parse(tensor);
        if (!status.IsOK())
          throw std::runtime_error(status.ErrorMessage());
        if (!parser.EndOfInput())
          throw std::runtime_error("Unexpected text after the input tensor.");
        if (tensor.name().empty() || !names.emplace(tensor.name()).second)
          throw std::runtime_error("Expected unique, named input tensors.");
        if (!input_names.contains(std::string(tensor.name())))
          throw std::runtime_error("Unknown model input: " + std::string(tensor.name()));
        if (!tensor.external_data().empty() || tensor.data_location() == ol::TensorProto::EXTERNAL)
          throw std::runtime_error("Input tensors must contain inline values.");
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
      rt::RuntimeSession session(snapshot_);
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
  std::vector<std::string> inputs_;
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
                   inputs.Set(index++, input);
                 }
                 return inputs;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  result.Set("run",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 RootPath(call, 1);
                 if (call.Length() < 1 || !call[0].IsArray())
                   throw std::runtime_error("Expected input tensor text.");
                 const auto values = call[0].As<Napi::Array>();
                 std::vector<std::string> inputs;
                 for (uint32_t index = 0; index < values.Length(); ++index) {
                   const auto value = values.Get(index);
                   if (!value.IsString())
                     throw std::runtime_error("Expected input tensor text.");
                   inputs.push_back(value.As<Napi::String>().Utf8Value());
                 }
                 auto *worker = new RunWorker(call.Env(), model, std::move(inputs));
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
