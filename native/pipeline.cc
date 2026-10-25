/** @file PipelineProto containers reuse retained ModelProto stage handles. */
#include "pipeline.h"
#include "addon.h"
#include "tensor_preview.h"
#include <cerrno>
#include <charconv>
#include <stdexcept>
#include <system_error>

namespace onnx_light_viewer {

ol::ModelProto &PipelineStage(ol::PipelineProto &pipeline, const std::string &role) {
  if (role == "embedding")
    return pipeline.ref_embedding();
  if (role == "prefill")
    return pipeline.ref_prefill();
  if (role == "decode")
    return pipeline.ref_decode();
  throw std::runtime_error("Unknown pipeline stage: " + role);
}

namespace {

Napi::Array PipelineMetadata(Napi::Env env, const ol::PipelineProto &pipeline) {
  auto result = Napi::Array::New(env);
  const auto add = [&](std::string_view key, std::string_view value) {
    auto entry = Napi::Object::New(env);
    entry.Set("key", Text(env, key));
    entry.Set("value", Text(env, value));
    result.Set(result.Length(), entry);
  };
  const auto add_number = [&](std::string_view key, const auto *value) {
    if (!value)
      return;
    char buffer[64];
    const auto [end, error] = std::to_chars(buffer, buffer + sizeof(buffer), *value);
    if (error != std::errc{})
      throw std::system_error(std::make_error_code(error), "Could not format pipeline metadata");
    add(key, std::string_view(buffer, end - buffer));
  };
  if (pipeline.has_name())
    add("name", pipeline.name());
  if (pipeline.has_producer_name())
    add("producer_name", pipeline.producer_name());
  if (pipeline.has_producer_version())
    add("producer_version", pipeline.producer_version());
  if (pipeline.has_domain())
    add("domain", pipeline.domain());
  add_number("pipeline_version", pipeline.ptr_pipeline_version());
  if (pipeline.has_doc_string())
    add("doc_string", pipeline.doc_string());
  add_number("max_new_tokens", pipeline.ptr_max_new_tokens());
  add_number("temperature", pipeline.ptr_temperature());
  add_number("seed", pipeline.ptr_seed());
  add_number("eos_token_id", pipeline.ptr_eos_token_id());
  add_number("pad_token_id", pipeline.ptr_pad_token_id());
  add_number("top_k", pipeline.ptr_top_k());
  add_number("top_p", pipeline.ptr_top_p());
  if (pipeline.has_input_ids_name())
    add("input_ids_name", pipeline.input_ids_name());
  if (pipeline.has_logits_name())
    add("logits_name", pipeline.logits_name());
  if (pipeline.has_attention_mask_name())
    add("attention_mask_name", pipeline.attention_mask_name());
  if (pipeline.has_position_ids_name())
    add("position_ids_name", pipeline.position_ids_name());
  for (const auto &entry : pipeline.metadata_props())
    add("metadata_props." + std::string(entry.key()), entry.value());
  return result;
}

class AssemblePipelineWorker : public Napi::AsyncWorker {
public:
  AssemblePipelineWorker(Napi::Env env, ol::PipelineProto descriptor,
                         std::map<std::string, std::string> stages)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        descriptor_(std::move(descriptor)), stages_(std::move(stages)) {}

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      for (const auto &[role, bytes] : stages_)
        if (!PipelineStage(descriptor_, role).ParseFromString(bytes))
          throw std::runtime_error("Could not decode the saved pipeline stage: " + role);
      if (!descriptor_.SerializeToString(bytes_))
        throw std::runtime_error("Could not serialize PipelineProto.");
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override {
    deferred_.Resolve(Napi::Buffer<uint8_t>::Copy(
        Env(), reinterpret_cast<const uint8_t *>(bytes_.data()), bytes_.size()));
  }
  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  Napi::Promise::Deferred deferred_;
  ol::PipelineProto descriptor_;
  std::map<std::string, std::string> stages_;
  std::string bytes_;
};

class LoadPipelineWorker : public Napi::AsyncWorker {
public:
  LoadPipelineWorker(Napi::Env env, std::string path, std::string name)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        path_(std::move(path)), name_(std::move(name)) {}

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      FileDescriptor file(OpenModelFile(path_));
      if (file.get() < 0)
        throw std::system_error(errno, std::generic_category(), "Could not open pipeline");
      ol::utils::FdReadStream stream(file.get());
      pipeline_ = ReadPipeline(stream, name_, path_);
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override { deferred_.Resolve(PipelineHandle(Env(), pipeline_)); }
  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  Napi::Promise::Deferred deferred_;
  std::string path_;
  std::string name_;
  std::shared_ptr<StoredPipeline> pipeline_;
};

} // namespace

Napi::Object PipelineHandle(Napi::Env env, std::shared_ptr<StoredPipeline> pipeline) {
  auto result = Napi::Object::New(env);
  result.Set("name", Text(env, pipeline->proto.name()));
  result.Set("metadata", PipelineMetadata(env, pipeline->proto));
  auto models = Napi::Object::New(env);
  for (const auto &[role, model] : pipeline->models)
    models.Set(role, ModelHandle(env, model));
  result.Set("models", models);
  result.Set("assemble",
             Napi::Function::New(env, [pipeline](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 if (call.Length() != 1 || !call[0].IsObject())
                   throw std::runtime_error("Expected pipeline stage buffers.");
                 const auto models = call[0].As<Napi::Object>();
                 if (models.GetPropertyNames().Length() != pipeline->models.size())
                   throw std::runtime_error("Pipeline stages must be preserved.");
                 std::map<std::string, std::string> stages;
                 for (const auto &[role, model] : pipeline->models) {
                   (void)model;
                   const auto value = models.Get(role);
                   if (!value.IsTypedArray() ||
                       value.As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array)
                     throw std::runtime_error("Missing pipeline stage: " + role);
                   const auto bytes = value.As<Napi::Uint8Array>();
                   stages.emplace(role, std::string(reinterpret_cast<const char *>(bytes.Data()),
                                                    bytes.ByteLength()));
                 }
                 auto worker = std::make_unique<AssemblePipelineWorker>(
                     call.Env(), pipeline->proto.SharedCopy(), std::move(stages));
                 const auto promise = worker->Promise();
                 worker->Queue();
                 worker.release();
                 return promise;
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  return result;
}

Napi::Value LoadPipeline(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  try {
    if (info.Length() != 2 || !info[0].IsTypedArray() ||
        info[0].As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array || !info[1].IsString())
      throw std::runtime_error("Expected pipeline bytes and a file name.");
    const auto bytes = info[0].As<Napi::Uint8Array>();
    ol::utils::StringStream stream(bytes.Data(), bytes.ByteLength());
    return PipelineHandle(env, ReadPipeline(stream, info[1].As<Napi::String>().Utf8Value()));
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value LoadPipelineFile(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  try {
    if (info.Length() != 2 || !info[0].IsString() || !info[1].IsString())
      throw std::runtime_error("Expected a pipeline path and a file name.");
    const auto path = info[0].As<Napi::String>().Utf8Value();
    if (path.find('\0') != std::string::npos)
      throw std::runtime_error("Invalid pipeline file path.");
    auto worker =
        std::make_unique<LoadPipelineWorker>(env, path, info[1].As<Napi::String>().Utf8Value());
    const auto promise = worker->Promise();
    worker->Queue();
    worker.release();
    return promise;
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

} // namespace onnx_light_viewer
