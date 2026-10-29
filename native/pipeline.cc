/** @file PipelineProto containers reuse retained ModelProto stage handles. */
#include "pipeline.h"
#include "addon.h"
#include "model_storage.h"
#include "tensor_preview.h"
#include <cerrno>
#include <charconv>
#include <cmath>
#include <limits>
#include <regex>
#include <set>
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

class PromptPipelineWorker : public Napi::AsyncWorker {
public:
  PromptPipelineWorker(Napi::Env env, std::string prefill, std::string embedding,
                       std::string directory, int output)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        prefill_(std::move(prefill)), embedding_(std::move(embedding)),
        directory_(std::move(directory)), output_(output) {}

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      ol::PipelineProto pipeline;
      pipeline.set_name("Mobius generation pipeline");
      pipeline.set_producer_name("onnx-light-viewer");
      const auto read = [&](ol::ModelProto &model, const std::string &path) {
        FileDescriptor file(OpenModelFile(path));
        if (file.get() < 0)
          throw std::system_error(errno, std::generic_category(), "Could not open " + path);
        ol::utils::FdReadStream stream(file.get());
        ol::ParseOptions options;
        options.skip_raw_data = output_ < 0;
        ReadModel(model, stream, options);
      };
      read(pipeline.ref_prefill(), prefill_);
      auto &graph = pipeline.ref_prefill().ref_graph();
      const auto names = [](const auto &ports) {
        std::set<std::string> result;
        for (const auto &port : ports)
          result.emplace(port.name());
        return result;
      };
      auto inputs = names(graph.input());
      const auto outputs = names(graph.output());
      if (!outputs.contains("logits")) {
        reason_ = "No logits output: this is not a token-generation decoder.";
        return;
      }
      std::set<std::string> supplied{"input_ids", "attention_mask", "position_ids"};
      if (!inputs.contains("input_ids")) {
        if (embedding_.empty()) {
          reason_ = "Separate embedding model is missing.";
          return;
        }
        read(pipeline.ref_embedding(), embedding_);
        const auto &embedding = pipeline.embedding().graph();
        if (!names(embedding.input()).contains("input_ids")) {
          reason_ = "Embedding model does not accept input_ids.";
          return;
        }
        auto embedding_supplied = supplied;
        for (const auto &initializer : embedding.initializer())
          embedding_supplied.emplace(initializer.name());
        for (const auto &input : embedding.input())
          if (!embedding_supplied.contains(std::string(input.name()))) {
            reason_ = "Embedding requires additional input: " + std::string(input.name());
            return;
          }
        for (const auto &name : names(embedding.output()))
          supplied.insert(name);
      }
      const auto initializers = names(graph.initializer());
      std::set<std::string> bound;
      for (const auto &binding : graph.persistent_bindings())
        bound.emplace(binding.input_name());
      const std::regex cache_name(R"(^past_key_values\.([0-9]+)\.(key|value)$)");
      for (const auto &input : graph.input()) {
        const std::string name = input.name();
        if (supplied.contains(name))
          continue;
        std::smatch match;
        if (!std::regex_match(name, match, cache_name)) {
          if (initializers.contains(name) || bound.contains(name))
            continue;
          reason_ = "Decoder requires additional input: " + name;
          return;
        }
        const std::string present = "present." + match[1].str() + "." + match[2].str();
        if (!outputs.contains(present) || !input.type().has_tensor_type()) {
          reason_ = "Missing cache output/type for " + name;
          return;
        }
        if (!bound.contains(name)) {
          auto *binding = graph.add_persistent_bindings();
          binding->set_input_name(name);
          binding->set_output_name(present);
        }
        if (initializers.contains(name))
          continue;
        const auto &type = input.type().tensor_type();
        const auto &dimensions = type.shape().dim();
        if (dimensions.size() != 4 || !dimensions[1].has_dim_value() ||
            !dimensions[3].has_dim_value() ||
            (dimensions[0].has_dim_value() && dimensions[0].dim_value() != 1) ||
            (dimensions[2].has_dim_value() && dimensions[2].dim_value() != 0)) {
          reason_ = "Cache needs an explicit empty [batch, heads, past, width] layout: " + name;
          return;
        }
        // Declare an empty initial cache; generation and feedback remain SDK-owned.
        auto *initial = graph.add_initializer();
        initial->set_name(name);
        initial->set_data_type(type.elem_type());
        for (const auto dimension :
             {int64_t{1}, dimensions[1].dim_value(), int64_t{0}, dimensions[3].dim_value()})
          initial->add_dims(dimension);
      }
      std::map<std::filesystem::path, std::string> locations;
      const auto references = [&](ol::ModelProto &model, const std::string &path) {
        std::vector<ol::EncodedValueProto *> encoded;
        const auto tensors = ModelSaveTensors(model, &encoded);
        const auto rebase = [&](auto &tensor) {
          for (auto &entry : tensor.ref_external_data())
            if (entry.key() == "location") {
              const auto source = (ModelFilePath(path).parent_path() / std::string(entry.value()))
                                      .lexically_normal();
              auto found = locations.find(source);
              if (found == locations.end()) {
                if (!std::filesystem::is_regular_file(source))
                  throw std::runtime_error("Missing external weights: " + source.string());
                const auto relative =
                    source.lexically_relative(ModelFilePath(directory_).lexically_normal());
                if (relative.empty() || *relative.begin() == "..")
                  throw std::runtime_error(
                      "Pipeline weights must remain inside the export package.");
                found = locations.emplace(source, relative.generic_string()).first;
              }
              entry.set_value(found->second);
            }
        };
        for (auto *tensor : tensors)
          rebase(*tensor);
        for (auto *value : encoded)
          rebase(*value);
      };
      references(pipeline.ref_prefill(), prefill_);
      if (pipeline.has_embedding())
        references(pipeline.ref_embedding(), embedding_);
      if (output_ >= 0 && !pipeline.SerializeToFileDescriptor(output_))
        throw std::runtime_error("Could not write the generation pipeline.");
      available_ = true;
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override {
    auto result = Napi::Object::New(Env());
    result.Set("available", available_);
    result.Set("reason", reason_);
    deferred_.Resolve(result);
  }
  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  Napi::Promise::Deferred deferred_;
  std::string prefill_, embedding_, directory_, reason_;
  int output_;
  bool available_ = false;
};

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

Napi::Value PreparePromptPipeline(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  try {
    if (info.Length() != 4 || !info[0].IsString() || !info[1].IsString() || !info[2].IsString() ||
        (!info[3].IsNull() && !info[3].IsNumber()))
      throw std::runtime_error("Expected pipeline source paths and an optional output descriptor.");
    std::vector<std::string> paths;
    for (size_t i = 0; i < 3; ++i) {
      auto path = info[i].As<Napi::String>().Utf8Value();
      if (path.find('\0') != std::string::npos)
        throw std::runtime_error("Invalid pipeline source path.");
      paths.push_back(std::move(path));
    }
    int output = -1;
    if (!info[3].IsNull()) {
      const auto number = info[3].As<Napi::Number>().DoubleValue();
      if (!std::isfinite(number) || std::floor(number) != number || number < 0 ||
          number > std::numeric_limits<int>::max())
        throw std::runtime_error("Invalid pipeline output descriptor.");
      output = static_cast<int>(number);
    }
    auto worker = std::make_unique<PromptPipelineWorker>(env, paths[0], paths[1], paths[2], output);
    const auto promise = worker->Promise();
    worker->Queue();
    worker.release();
    return promise;
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

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
