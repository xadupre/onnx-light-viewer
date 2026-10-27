/** @file Retained model handles and their shared ownership. */
#include "model.h"
#include "graph.h"
#include <onnx_proto/onnx_verify.h>
#include <stdexcept>

namespace onnx_light_viewer {
std::shared_ptr<StoredModel> CloneStoredModel(const std::shared_ptr<StoredModel> &model) {
  auto copy = std::make_shared<StoredModel>();
  copy->backing = model;
  copy->model = std::make_shared<ol::ModelProto>(model->model->SharedCopy());
  copy->file_name = model->file_name;
  copy->file_path = model->file_path;
  copy->pipeline_stage = model->pipeline_stage;
  copy->source_size = model->source_size;
  copy->source_time = model->source_time;
  copy->skipped = model->skipped;
  copy->has_training_info = model->has_training_info;
  copy->external_weights = model->external_weights;
  copy->fusion_base = model->fusion_base;
  copy->fusion_previous = model->fusion_previous;
  copy->fusion_history = model->fusion_history;
  copy->optimization_report = model->optimization_report;
  copy->optimization_wall_time_ns = model->optimization_wall_time_ns;
  copy->rewriting_base = model->rewriting_base;
  copy->rewriting_name = model->rewriting_name;
  copy->rewriting_batches = model->rewriting_batches;
  for (const auto &plan : model->fusion_plans)
    copy->fusion_plans.push_back(plan.SharedCopy());
  return copy;
}

/** Method closures share ownership, even if JavaScript retains only one method. */
Napi::Object ModelHandle(Napi::Env env, std::shared_ptr<StoredModel> model) {
  auto result = Napi::Object::New(env);
  result.Set("clone",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 return ModelHandle(call.Env(), CloneStoredModel(model));
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), error.what()).ThrowAsJavaScriptException();
                 return call.Env().Undefined();
               }
             }));
  AddFusionMethods(env, result, model);
  AddSerializationMethods(env, result, model);
  AddMemoryProfileMethods(env, result, model);
  AddTensorStatisticsMethods(env, result, model);
  result.Set("verify",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               try {
                 ol::VerifyModel(*model->model);
               } catch (const std::exception &error) {
                 Napi::Error::New(call.Env(), std::string("VerifyModel: ") + error.what())
                     .ThrowAsJavaScriptException();
               }
               return call.Env().Undefined();
             }));
  result.Set(
      "summary", Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
        const auto env = call.Env();
        try {
          if (call.Length() > 1 ||
              (call.Length() == 1 && !call[0].IsUndefined() && !call[0].IsBoolean()))
            throw std::runtime_error("Expected an optional inference flag.");
          if (call.Length() == 1 && call[0].IsBoolean() && call[0].As<Napi::Boolean>().Value())
            InferShapes(*model->model);
          return ModelSummary(env, *model->model, model->file_name);
        } catch (const std::exception &error) {
          Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
          return env.Undefined();
        }
      }));
  result.Set("metadata",
             Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
               const auto env = call.Env();
               if (call.Length() > 1 ||
                   (call.Length() == 1 && !call[0].IsUndefined() && !call[0].IsArray())) {
                 Napi::TypeError::New(env, "Expected an optional subgraph path.")
                     .ThrowAsJavaScriptException();
                 return env.Undefined();
               }
               try {
                 return ModelMetadata(env, *model->model, ReadPath(call[0]));
               } catch (const std::exception &error) {
                 Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
                 return env.Undefined();
               }
             }));
  result.Set(
      "graph", Napi::Function::New(env, [model](const Napi::CallbackInfo &call) -> Napi::Value {
        const auto env = call.Env();
        if (call.Length() > 2 ||
            (call.Length() >= 1 && !call[0].IsUndefined() && !call[0].IsBoolean()) ||
            (call.Length() == 2 && !call[1].IsUndefined() && !call[1].IsArray())) {
          Napi::TypeError::New(env, "Expected an optional inference flag and subgraph path.")
              .ThrowAsJavaScriptException();
          return env.Undefined();
        }
        try {
          const auto path = ReadPath(call[1]);
          GraphScopes(*model->model, path);
          if (call.Length() >= 1 && call[0].IsBoolean() && call[0].As<Napi::Boolean>().Value()) {
            InferShapes(*model->model);
          }
          return ModelGraph(env, *model->model, model->file_name, path);
        } catch (const std::exception &error) {
          Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
          return env.Undefined();
        }
      }));
  return result;
}

} // namespace onnx_light_viewer
