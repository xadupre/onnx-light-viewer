#pragma once

#include <napi.h>
#include <onnx_proto/onnx.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {
struct TestCase;
}

namespace ONNX_LIGHT_NAMESPACE::core::builder {
struct OptimizationReport;
}

namespace onnx_light_viewer {
namespace ol = ONNX_LIGHT_NAMESPACE;

inline constexpr std::string_view skippedPayloadLocation = "onnx-light-viewer:payload-omitted";

struct ProtoPathStep {
  uint32_t field;
  size_t index;
  bool operator==(const ProtoPathStep &) const = default;
};

struct SkippedTensorPayload {
  std::vector<ProtoPathStep> path;
  std::set<uint32_t> fields;
};

/**
 * Owns the parsed model and its backing storage. Declaration order keeps the
 * source buffer or backend case alive until after the model is destroyed.
 */
struct StoredModel {
  struct ExternalWeight {
    ol::TensorProto reference;
    ol::TensorProto value;
  };
  struct FusionMatch {
    std::string pattern;
    std::vector<size_t> nodes;
  };
  Napi::Reference<Napi::Uint8Array> source;
  std::shared_ptr<StoredModel> backing;
  std::shared_ptr<ol::core::backend_test::TestCase> backend_case;
  std::shared_ptr<ol::ModelProto> model;
  std::string file_name;
  std::string file_path;
  std::string pipeline_stage;
  uintmax_t source_size = 0;
  std::filesystem::file_time_type source_time;
  std::vector<SkippedTensorPayload> skipped;
  bool has_training_info = false;
  std::vector<ExternalWeight> external_weights;
  std::shared_ptr<StoredModel> fusion_base;
  std::shared_ptr<StoredModel> fusion_previous;
  std::vector<ol::RewritingProto> fusion_plans;
  std::vector<std::vector<FusionMatch>> fusion_history;
  std::shared_ptr<const ol::core::builder::OptimizationReport> optimization_report;
  int64_t optimization_wall_time_ns = 0;
  std::shared_ptr<StoredModel> rewriting_base;
  std::string rewriting_name;
  std::vector<size_t> rewriting_batches;
};

std::shared_ptr<StoredModel> CloneStoredModel(const std::shared_ptr<StoredModel> &model);
Napi::Object ModelHandle(Napi::Env env, std::shared_ptr<StoredModel> model);
void AddSerializationMethods(Napi::Env env, Napi::Object result,
                             std::shared_ptr<StoredModel> model);
void AddFusionMethods(Napi::Env env, Napi::Object result, std::shared_ptr<StoredModel> model);
void AddMemoryProfileMethods(Napi::Env env, Napi::Object result,
                             std::shared_ptr<StoredModel> model);
void AddTensorStatisticsMethods(Napi::Env env, Napi::Object result,
                                std::shared_ptr<StoredModel> model);

} // namespace onnx_light_viewer
