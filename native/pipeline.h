#pragma once

#include "model_io.h"
#include <map>

namespace onnx_light_viewer {

struct StoredPipeline {
  ol::PipelineProto proto;
  std::map<std::string, std::shared_ptr<StoredModel>> models;
};

ol::ModelProto &PipelineStage(ol::PipelineProto &pipeline, const std::string &role);
std::shared_ptr<StoredPipeline> ReadPipeline(ol::utils::BinaryStream &source,
                                             const std::string &file_name,
                                             const std::string &file_path = "");
Napi::Object PipelineHandle(Napi::Env env, std::shared_ptr<StoredPipeline> pipeline);

} // namespace onnx_light_viewer
