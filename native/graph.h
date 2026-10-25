#pragma once

#include <napi.h>
#include <onnx_proto/onnx.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace onnx_light_viewer {
namespace ol = ONNX_LIGHT_NAMESPACE;

struct GraphStep {
  int node; // -1 opens a function definition without a call site.
  int attribute;
  int graph;
  int function = -1;
};
using GraphPath = std::vector<GraphStep>;

struct GraphSelection {
  std::vector<std::unique_ptr<ol::GraphProto>> functions;
  std::vector<const ol::GraphProto *> scopes;
  const ol::FunctionProto *function = nullptr;
  size_t capture_start = 0;
};

std::string Domain(std::string_view domain);
GraphPath ReadPath(const Napi::Value &value);
Napi::Array PathValue(Napi::Env env, const GraphPath &path);
std::string FunctionName(const ol::FunctionProto &function);
GraphSelection GraphScopes(const ol::ModelProto &model, const GraphPath &path);
Napi::Value Schema(Napi::Env env, const std::string &op_type, const std::string &domain,
                   int64_t opset);
Napi::Object ModelGraph(Napi::Env env, const ol::ModelProto &model, const std::string &file_name,
                        const GraphPath &path = {});
Napi::Object ModelSummary(Napi::Env env, const ol::ModelProto &model, const std::string &file_name);
Napi::Object ModelMetadata(Napi::Env env, const ol::ModelProto &model, const GraphPath &path);
void InferShapes(ol::ModelProto &model);

} // namespace onnx_light_viewer
