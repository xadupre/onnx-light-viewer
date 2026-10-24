#pragma once

#include "model.h"

#include <optional>

namespace onnx_light_viewer {

std::filesystem::path ModelFilePath(const std::string &path);
int OpenModelFile(const std::string &path);

class FileDescriptor {
public:
  explicit FileDescriptor(int fd) : fd_(fd) {}
  ~FileDescriptor();
  int get() const { return fd_; }

private:
  int fd_;
};

void ReadModel(ol::ModelProto &model, ol::utils::BinaryStream &stream, ol::ParseOptions options);
void ReadModel(ol::ModelProto &model, const Napi::Uint8Array &bytes, bool skip_raw_data);
std::optional<std::string> ReadPassword(const Napi::Value &value);
ol::TensorProto *FindTensor(ol::ModelProto &model, const SkippedTensorPayload &payload);
void RestoreModelPayloads(ol::ModelProto &snapshot, const StoredModel &source,
                          const std::string &file_path);

} // namespace onnx_light_viewer
