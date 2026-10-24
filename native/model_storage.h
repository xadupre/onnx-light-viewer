#pragma once

#include "model.h"

namespace onnx_light_viewer {

template <typename Proto> std::string ExternalDataLocation(const Proto &tensor) {
  std::string location;
  for (const auto &entry : tensor.external_data())
    if (entry.key() == "location")
      location = entry.value();
  return location;
}

void LoadExternalTensor(ol::TensorProto &tensor, const std::filesystem::path &directory);
void LoadExternalTensor(ol::EncodedValueProto &tensor, const std::filesystem::path &directory);
std::vector<ol::TensorProto *>
ModelSaveTensors(ol::ModelProto &model, std::vector<ol::EncodedValueProto *> *encoded = nullptr,
                 std::set<const ol::TensorProto *> *layout_tensors = nullptr);

} // namespace onnx_light_viewer
