/** @file Model sources, bounded protobuf parsing and payload recovery. */
#include "model_io.h"
#include "addon.h"
#include "graph.h"
#include "pipeline.h"
#include "tensor_preview.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <limits>
#include <map>
#include <onnx_core/backend_test/test_case.h>
#include <stdexcept>
#include <system_error>
#include <tuple>

#ifdef ONNX_LIGHT_HAS_OPENSSL
#include <openssl/evp.h>
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace onnx_light_viewer {
namespace {

constexpr int64_t previewPayloadThreshold = 256; // Retain up to 15 complex128 elements.

enum class ProtoKind {
  Pipeline,
  Model,
  Graph,
  Node,
  Attribute,
  SparseTensor,
  Function,
  Tensor,
  Opaque
};

struct ProtoFrame {
  ProtoKind kind;
  std::vector<ProtoPathStep> path;
  uint32_t field = 0;
  uint8_t wire_type = 0;
  std::map<uint32_t, size_t> child_counts;
  std::set<uint32_t> singular_children;
  std::map<uint32_t, uint64_t> payload_sizes;
  std::set<uint32_t> skipped_fields;
  bool packed_values = false;
};

struct ChildProto {
  ProtoKind kind;
  bool repeated;
};

struct LimitScope {
  bool child = false;
  bool packed = false;
};

std::optional<ChildProto> ChildMessage(ProtoKind parent, uint32_t field) {
  switch (parent) {
  case ProtoKind::Pipeline:
    if (field >= 7 && field <= 9)
      return ChildProto{ProtoKind::Model, false};
    break;
  case ProtoKind::Model:
    if (field == 7)
      return ChildProto{ProtoKind::Graph, false};
    if (field == 25)
      return ChildProto{ProtoKind::Function, true};
    break;
  case ProtoKind::Graph:
    if (field == 1)
      return ChildProto{ProtoKind::Node, true};
    if (field == 5)
      return ChildProto{ProtoKind::Tensor, true};
    if (field == 15)
      return ChildProto{ProtoKind::SparseTensor, true};
    break;
  case ProtoKind::Node:
    if (field == 5)
      return ChildProto{ProtoKind::Attribute, true};
    break;
  case ProtoKind::Attribute:
    if (field == 5)
      return ChildProto{ProtoKind::Tensor, false};
    if (field == 6)
      return ChildProto{ProtoKind::Graph, false};
    if (field == 10)
      return ChildProto{ProtoKind::Tensor, true};
    if (field == 11)
      return ChildProto{ProtoKind::Graph, true};
    if (field == 22)
      return ChildProto{ProtoKind::SparseTensor, false};
    if (field == 23)
      return ChildProto{ProtoKind::SparseTensor, true};
    break;
  case ProtoKind::SparseTensor:
    if (field == 1 || field == 2)
      return ChildProto{ProtoKind::Tensor, false};
    break;
  case ProtoKind::Function:
    if (field == 7)
      return ChildProto{ProtoKind::Node, true};
    if (field == 11)
      return ChildProto{ProtoKind::Attribute, true};
    break;
  case ProtoKind::Tensor:
  case ProtoKind::Opaque:
    break;
  }
  return std::nullopt;
}

bool IsTensorPayloadField(uint32_t field) {
  return (field >= 4 && field <= 7) || field == 9 || field == 10 || field == 11;
}

bool IsPackedTensorField(uint32_t field) {
  return field == 4 || field == 5 || field == 7 || field == 10 || field == 11;
}

/** Adapts a buffered descriptor stream to omit large TensorProto payloads. */
class PayloadSkippingStream : public ol::utils::BinaryStream {
public:
  PayloadSkippingStream(ol::utils::BinaryStream &stream, uint64_t threshold,
                        ProtoKind kind = ProtoKind::Model,
                        const std::vector<SkippedTensorPayload> *retained_paths = nullptr)
      : stream_(stream), threshold_(threshold), retained_paths_(retained_paths) {
    ProtoFrame root;
    root.kind = kind;
    frames_.push_back(std::move(root));
  }

  uint64_t next_uint64() override {
    auto &frame = frames_.back();
    if (frame.kind != ProtoKind::Tensor || !IsTensorPayloadField(frame.field))
      return stream_.next_uint64();

    if (frame.wire_type == 0 && !frame.packed_values) {
      const auto before = stream_.tell();
      const auto value = stream_.next_uint64();
      RecordScalarBytes(frame, static_cast<uint64_t>(stream_.tell() - before));
      return value;
    }
    if (frame.packed_values || frame.wire_type != 2)
      return stream_.next_uint64();

    const auto before = stream_.tell();
    const uint64_t length = stream_.next_uint64();
    const uint64_t prefix_size =
        frame.field == 6 ? static_cast<uint64_t>(stream_.tell() - before) : 0;
    const uint64_t size = length + prefix_size;
    const bool retain_path = RetainPath(frame);
    const bool skip =
        !retain_path && (frame.field == 9 ? length >= threshold_
                                          : frame.payload_sizes[frame.field] + size >= threshold_);
    frame.payload_sizes[frame.field] += size;
    if (!skip) {
      if (IsPackedTensorField(frame.field))
        frame.packed_values = true;
      return length;
    }

    stream_.skip_bytes(length);
    frame.skipped_fields.insert(frame.field);
    return 0;
  }

  ol::utils::FieldNumber next_field() override {
    auto field = stream_.next_field();
    auto &frame = frames_.back();
    frame.field = static_cast<uint32_t>(field.field_number);
    frame.wire_type = static_cast<uint8_t>(field.wire_type);
    if (frame.kind == ProtoKind::Model && frame.field == 20)
      has_training_info_ = true;
    if (frame.kind == ProtoKind::Tensor && frame.skipped_fields.contains(frame.field)) {
      field.field_number = 0x1fffffff;
    }
    return field;
  }

  void CanRead(uint64_t len, const char *msg) override { stream_.CanRead(len, msg); }
  bool NotEnd() const override { return stream_.NotEnd(); }
  ol::utils::offset_t tell() const override { return stream_.tell(); }
  std::string tell_around() const override { return stream_.tell_around(); }

  const uint8_t *read_bytes(ol::utils::offset_t len, uint8_t *buffer = nullptr) override {
    const auto before = stream_.tell();
    const auto *result = stream_.read_bytes(len, buffer);
    auto &frame = frames_.back();
    if (frame.kind == ProtoKind::Tensor && !frame.packed_values &&
        IsTensorPayloadField(frame.field) && frame.wire_type != 2) {
      RecordScalarBytes(frame, static_cast<uint64_t>(stream_.tell() - before));
    }
    if (frame.kind == ProtoKind::Tensor && frame.packed_values)
      frame.packed_values = false;
    return result;
  }

  void skip_bytes(ol::utils::offset_t len) override { stream_.skip_bytes(len); }
  int64_t size() const override { return stream_.size(); }
  int64_t ByteCount() const override { return stream_.ByteCount(); }

  void LimitToNext(uint64_t len) override {
    auto &frame = frames_.back();
    const auto child = ChildMessage(frame.kind, frame.field);
    if (child && !child->repeated && frame.wire_type == 2 &&
        !frame.singular_children.insert(frame.field).second) {
      throw std::runtime_error("Duplicate singular protobuf message field.");
    }
    stream_.LimitToNext(len);
    limit_scopes_.push_back({});
    if (child && frame.wire_type == 2) {
      const size_t index = child->repeated ? frame.child_counts[frame.field]++ : 0;
      auto path = frame.path;
      path.push_back({frame.field, index});
      ProtoFrame child_frame;
      child_frame.kind = child->kind;
      child_frame.path = std::move(path);
      frames_.push_back(std::move(child_frame));
      limit_scopes_.back().child = true;
    } else if (frame.kind == ProtoKind::Tensor && frame.packed_values) {
      limit_scopes_.back().packed = true;
    } else if (frame.wire_type == 2) {
      ProtoFrame opaque_frame;
      opaque_frame.kind = ProtoKind::Opaque;
      frames_.push_back(std::move(opaque_frame));
      limit_scopes_.back().child = true;
    }
  }

  void Restore() override {
    const auto scope = limit_scopes_.back();
    if (scope.child) {
      auto &frame = frames_.back();
      if (frame.kind == ProtoKind::Tensor && !frame.skipped_fields.empty())
        skipped_.push_back({frame.path, std::move(frame.skipped_fields)});
      frames_.pop_back();
    }
    if (scope.packed && frames_.back().kind == ProtoKind::Tensor)
      frames_.back().packed_values = false;
    limit_scopes_.pop_back();
    stream_.Restore();
  }

  const std::vector<SkippedTensorPayload> &skipped() const { return skipped_; }
  bool has_training_info() const { return has_training_info_; }

private:
  void LimitTo(uint64_t) override {}

  bool RetainPath(const ProtoFrame &frame) const {
    if (!retained_paths_)
      return false;
    return std::any_of(retained_paths_->begin(), retained_paths_->end(), [&](const auto &payload) {
      if (payload.path.size() != frame.path.size())
        return false;
      for (size_t index = 0; index < frame.path.size(); ++index)
        if (payload.path[index].field != frame.path[index].field ||
            payload.path[index].index != frame.path[index].index)
          return false;
      return true;
    });
  }

  void RecordScalarBytes(ProtoFrame &frame, uint64_t size) {
    if (RetainPath(frame))
      return;
    frame.payload_sizes[frame.field] += size;
    if (frame.payload_sizes[frame.field] >= threshold_)
      frame.skipped_fields.insert(frame.field);
  }

  ol::utils::BinaryStream &stream_;
  uint64_t threshold_;
  std::vector<ProtoFrame> frames_;
  std::vector<LimitScope> limit_scopes_;
  std::vector<SkippedTensorPayload> skipped_;
  const std::vector<SkippedTensorPayload> *retained_paths_;
  bool has_training_info_ = false;
};

void ApplySkippedPayloads(ol::ModelProto &model, const std::vector<SkippedTensorPayload> &skipped) {
  for (const auto &payload : skipped) {
    auto *tensor = FindTensor(model, payload);
    if (!tensor)
      throw std::runtime_error("Could not locate an omitted tensor payload.");
    for (const auto field : payload.fields) {
      switch (field) {
      case 4:
        tensor->clear_float_data();
        break;
      case 5:
        tensor->clear_int32_data();
        break;
      case 6:
        tensor->clear_string_data();
        break;
      case 7:
        tensor->clear_int64_data();
        break;
      case 9:
        tensor->clear_raw_data();
        break;
      case 10:
        tensor->clear_double_data();
        break;
      case 11:
        tensor->clear_uint64_data();
        break;
      }
    }
    tensor->set_data_location(ol::TensorProto::EXTERNAL);
    const bool has_location = std::any_of(
        tensor->external_data().begin(), tensor->external_data().end(),
        [](const auto &entry) { return entry.key() == "location" && !entry.value().empty(); });
    if (!has_location) {
      auto *location = tensor->add_external_data();
      location->set_key("location");
      location->set_value(std::string(skippedPayloadLocation));
    }
  }
}

void CloseModelFile(int fd) {
#ifdef _WIN32
  _close(fd);
#else
  close(fd);
#endif
}

bool HasTrainingInfo(const uint8_t *bytes, size_t length) {
  ol::utils::StringStream stream(bytes, length);
  std::vector<uint32_t> groups;
  const auto skip = [&](uint64_t size) {
    if (size > length - static_cast<size_t>(stream.tell()))
      throw std::runtime_error("Truncated model field.");
    stream.skip_bytes(static_cast<ol::utils::offset_t>(size));
  };
  while (stream.NotEnd()) {
    const auto field = stream.next_field();
    if (groups.empty() && field.field_number == 20)
      return true;
    switch (static_cast<uint8_t>(field.wire_type)) {
    case 0:
      stream.next_uint64();
      break;
    case 1:
      skip(8);
      break;
    case 2: {
      const auto size = stream.next_uint64();
      skip(size);
      break;
    }
    case 3:
      if (groups.size() >= static_cast<size_t>(ol::ParseOptions{}.max_recursion_depth))
        throw std::runtime_error("Protobuf group nesting exceeds the model parser limit.");
      groups.push_back(field.field_number);
      break;
    case 4:
      if (groups.empty() || groups.back() != field.field_number)
        throw std::runtime_error("Invalid protobuf group in ONNX model.");
      groups.pop_back();
      break;
    case 5:
      skip(4);
      break;
    default:
      throw std::runtime_error("Invalid protobuf wire type in ONNX model.");
    }
  }
  if (!groups.empty())
    throw std::runtime_error("Unterminated protobuf group in ONNX model.");
  return false;
}

void ReadEncryptedModel(ol::ModelProto &model, const std::string &bytes,
                        const std::string &password, bool &has_training_info) {
#ifdef ONNX_LIGHT_HAS_OPENSSL
  // The combined native decrypt/parse API discards training_info before it can be checked.
  const std::string_view magic(bytes.data(), std::min<size_t>(8, bytes.size()));
  const bool legacy = magic == "ONNXCRY1";
  const EVP_CIPHER *cipher = legacy                ? EVP_aes_256_cbc()
                             : magic == "ONNXCRY2" ? EVP_chacha20_poly1305()
                             : magic == "ONNXCRY3" ? EVP_aes_256_gcm()
                                                   : nullptr;
  const size_t header_size = legacy ? 40 : 52;
  if (!cipher || bytes.size() < header_size ||
      password.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("Invalid encrypted ONNX model envelope.");
  const auto *input = reinterpret_cast<const uint8_t *>(bytes.data());
  std::array<uint8_t, 32> key;
  if (PKCS5_PBKDF2_HMAC(password.data(), static_cast<int>(password.size()), input + 8, 16, 100000,
                        EVP_sha256(), static_cast<int>(key.size()), key.data()) != 1)
    throw std::runtime_error("Could not derive the encrypted model key.");
  const auto context = std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)>(
      EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
  if (!context || EVP_DecryptInit_ex(context.get(), cipher, nullptr, key.data(), input + 24) != 1)
    throw std::runtime_error("Could not initialize encrypted model decryption.");
  int written = 0;
  if (magic == "ONNXCRY3" && EVP_DecryptUpdate(context.get(), nullptr, &written, input, 36) != 1)
    throw std::runtime_error("Could not authenticate the encrypted model header.");
  if (!legacy && EVP_CIPHER_CTX_ctrl(context.get(), EVP_CTRL_AEAD_SET_TAG, 16,
                                     const_cast<uint8_t *>(input + 36)) != 1)
    throw std::runtime_error("Could not set the encrypted model authentication tag.");
  std::string plaintext(bytes.size() - header_size + EVP_MAX_BLOCK_LENGTH, '\0');
  size_t output_size = 0;
  for (size_t offset = header_size; offset < bytes.size();) {
    const int size = static_cast<int>(std::min<size_t>(1024 * 1024, bytes.size() - offset));
    if (EVP_DecryptUpdate(context.get(),
                          reinterpret_cast<uint8_t *>(plaintext.data()) + output_size, &written,
                          input + offset, size) != 1)
      throw std::runtime_error("Could not decrypt the ONNX model.");
    output_size += static_cast<size_t>(written);
    offset += static_cast<size_t>(size);
  }
  if (EVP_DecryptFinal_ex(context.get(),
                          reinterpret_cast<uint8_t *>(plaintext.data()) + output_size,
                          &written) != 1)
    throw std::runtime_error("Could not decrypt the ONNX model: wrong password or damaged file.");
  output_size += static_cast<size_t>(written);
  try {
    has_training_info =
        HasTrainingInfo(reinterpret_cast<const uint8_t *>(plaintext.data()), output_size);
    ol::utils::StringStream stream(plaintext.data(), output_size);
    ReadModel(model, stream, ol::ParseOptions{});
  } catch (const std::exception &) {
    // Parser diagnostics can contain nearby plaintext bytes.
    throw std::runtime_error("Could not decode the decrypted ONNX model.");
  }
#else
  (void)model;
  (void)bytes;
  (void)password;
  (void)has_training_info;
  throw std::runtime_error("This native build does not support encrypted ONNX models.");
#endif
}

class LoadModelWorker : public Napi::AsyncWorker {
public:
  LoadModelWorker(Napi::Env env, std::string path, std::string file_name,
                  std::optional<std::string> password)
      : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)),
        path_(std::move(path)), model_(std::make_shared<StoredModel>()),
        password_(std::move(password)) {
    model_->file_name = std::move(file_name);
    model_->file_path = path_;
    model_->model = std::make_shared<ol::ModelProto>();
  }

  Napi::Promise Promise() const { return deferred_.Promise(); }

  void Execute() override {
    try {
      FileDescriptor file(OpenModelFile(path_));
      if (file.get() < 0)
        throw std::system_error(errno, std::generic_category(), "Could not open model file");
      ol::utils::FdReadStream source(file.get());
      if (password_) {
        const auto size = std::filesystem::file_size(ModelFilePath(path_));
        if (size > std::string{}.max_size())
          throw std::runtime_error("Encrypted ONNX model exceeds the maximum supported size.");
        std::string bytes(static_cast<size_t>(size), '\0');
        source.read_bytes(bytes.size(), reinterpret_cast<uint8_t *>(bytes.data()));
        ReadEncryptedModel(*model_->model, bytes, *password_, model_->has_training_info);
        return;
      }
      const auto path = ModelFilePath(path_);
      model_->source_size = std::filesystem::file_size(path);
      model_->source_time = std::filesystem::last_write_time(path);
      ol::ParseOptions options;
      options.raw_data_threshold = previewPayloadThreshold;
      PayloadSkippingStream stream(source, static_cast<uint64_t>(options.raw_data_threshold));
      ReadModel(*model_->model, stream, options);
      ApplySkippedPayloads(*model_->model, stream.skipped());
      model_->skipped = stream.skipped();
      model_->has_training_info = stream.has_training_info();
      if (std::filesystem::file_size(path) != model_->source_size ||
          std::filesystem::last_write_time(path) != model_->source_time)
        throw std::runtime_error("The source model changed while loading. Reopen it.");
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }

  void OnOK() override { deferred_.Resolve(ModelHandle(Env(), model_)); }
  void OnError(const Napi::Error &error) override { deferred_.Reject(error.Value()); }

private:
  Napi::Promise::Deferred deferred_;
  std::string path_;
  std::shared_ptr<StoredModel> model_;
  std::optional<std::string> password_;
};

} // namespace

ol::TensorProto *FindTensor(ol::ModelProto &model, const SkippedTensorPayload &payload) {
  const auto child_at = [](auto &values, size_t index) -> auto & {
    if (index >= values.size())
      throw std::runtime_error("Invalid tensor payload path.");
    return values[index];
  };
  ProtoKind kind = ProtoKind::Model;
  ol::GraphProto *graph = nullptr;
  ol::NodeProto *node = nullptr;
  ol::AttributeProto *attribute = nullptr;
  ol::SparseTensorProto *sparse = nullptr;
  ol::FunctionProto *function = nullptr;
  ol::TensorProto *tensor = nullptr;

  for (const auto &step : payload.path) {
    switch (kind) {
    case ProtoKind::Pipeline:
      return nullptr;
    case ProtoKind::Model:
      if (step.field == 7)
        graph = &model.ref_graph();
      else if (step.field == 25)
        function = &child_at(model.ref_functions(), step.index);
      break;
    case ProtoKind::Graph:
      if (step.field == 1)
        node = &child_at(graph->ref_node(), step.index);
      else if (step.field == 5)
        tensor = &child_at(graph->ref_initializer(), step.index);
      else if (step.field == 15)
        sparse = &child_at(graph->ref_sparse_initializer(), step.index);
      break;
    case ProtoKind::Node:
      attribute = &child_at(node->ref_attribute(), step.index);
      break;
    case ProtoKind::Attribute:
      if (step.field == 5)
        tensor = &attribute->ref_t();
      else if (step.field == 6)
        graph = &attribute->ref_g();
      else if (step.field == 10)
        tensor = &child_at(attribute->ref_tensors(), step.index);
      else if (step.field == 11)
        graph = &child_at(attribute->ref_graphs(), step.index);
      else if (step.field == 22)
        sparse = &attribute->ref_sparse_tensor();
      else if (step.field == 23)
        sparse = &child_at(attribute->ref_sparse_tensors(), step.index);
      break;
    case ProtoKind::SparseTensor:
      tensor = step.field == 1 ? &sparse->ref_values() : &sparse->ref_indices();
      break;
    case ProtoKind::Function:
      if (step.field == 7)
        node = &child_at(function->ref_node(), step.index);
      else
        attribute = &child_at(function->ref_attribute_proto(), step.index);
      break;
    case ProtoKind::Tensor:
      break;
    case ProtoKind::Opaque:
      return nullptr;
    }
    const auto child = ChildMessage(kind, step.field);
    if (!child)
      return nullptr;
    kind = child->kind;
  }
  return kind == ProtoKind::Tensor ? tensor : nullptr;
}

std::filesystem::path ModelFilePath(const std::string &path) {
  return std::filesystem::path(std::u8string(path.begin(), path.end()));
}

int OpenModelFile(const std::string &path) {
#ifdef _WIN32
  const auto unicode_path = ModelFilePath(path);
  return _wopen(unicode_path.c_str(), _O_RDONLY | _O_BINARY);
#else
  return open(path.c_str(), O_RDONLY);
#endif
}

FileDescriptor::~FileDescriptor() {
  if (fd_ >= 0)
    CloseModelFile(fd_);
}

void ReadModel(ol::ModelProto &model, ol::utils::BinaryStream &stream, ol::ParseOptions options) {
  if (!model.ParseFromStream(stream, options)) {
    throw std::runtime_error("Could not decode the ONNX model.");
  }
  if (!model.has_graph() || !model.has_ir_version() || model.ir_version() <= 0) {
    throw std::runtime_error("This file does not contain an ONNX model graph.");
  }
}

/** With raw data retained, the caller must keep bytes alive for borrowed fields. */
void ReadModel(ol::ModelProto &model, const Napi::Uint8Array &bytes, bool skip_raw_data) {
  ol::utils::StringStream stream(bytes.Data(), bytes.ByteLength());
  ol::ParseOptions options;
  options.skip_raw_data = skip_raw_data;
  options.raw_data_threshold = previewPayloadThreshold;
  options.no_copy = !skip_raw_data;
  ReadModel(model, stream, options);
}

std::optional<std::string> ReadPassword(const Napi::Value &value) {
  if (value.IsUndefined())
    return std::nullopt;
  if (!value.IsString() || value.As<Napi::String>().Utf8Value().empty())
    throw std::runtime_error("Expected a nonempty password.");
  return value.As<Napi::String>().Utf8Value();
}

namespace {
void ReadSourceModel(ol::ModelProto &model, ol::utils::BinaryStream &stream,
                     const StoredModel &source,
                     const std::vector<SkippedTensorPayload> *retained_paths) {
  const auto read_stage = [&](ol::utils::BinaryStream &stage) {
    if (retained_paths) {
      PayloadSkippingStream selected(stage, 0, ProtoKind::Model, retained_paths);
      ReadModel(model, selected, ol::ParseOptions{});
    } else {
      ReadModel(model, stage, ol::ParseOptions{});
    }
  };
  if (source.pipeline_stage.empty()) {
    read_stage(stream);
    return;
  }

  const uint64_t stage = source.pipeline_stage == "embedding" ? 7
                         : source.pipeline_stage == "prefill" ? 8
                         : source.pipeline_stage == "decode"  ? 9
                                                              : 0;
  if (!stage)
    throw std::runtime_error("Unknown source pipeline stage: " + source.pipeline_stage);
  bool found = false;
  std::vector<uint64_t> groups;
  const auto skip = [&](uint64_t size) {
    stream.CanRead(size, "Truncated source pipeline field.");
    stream.skip_bytes(static_cast<ol::utils::offset_t>(size));
  };
  while (stream.NotEnd()) {
    const auto field = stream.next_field();
    if (groups.empty() && field.field_number == stage && field.wire_type == 2) {
      const auto size = stream.next_uint64();
      stream.CanRead(size, "Truncated source pipeline stage.");
      stream.LimitToNext(size);
      read_stage(stream);
      stream.Restore();
      found = true;
      break;
    }
    switch (field.wire_type) {
    case 0:
      stream.next_uint64();
      break;
    case 1:
      skip(8);
      break;
    case 2:
      skip(stream.next_uint64());
      break;
    case 3:
      if (groups.size() >= static_cast<size_t>(ol::ParseOptions{}.max_recursion_depth))
        throw std::runtime_error("Protobuf group nesting exceeds the pipeline parser limit.");
      groups.push_back(field.field_number);
      break;
    case 4:
      if (groups.empty() || groups.back() != field.field_number)
        throw std::runtime_error("Invalid protobuf group in source pipeline.");
      groups.pop_back();
      break;
    case 5:
      skip(4);
      break;
    default:
      throw std::runtime_error("Invalid protobuf wire type in source pipeline.");
    }
  }
  if (!found)
    throw std::runtime_error("Could not reopen the source pipeline stage.");
}
} // namespace

void RestoreModelPayloads(ol::ModelProto &snapshot, const StoredModel &source,
                          const std::string &file_path) {
  if (source.has_training_info)
    throw std::runtime_error("Editing or saving models containing training_info is not supported "
                             "by onnx-light " ONNX_LIGHT_VERSION ".");
  if (source.skipped.empty())
    return;
  const auto path = ModelFilePath(file_path);
  const auto check_source = [&] {
    if (std::filesystem::file_size(path) != source.source_size ||
        std::filesystem::last_write_time(path) != source.source_time)
      throw std::runtime_error("The source model changed on disk. Reopen it before saving.");
  };
  check_source();
  FileDescriptor file(OpenModelFile(file_path));
  if (file.get() < 0)
    throw std::system_error(errno, std::generic_category(), "Could not reopen source model");
  ol::utils::FdReadStream stream(file.get());
  ol::ModelProto original;
  ReadSourceModel(original, stream, source, nullptr);
  check_source();
  for (const auto &payload : source.skipped) {
    auto *target = FindTensor(snapshot, payload);
    const auto *tensor = FindTensor(original, payload);
    if (!target || !tensor || target->name() != tensor->name() ||
        target->data_type() != tensor->data_type() ||
        target->dims().size() != tensor->dims().size() ||
        !std::equal(target->dims().begin(), target->dims().end(), tensor->dims().begin()))
      throw std::runtime_error("Could not restore an omitted tensor payload.");
    *target = tensor->SharedCopy();
  }
}

void RestoreStatisticsPayloads(ol::ModelProto &snapshot, const StoredModel &source,
                               const std::string &file_path,
                               const std::vector<SkippedTensorPayload> &payloads) {
  struct Target {
    SkippedTensorPayload payload;
    std::string name;
    bool has_name;
    int data_type;
    std::vector<int64_t> dims;
  };
  std::vector<Target> targets;
  for (const auto &requested : payloads) {
    const auto found =
        std::find_if(source.skipped.begin(), source.skipped.end(),
                     [&](const auto &payload) { return payload.path == requested.path; });
    if (found == source.skipped.end())
      continue;
    if (std::none_of(targets.begin(), targets.end(),
                     [&](const auto &target) { return target.payload.path == found->path; })) {
      const auto *tensor = FindTensor(snapshot, *found);
      if (!tensor)
        throw std::runtime_error("Could not locate an omitted tensor payload.");
      targets.push_back({*found,
                         tensor->str_name(),
                         tensor->has_name(),
                         tensor->data_type(),
                         {tensor->dims().begin(), tensor->dims().end()}});
    }
  }
  if (targets.empty())
    return;

  const auto path = ModelFilePath(file_path);
  const auto check_source = [&] {
    if (std::filesystem::file_size(path) != source.source_size ||
        std::filesystem::last_write_time(path) != source.source_time)
      throw std::runtime_error("The source model changed on disk. Reopen it before saving.");
  };
  check_source();
  FileDescriptor file(OpenModelFile(file_path));
  if (file.get() < 0)
    throw std::system_error(errno, std::generic_category(), "Could not reopen source model");
  ol::utils::FdReadStream stream(file.get());
  ol::ModelProto original;
  std::vector<SkippedTensorPayload> retained_paths;
  retained_paths.reserve(targets.size());
  for (const auto &target : targets)
    retained_paths.push_back(target.payload);
  ReadSourceModel(original, stream, source, &retained_paths);
  check_source();

  for (const auto &target : targets) {
    auto *destination = FindTensor(snapshot, target.payload);
    auto *tensor = FindTensor(original, target.payload);
    if (!destination || !tensor)
      throw std::runtime_error("Could not locate an omitted tensor payload.");
    if (destination->str_name() != target.name || destination->has_name() != target.has_name ||
        tensor->str_name() != target.name || tensor->has_name() != target.has_name ||
        tensor->data_type() != target.data_type || tensor->dims().size() != target.dims.size() ||
        !std::equal(tensor->dims().begin(), tensor->dims().end(), target.dims.begin()))
      throw std::runtime_error("Could not restore an omitted tensor payload.");
    *destination = tensor->SharedCopy();
  }
}

std::shared_ptr<StoredPipeline> ReadPipeline(ol::utils::BinaryStream &source,
                                             const std::string &file_name,
                                             const std::string &file_path) {
  auto pipeline = std::make_shared<StoredPipeline>();
  const bool local = !file_path.empty();
  const auto path = ModelFilePath(file_path);
  const auto size = local ? std::filesystem::file_size(path) : 0;
  const auto time =
      local ? std::filesystem::last_write_time(path) : std::filesystem::file_time_type{};
  ol::ParseOptions options;
  PayloadSkippingStream stream(
      source, local ? previewPayloadThreshold : std::numeric_limits<uint64_t>::max(),
      ProtoKind::Pipeline);
  if (!pipeline->proto.ParseFromStream(stream, options) || !pipeline->proto.has_prefill())
    throw std::runtime_error("This file does not contain a PipelineProto with a prefill model.");
  for (const auto &[role, field, present] :
       {std::tuple{"embedding", 7u, pipeline->proto.has_embedding()},
        std::tuple{"prefill", 8u, pipeline->proto.has_prefill()},
        std::tuple{"decode", 9u, pipeline->proto.has_decode()}}) {
    if (!present)
      continue;
    auto stored = std::make_shared<StoredModel>();
    auto &model = PipelineStage(pipeline->proto, role);
    if (!model.has_graph() || !model.has_ir_version() || model.ir_version() <= 0)
      throw std::runtime_error(std::string("Invalid pipeline model: ") + role);
    stored->model = std::make_shared<ol::ModelProto>(std::move(model));
    stored->file_name = file_name + ": " + role;
    stored->file_path = file_path;
    stored->pipeline_stage = role;
    stored->source_size = size;
    stored->source_time = time;
    // An unsupported training_info anywhere makes every pipeline export unsafe.
    stored->has_training_info = stream.has_training_info();
    for (const auto &payload : stream.skipped())
      if (!payload.path.empty() && payload.path.front().field == field) {
        auto relative = payload;
        relative.path.erase(relative.path.begin());
        stored->skipped.push_back(std::move(relative));
      }
    ApplySkippedPayloads(*stored->model, stored->skipped);
    pipeline->models.emplace(role, std::move(stored));
  }
  pipeline->proto.clear_embedding();
  pipeline->proto.clear_prefill();
  pipeline->proto.clear_decode();
  if (local &&
      (std::filesystem::file_size(path) != size || std::filesystem::last_write_time(path) != time))
    throw std::runtime_error("The source pipeline changed while loading. Reopen it.");
  return pipeline;
}

Napi::Value DecodeGraph(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if ((info.Length() != 2 && info.Length() != 3) || !info[0].IsTypedArray() ||
      info[0].As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array || !info[1].IsString() ||
      (info.Length() == 3 && !info[2].IsBoolean())) {
    Napi::TypeError::New(env, "Expected a Uint8Array, a file name and an optional inference flag.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  // C++ exceptions must not unwind through the Node-API boundary.
  try {
    const bool infer_shapes = info.Length() == 3 && info[2].As<Napi::Boolean>().Value();
    ol::ModelProto model;
    ReadModel(model, info[0].As<Napi::Uint8Array>(), !infer_shapes);
    if (infer_shapes)
      InferShapes(model);
    return ModelGraph(env, model, info[1].As<Napi::String>().Utf8Value());
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value LoadModel(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if ((info.Length() != 2 && info.Length() != 3) || !info[0].IsTypedArray() ||
      info[0].As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array || !info[1].IsString()) {
    Napi::TypeError::New(env, "Expected a Uint8Array and a file name.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  try {
    auto model = std::make_shared<StoredModel>();
    model->source = Napi::Persistent(info[0].As<Napi::Uint8Array>());
    model->file_name = info[1].As<Napi::String>().Utf8Value();
    model->model = std::make_shared<ol::ModelProto>();
    const auto password = ReadPassword(info[2]);
    if (password) {
      const auto bytes = model->source.Value();
      ReadEncryptedModel(
          *model->model,
          std::string(reinterpret_cast<const char *>(bytes.Data()), bytes.ByteLength()), *password,
          model->has_training_info);
    } else {
      const auto bytes = model->source.Value();
      model->has_training_info = HasTrainingInfo(bytes.Data(), bytes.ByteLength());
      ReadModel(*model->model, model->source.Value(), false);
    }
    return ModelHandle(env, model);
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value LoadModelFile(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if ((info.Length() != 2 && info.Length() != 3) || !info[0].IsString() || !info[1].IsString()) {
    Napi::TypeError::New(env, "Expected a file path and a file name.").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  try {
    const auto path = info[0].As<Napi::String>().Utf8Value();
    if (path.find('\0') != std::string::npos)
      throw std::runtime_error("Invalid model file path.");
    auto worker = std::make_unique<LoadModelWorker>(
        env, path, info[1].As<Napi::String>().Utf8Value(), ReadPassword(info[2]));
    const auto promise = worker->Promise();
    worker->Queue();
    worker.release();
    return promise;
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value ListBackendTests(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  try {
    const auto cases = ol::core::backend_test::CollectTestCases("", true);
    auto result = Napi::Array::New(env);
    uint32_t index = 0;
    for (const auto &test : cases) {
      auto item = Napi::Object::New(env);
      item.Set("id", index);
      item.Set("name", test.name);
      item.Set("modelName", test.model_name);
      item.Set("kind", Text(env, ol::core::backend_test::TestCaseKindName(test.kind)));
      item.Set("tag", Text(env, ol::core::backend_test::TestCaseTagName(test.tag)));
      result.Set(index++, item);
    }
    return result;
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value LoadBackendTest(const Napi::CallbackInfo &info) {
  const auto env = info.Env();
  if (info.Length() != 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected a backend test ID.").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  const auto id = info[0].As<Napi::Number>().DoubleValue();
  if (!std::isfinite(id) || id < 0 || std::floor(id) != id) {
    Napi::RangeError::New(env, "Invalid onnx-light backend test ID.").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  try {
    auto cases = ol::core::backend_test::CollectTestCases("", true);
    if (id >= cases.size()) {
      throw std::runtime_error("Unknown onnx-light backend test ID: " + std::to_string(id));
    }
    auto model = std::make_shared<StoredModel>();
    model->backend_case = std::make_shared<ol::core::backend_test::TestCase>(
        std::move(cases[static_cast<size_t>(id)]));
    model->backend_case->set_expected_outputs_generated(false);
    model->model = model->backend_case->model_handle();
    model->file_name = model->backend_case->name;
    return ModelHandle(env, model);
  } catch (const std::exception &error) {
    Napi::Error::New(env, error.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

} // namespace onnx_light_viewer
