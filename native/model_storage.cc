/** @file External payload materialization and model tensor traversal. */
#include "model_storage.h"
#include "model_io.h"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace onnx_light_viewer {
namespace {

template <typename Proto>
void LoadExternalTensorImpl(Proto &tensor, const std::filesystem::path &directory) {
  const auto location = ExternalDataLocation(tensor);
  int64_t offset = 0;
  int64_t length = -1;
  const auto number = [](std::string_view text) {
    int64_t value;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
      throw std::runtime_error("Invalid external weight offset or length.");
    return value;
  };
  for (const auto &entry : tensor.external_data()) {
    if (entry.key() == "offset")
      offset = number(std::string_view(entry.value()));
    else if (entry.key() == "length" || entry.key() == "size")
      length = number(std::string_view(entry.value()));
  }
  const auto relative = ModelFilePath(location);
  const auto normalized = relative.lexically_normal();
  size_t components = 0;
  bool unsafe = location.empty() || location.find('\0') != std::string::npos ||
                normalized.empty() || normalized.has_root_path() || normalized.filename().empty() ||
                normalized == std::filesystem::path(".");
  for (const auto &part : relative)
    unsafe |= part == std::filesystem::path("..");
  for (const auto &part : normalized)
    if (part != std::filesystem::path("."))
      unsafe |= ++components > 2;
  if (unsafe)
    throw std::runtime_error("External weight location must be a safe relative path.");
  const auto base_directory = directory.empty() ? std::filesystem::current_path() : directory;
  const auto path = base_directory / normalized;
  if (std::filesystem::is_symlink(path))
    throw std::runtime_error("External weight files must not be symbolic links.");
  const auto base = std::filesystem::weakly_canonical(base_directory);
  const auto target = std::filesystem::weakly_canonical(path);
  auto prefix = base.native();
  if (!prefix.empty() && prefix.back() != std::filesystem::path::preferred_separator)
    prefix += std::filesystem::path::preferred_separator;
  if (target != base && target.native().find(prefix) != 0)
    throw std::runtime_error("External weight file resolves outside the model directory.");
  if (std::filesystem::hard_link_count(path) > 1)
    throw std::runtime_error("External weight files must not have multiple hard links.");
  const auto utf8_path = path.u8string();
  FileDescriptor file(OpenModelFile(std::string(utf8_path.begin(), utf8_path.end())));
  if (file.get() < 0)
    throw std::system_error(errno, std::generic_category(), "Could not open external weights");
  ol::utils::FdReadStream stream(file.get());
  const auto file_size = std::filesystem::file_size(path);
  offset = std::max<int64_t>(0, offset);
  if (static_cast<uint64_t>(offset) > file_size)
    throw std::runtime_error("External weight offset is past the end of the file.");
  const auto available = file_size - static_cast<uint64_t>(offset);
  if (available > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    throw std::runtime_error("External weight length exceeds the available data.");
  if (length < 0)
    length = static_cast<int64_t>(available);
  if (static_cast<uint64_t>(length) > available ||
      static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max())
    throw std::runtime_error("External weight length exceeds the available data.");
  stream.skip_bytes(offset);
  tensor.ref_raw_data().resize(static_cast<size_t>(length));
  if (length)
    stream.read_bytes(length, tensor.ref_raw_data().data());
  tensor.set_data_location(ol::TensorProto::DEFAULT);
  tensor.clear_external_data();
}

} // namespace

void LoadExternalTensor(ol::TensorProto &tensor, const std::filesystem::path &directory) {
  LoadExternalTensorImpl(tensor, directory);
}

void LoadExternalTensor(ol::EncodedValueProto &tensor, const std::filesystem::path &directory) {
  LoadExternalTensorImpl(tensor, directory);
}

} // namespace onnx_light_viewer
