#include "addon.h"
#include <algorithm>
#include <codecvt>
#include <cstdio>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <locale>
#include <map>
#include <memory>
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES
#include <miniz.h>
#include <model_package.h>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <locale.h>
#endif

namespace onnx_light_viewer {
namespace {
namespace fs = std::filesystem;
constexpr size_t MaxEntries = 10000;
constexpr uint64_t MaxSize = uint64_t{8} * 1024 * 1024 * 1024;
using Bytes = std::vector<uint8_t>;

fs::path FromUtf8(const std::string &value) {
  return fs::path(std::u8string(value.begin(), value.end()));
}

std::string Utf8(const fs::path &path) {
  const auto value = path.u8string();
  return {reinterpret_cast<const char *>(value.data()), value.size()};
}

std::string Lower(std::string value) {
  for (auto &c : value)
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c + ('a' - 'A'));
  return value;
}

std::string PathKey(const std::string &value) {
#ifdef _WIN32
  const auto wide = FromUtf8(value).wstring();
  std::wstring folded(wide.size(), L'\0');
  if (!LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE, wide.data(),
                     static_cast<int>(wide.size()), folded.data(), static_cast<int>(folded.size()),
                     nullptr, nullptr, 0))
    throw std::runtime_error("Cannot case-fold ZIP filename");
  return Utf8(fs::path(folded));
#else
#ifdef __APPLE__
  static const auto locale = newlocale(LC_CTYPE_MASK, "en_US.UTF-8", nullptr);
#else
  static const auto locale = newlocale(LC_CTYPE_MASK, "C.UTF-8", nullptr);
#endif
  if (!locale)
    throw std::runtime_error("A UTF-8 locale is required for portable ZIP filenames");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  auto wide = converter.from_bytes(value);
  for (auto &character : wide)
    character = static_cast<wchar_t>(towlower_l(character, locale));
  return converter.to_bytes(wide);
#endif
}

bool IsModel(const std::string &name) {
  const auto lower = Lower(name);
  return lower.ends_with(".onnx") || lower.ends_with(".onnx.enc") || lower.ends_with(".pipe-onnx");
}

struct Names {
  std::set<std::string> explicit_paths;
  std::map<std::string, std::pair<std::string, bool>> paths;

  void Add(const std::string &name, bool directory) {
    auto path = directory ? name.substr(0, name.size() - 1) : name;
    if (path.empty() || name.find_first_of("\\:\0<>\"|?*", 0, 9) != std::string::npos ||
        std::any_of(name.begin(), name.end(),
                    [](unsigned char character) { return character < 32; }))
      throw std::runtime_error("Unsafe ZIP entry name: " + name);
    // JSON's strict UTF-8 encoder rejects invalid byte sequences without locale-dependent folding.
    try {
      (void)nlohmann::json(name).dump();
    } catch (const nlohmann::json::exception &) {
      throw std::runtime_error("Unsafe ZIP entry name: invalid UTF-8");
    }
    if (name.size() >= MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE)
      throw std::runtime_error("Unsafe ZIP entry name: path too long");
    if (!explicit_paths.insert(PathKey(path)).second)
      throw std::runtime_error("Duplicate ZIP entry: " + name);
    size_t start = 0;
    while (start <= path.size()) {
      const auto end = path.find('/', start);
      const auto part = path.substr(start, end == std::string::npos ? end : end - start);
      const auto base = Lower(part.substr(0, part.find('.')));
      const auto suffix = base.size() >= 3 ? base.substr(3) : std::string();
      const bool device_number = (suffix.size() == 1 && suffix[0] >= '1' && suffix[0] <= '9') ||
                                 suffix == "\xc2\xb9" || suffix == "\xc2\xb2" ||
                                 suffix == "\xc2\xb3";
      if (part.empty() || part == "." || part == ".." || part.back() == '.' || part.back() == ' ' ||
          part.size() > 255 || base == "con" || base == "prn" || base == "aux" || base == "nul" ||
          base == "conin$" || base == "conout$" ||
          ((base.starts_with("com") || base.starts_with("lpt")) && device_number))
        throw std::runtime_error("Unsafe ZIP entry name: " + name);
      const auto current = path.substr(0, end);
      const auto key = PathKey(current);
      const bool is_directory = end != std::string::npos || directory;
      const auto previous = paths.find(key);
      if (previous != paths.end() &&
          (previous->second.first != current || !previous->second.second || !is_directory))
        throw std::runtime_error("Conflicting ZIP entry: " + name);
      paths[key] = {current, is_directory};
      if (end == std::string::npos)
        break;
      start = end + 1;
    }
  }
};

struct Zip {
  mz_zip_archive value{};
  ~Zip() { mz_zip_end(&value); }
  void Check(bool success, const std::string &message) {
    if (!success)
      throw std::runtime_error(message + ": " +
                               mz_zip_get_error_string(mz_zip_get_last_error(&value)));
  }
};

void CheckStatus(ModelPackageStatus *status) {
  if (!status)
    return;
  const std::string message = ModelPackageStatus_Message(status);
  ModelPackageStatus_Release(status);
  throw std::runtime_error("Invalid model package: " + message);
}

std::vector<std::string> DeclaredModels(const fs::path &root) {
  // Installed layouts intentionally allow external paths in the upstream API,
  // even when allow_external_paths is false. ZIP imports must remain portable.
  std::ifstream manifest(root / "manifest.json", std::ios::binary);
  auto guard = nlohmann::json::parse(manifest, nullptr, false);
  if (guard.is_object() && guard.value("layout", nlohmann::json()) == "installed")
    throw std::runtime_error("ZIP model packages must use portable layout, not installed");

  ::ModelPackage *raw = nullptr;
  const ModelPackageOpenOptions options{false, false, true};
  CheckStatus(ModelPackage_Open(Utf8(root).c_str(), &options, &raw));
  std::unique_ptr<::ModelPackage, decltype(&ModelPackage_Close)> package(raw, ModelPackage_Close);
  const auto *info = ModelPackage_Info(package.get());
  std::vector<std::string> result;
  for (size_t c = 0; c < info->num_components; ++c) {
    const auto &component = info->components[c];
    for (size_t v = 0; v < component.num_variants; ++v) {
      const auto &variant = component.variants[v];
      const auto *executor = ModelVariantInfo_FindExecutorInfo(&variant, "ort");
      if (!executor)
        continue;
      const auto payload = nlohmann::json::parse(executor->json, nullptr, false);
      if (!payload.is_object() || !payload.contains("model_file") ||
          !payload["model_file"].is_string())
        throw std::runtime_error("Invalid executor_info.ort.model_file for " +
                                 std::string(component.name) + "/" + variant.name);
      if (payload.contains("session_options")) {
        const auto &session_options = payload["session_options"];
        if (!session_options.is_object())
          throw std::runtime_error("Invalid executor_info.ort.session_options");
        if (session_options.contains("session.model_external_initializers_file_folder_path"))
          throw std::runtime_error(
              "Unsupported model package session option: "
              "session.model_external_initializers_file_folder_path. "
              "The viewer currently loads external weights relative to the model file.");
      }
      const auto reference = payload["model_file"].get<std::string>();
      if (reference.empty() || reference.find('\0') != std::string::npos)
        throw std::runtime_error("Invalid executor_info.ort.model_file");
      const char *resolved = nullptr;
      CheckStatus(ModelPackage_ResolveStringRef(package.get(), variant.variant_directory,
                                                reference.c_str(), true, &resolved));
      const auto path = FromUtf8(resolved);
      if (!fs::is_regular_file(path) || !IsModel(Utf8(path)))
        throw std::runtime_error("Declared model_file is not an ONNX model: " + reference);
      result.emplace_back(resolved);
    }
  }
  if (result.empty())
    throw std::runtime_error(
        "Model package declares no ONNX models in executor_info.ort.model_file");
  return result;
}

struct Entry {
  std::string name;
  mz_zip_archive_file_stat stat{};
  bool directory = false;
};

void ValidateEmptyHeader(const Bytes &bytes, const Entry &entry) {
  // miniz intentionally skips local-header validation for zero-length entries.
  if (entry.stat.m_uncomp_size != 0)
    return;
  const auto offset = entry.stat.m_local_header_ofs;
  if (offset > bytes.size() || bytes.size() - offset < 30)
    throw std::runtime_error("Invalid ZIP empty entry header");
  const auto *header = bytes.data() + static_cast<size_t>(offset);
  const auto u16 = [](const uint8_t *p) { return uint32_t{p[0]} | (uint32_t{p[1]} << 8); };
  const auto u32 = [&](const uint8_t *p) { return u16(p) | (u16(p + 2) << 16); };
  const auto u64 = [&](const uint8_t *p) {
    return uint64_t{u32(p)} | (uint64_t{u32(p + 4)} << 32);
  };
  const auto name_size = u16(header + 26);
  const auto extra_size = u16(header + 28);
  const auto remaining = bytes.size() - offset - 30;
  if (u32(header) != 0x04034b50 || u16(header + 6) != entry.stat.m_bit_flag ||
      u16(header + 8) != entry.stat.m_method || name_size != entry.name.size() ||
      remaining < name_size + extra_size ||
      entry.stat.m_comp_size > remaining - name_size - extra_size ||
      std::memcmp(header + 30, entry.name.data(), name_size) != 0 ||
      (!(entry.stat.m_bit_flag & 8) && u32(header + 14) != entry.stat.m_crc32))
    throw std::runtime_error("Invalid ZIP empty entry header");
  if (entry.stat.m_crc32 != 0)
    throw std::runtime_error("Corrupt ZIP entry: " + entry.name);
  if (!(entry.stat.m_bit_flag & 8)) {
    uint64_t compressed_size = u32(header + 18);
    uint64_t uncompressed_size = u32(header + 22);
    if (compressed_size == UINT32_MAX || uncompressed_size == UINT32_MAX) {
      const auto *extra = header + 30 + name_size;
      const auto *end = extra + extra_size;
      bool found = false;
      while (end - extra >= 4) {
        const auto id = u16(extra);
        const auto size = u16(extra + 2);
        extra += 4;
        if (static_cast<size_t>(end - extra) < size)
          break;
        if (id == 1) {
          const auto required = (uncompressed_size == UINT32_MAX ? 8u : 0u) +
                                (compressed_size == UINT32_MAX ? 8u : 0u);
          if (size < required)
            break;
          if (uncompressed_size == UINT32_MAX) {
            uncompressed_size = u64(extra);
            extra += 8;
          }
          if (compressed_size == UINT32_MAX)
            compressed_size = u64(extra);
          found = true;
          break;
        }
        extra += size;
      }
      if (!found)
        throw std::runtime_error("Invalid ZIP64 empty entry sizes");
    }
    if (compressed_size != entry.stat.m_comp_size || uncompressed_size != 0)
      throw std::runtime_error("Invalid ZIP empty entry sizes");
  }
  if (entry.stat.m_method == 8 && !entry.stat.m_comp_size)
    throw std::runtime_error("Corrupt ZIP entry: " + entry.name);
  if (entry.directory && entry.stat.m_comp_size) {
    uint8_t scratch = 0;
    const auto *compressed = header + 30 + name_size + extra_size;
    if (entry.stat.m_method != 8 ||
        tinfl_decompress_mem_to_mem(&scratch, 1, compressed,
                                    static_cast<size_t>(entry.stat.m_comp_size), 0) != 0)
      throw std::runtime_error("Corrupt ZIP entry: " + entry.name);
  }
}

std::vector<std::string> Extract(const Bytes &bytes, const fs::path &directory) {
  if (!fs::is_directory(fs::symlink_status(directory)) || !fs::is_empty(directory))
    throw std::runtime_error("ZIP extraction requires an empty private directory");
  Zip zip;
  zip.Check(mz_zip_reader_init_mem(&zip.value, bytes.data(), bytes.size(), 0),
            "Invalid ZIP archive");
  const auto count = mz_zip_reader_get_num_files(&zip.value);
  if (count > MaxEntries)
    throw std::runtime_error("Invalid ZIP directory or too many ZIP entries");
  std::vector<Entry> entries;
  Names names;
  uint64_t total = 0;
  std::vector<std::string> models;
  std::vector<fs::path> manifests;
  for (mz_uint i = 0; i < count; ++i) {
    Entry entry;
    zip.Check(mz_zip_reader_file_stat(&zip.value, i, &entry.stat), "Invalid ZIP directory");
    const auto length = mz_zip_reader_get_filename(&zip.value, i, nullptr, 0);
    if (!length || length > 4097)
      throw std::runtime_error("Unsafe ZIP entry name: path too long");
    std::vector<char> name(length);
    zip.Check(mz_zip_reader_get_filename(&zip.value, i, name.data(), length) == length,
              "Invalid ZIP filename");
    entry.name.assign(name.data(), length - 1);
    entry.directory = entry.name.ends_with("/");
    names.Add(entry.name, entry.directory);
    const auto mode = (entry.stat.m_external_attr >> 16) & 0170000;
    if (mode && mode != (entry.directory ? 0040000 : 0100000))
      throw std::runtime_error("Unsupported ZIP entry type: " + entry.name);
    if (mz_zip_reader_is_file_encrypted(&zip.value, i))
      throw std::runtime_error("Encrypted ZIP entries are unsupported");
    if (entry.stat.m_method != 0 && entry.stat.m_method != 8)
      throw std::runtime_error("ZIP entry compression is unsupported");
    if (!entry.stat.m_is_supported || entry.stat.m_is_directory != entry.directory)
      throw std::runtime_error("Unsupported ZIP entry type or compression: " + entry.name);
    if (entry.stat.m_uncomp_size > MaxSize - total ||
        (entry.directory && entry.stat.m_uncomp_size != 0) ||
        (entry.stat.m_method == 0 && entry.stat.m_comp_size != entry.stat.m_uncomp_size))
      throw std::runtime_error("ZIP entry size or package limit is invalid");
    total += entry.stat.m_uncomp_size;
    ValidateEmptyHeader(bytes, entry);
    zip.Check(mz_zip_validate_file(&zip.value, i, MZ_ZIP_FLAG_VALIDATE_HEADERS_ONLY),
              "Invalid ZIP entry headers");
    if (!entry.directory && IsModel(entry.name))
      models.push_back(Utf8(directory / FromUtf8(entry.name)));
    if (!entry.directory && FromUtf8(entry.name).filename() == "manifest.json")
      manifests.push_back(FromUtf8(entry.name).parent_path());
    entries.push_back(std::move(entry));
  }
  for (mz_uint i = 0; i < count; ++i) {
    const auto &entry = entries[i];
    const auto target = directory / FromUtf8(entry.name);
    fs::create_directories(entry.directory ? target : target.parent_path());
    if (entry.directory)
      continue;
    fs::permissions(target.parent_path(), fs::perms::owner_all, fs::perm_options::replace);
#ifdef _WIN32
    FILE *file = _wfopen(target.c_str(), L"wbx");
#else
    FILE *file = std::fopen(target.c_str(), "wbx");
#endif
    if (!file)
      throw std::runtime_error("Cannot create ZIP entry: " + entry.name);
    const auto close = [](FILE *value) { std::fclose(value); };
    std::unique_ptr<FILE, decltype(close)> output(file, close);
    fs::permissions(target, fs::perms::owner_read | fs::perms::owner_write,
                    fs::perm_options::replace);
    zip.Check(mz_zip_reader_extract_to_cfile(&zip.value, i, output.get(), 0),
              "Corrupt ZIP entry: " + entry.name);
    if (std::fflush(output.get()) != 0)
      throw std::runtime_error("Cannot write ZIP entry: " + entry.name);
  }
  if (!manifests.empty()) {
    // Only a package root (possibly beneath wrapper directories) is eligible.
    std::sort(manifests.begin(), manifests.end(), [](const auto &a, const auto &b) {
      return std::distance(a.begin(), a.end()) < std::distance(b.begin(), b.end());
    });
    const auto root = manifests.front();
    for (const auto &entry : entries) {
      if (entry.directory) {
        const auto ancestor = root.lexically_relative(FromUtf8(entry.name));
        if (!ancestor.empty() && *ancestor.begin() != "..")
          continue;
      }
      auto relative = FromUtf8(entry.name).lexically_relative(root);
      if (relative.empty() || *relative.begin() == "..")
        throw std::runtime_error("Ambiguous model package root");
    }
    models = DeclaredModels(directory / root);
  }
  if (models.empty())
    throw std::runtime_error("ZIP package contains no ONNX models");
  std::sort(models.begin(), models.end());
  models.erase(std::unique(models.begin(), models.end()), models.end());
  return models;
}

struct OutputEntry {
  std::string name;
  Bytes bytes;
};

Bytes Create(const std::vector<OutputEntry> &entries) {
  if (entries.empty() || !IsModel(entries.front().name))
    throw std::runtime_error("The model filename must end in .onnx, .onnx.enc or .pipe-onnx");
  if (entries.size() > MaxEntries)
    throw std::runtime_error("ZIP package limit exceeded: too many ZIP entries");
  Names names;
  uint64_t total = 0;
  for (const auto &entry : entries) {
    names.Add(entry.name, false);
    if (entry.bytes.size() > MaxSize - total)
      throw std::runtime_error("ZIP package limit exceeded");
    total += entry.bytes.size();
  }
  Zip zip;
  zip.Check(mz_zip_writer_init_heap(&zip.value, 0, 0), "Cannot create ZIP");
  for (const auto &entry : entries)
    zip.Check(mz_zip_writer_add_mem(&zip.value, entry.name.c_str(), entry.bytes.data(),
                                    entry.bytes.size(), MZ_DEFAULT_COMPRESSION),
              "Cannot create ZIP entry: " + entry.name);
  void *data = nullptr;
  size_t size = 0;
  zip.Check(mz_zip_writer_finalize_heap_archive(&zip.value, &data, &size), "Cannot finish ZIP");
  std::unique_ptr<void, decltype(&mz_free)> allocation(data, mz_free);
  const auto *begin = static_cast<const uint8_t *>(data);
  return {begin, begin + size};
}

Bytes Snapshot(const Napi::Value &value) {
  if (!value.IsTypedArray() || value.As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array)
    throw std::runtime_error("ZIP data must be a Uint8Array");
  auto bytes = value.As<Napi::Uint8Array>();
  if (!bytes.ElementLength())
    return {};
  return {bytes.Data(), bytes.Data() + bytes.ElementLength()};
}

class PackageWorker : public Napi::AsyncWorker {
public:
  PackageWorker(Napi::Env env, Bytes bytes, fs::path directory)
      : Napi::AsyncWorker(env), deferred(Napi::Promise::Deferred::New(env)),
        input(std::move(bytes)), directory(std::move(directory)), extracting(true) {}
  PackageWorker(Napi::Env env, std::vector<OutputEntry> entries)
      : Napi::AsyncWorker(env), deferred(Napi::Promise::Deferred::New(env)),
        entries(std::move(entries)) {}
  Napi::Promise Promise() { return deferred.Promise(); }

private:
  Napi::Promise::Deferred deferred;
  Bytes input;
  fs::path directory;
  bool extracting = false;
  std::vector<OutputEntry> entries;
  Bytes output;
  std::vector<std::string> models;
  void Execute() override {
    try {
      if (extracting)
        models = Extract(input, directory);
      else
        output = Create(entries);
    } catch (const std::exception &error) {
      SetError(error.what());
    }
  }
  void OnOK() override {
    if (extracting) {
      auto result = Napi::Array::New(Env(), models.size());
      for (size_t i = 0; i < models.size(); ++i)
        result.Set(i, Napi::String::New(Env(), models[i]));
      deferred.Resolve(result);
    } else {
      deferred.Resolve(Napi::Buffer<uint8_t>::Copy(Env(), output.data(), output.size()));
    }
  }
  void OnError(const Napi::Error &error) override { deferred.Reject(error.Value()); }
};
} // namespace

Napi::Value ExtractModelPackage(const Napi::CallbackInfo &info) {
  try {
    if (info.Length() != 2 || !info[1].IsString())
      throw std::runtime_error("Expected ZIP bytes and a private extraction directory");
    auto *worker = new PackageWorker(info.Env(), Snapshot(info[0]),
                                     FromUtf8(info[1].As<Napi::String>().Utf8Value()));
    auto promise = worker->Promise();
    worker->Queue();
    return promise;
  } catch (const std::exception &error) {
    Napi::Error::New(info.Env(), error.what()).ThrowAsJavaScriptException();
    return info.Env().Undefined();
  }
}

Napi::Value CreateModelPackage(const Napi::CallbackInfo &info) {
  try {
    if (info.Length() != 3 || !info[0].IsString() || !info[2].IsArray())
      throw std::runtime_error("Expected model filename, bytes, and weight entries");
    auto weights = info[2].As<Napi::Array>();
    if (weights.Length() >= MaxEntries)
      throw std::runtime_error("ZIP package limit exceeded: too many ZIP entries");
    std::vector<OutputEntry> entries;
    entries.push_back({info[0].As<Napi::String>().Utf8Value(), Snapshot(info[1])});
    for (uint32_t i = 0; i < weights.Length(); ++i) {
      const auto value = weights.Get(i);
      if (!value.IsObject())
        throw std::runtime_error("Invalid ZIP weight entry");
      const auto object = value.As<Napi::Object>();
      if (!object.Get("name").IsString())
        throw std::runtime_error("Invalid ZIP weight name");
      entries.push_back(
          {object.Get("name").As<Napi::String>().Utf8Value(), Snapshot(object.Get("bytes"))});
    }
    auto *worker = new PackageWorker(info.Env(), std::move(entries));
    auto promise = worker->Promise();
    worker->Queue();
    return promise;
  } catch (const std::exception &error) {
    Napi::Error::New(info.Env(), error.what()).ThrowAsJavaScriptException();
    return info.Env().Undefined();
  }
}
} // namespace onnx_light_viewer
