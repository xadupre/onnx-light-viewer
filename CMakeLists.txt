cmake_minimum_required(VERSION 3.24)
project(onnx_light_viewer LANGUAGES C CXX)

if(MSVC)
  # Override cmake-js's static CRT before creating CPU dependency targets.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

include(FetchContent)

set(ONNX_LIGHT_VERSION "0.2.5")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_asset "onnx-light-cpp-linux-x86_64.tar.gz")
  set(_sha256 "96de7f7799f218acec77ddf5bb14771028122253bcf98ca896d36e72bac83da1")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(_asset "onnx-light-cpp-linux-aarch64.tar.gz")
  set(_sha256 "ab1dcab710427a2366276f5dc26f4d3f46a8af1297fd9f762c8d25caa30fd04c")
elseif(APPLE)
  set(_asset "onnx-light-cpp-macos-universal2.tar.gz")
  set(_sha256 "cc43feb08c5e962b08995ee85bea638374ac440b67601c5e0814e72020af6dd6")
elseif(WIN32 AND NODE_ARCH STREQUAL "x64")
  set(_asset "onnx-light-cpp-windows-AMD64.zip")
  set(_sha256 "c1f8af8d402d3d1b0b95272a7816b3a64aa616655f3452d51be05eb9c1955f63")
elseif(WIN32 AND NODE_ARCH STREQUAL "arm64")
  set(_asset "onnx-light-cpp-windows-ARM64.zip")
  set(_sha256 "d475332e8fb5dc30fa611c410d5eb7d41b119f13cde364b07ab734b51f45c38a")
else()
  message(FATAL_ERROR "No published onnx-light C++ release for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/${NODE_ARCH}.")
endif()

FetchContent_Declare(
  onnx_light_cpp
  URL "https://github.com/xadupre/onnx-light/releases/download/${ONNX_LIGHT_VERSION}/${_asset}"
  URL_HASH "SHA256=${_sha256}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnx_light_cpp)
find_package(onnx_light CONFIG REQUIRED
  PATHS "${onnx_light_cpp_SOURCE_DIR}/lib/cmake/onnx_light"
  NO_DEFAULT_PATH
)

set(ONNX_RUNTIME_GENAI_VERSION "0.17.0")
set(_genai_headers "${CMAKE_CURRENT_BINARY_DIR}/genai-include")
file(MAKE_DIRECTORY "${_genai_headers}")
file(DOWNLOAD
  "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/v${ONNX_RUNTIME_GENAI_VERSION}/src/ort_genai_c.h"
  "${_genai_headers}/ort_genai_c.h"
  EXPECTED_HASH "SHA256=9c69cc16a779aec3191b8d1f19a629e59667f34597e74fcbf0cdc24f39e0aef3"
  TLS_VERIFY ON
)

set(ONNX_LIGHT_CPU_VERSION "0.2.5")
set(ONNX_LIGHT_CPU_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_CPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_CPU_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_CPU_INSTALL OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  onnx_light_cpu
  URL "https://github.com/xadupre/onnx-light-cpu/archive/refs/tags/${ONNX_LIGHT_CPU_VERSION}.tar.gz"
  URL_HASH "SHA256=b30dc73b499b5c2c1309a74e170228dc681afd92f8eda29a0688ce6017d68f89"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnx_light_cpu)
set_target_properties(lib_onnx_light_cpu_backend_test PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_library(onnx_light_viewer SHARED
  native/addon.cc
  native/model.cc
  native/model_io.cc
  native/pipeline.cc
  native/model_storage.cc
  native/model_tensors.cc
  native/model_serialization.cc
  native/fusion.cc
  native/graph_analysis.cc
  native/memory_profile.cc
  native/model_dimensions.cc
  native/graph_snapshot.cc
  native/tensor_preview.cc
  native/tensor_statistics.cc
  native/prompt.cc
  native/run.cc
  ${CMAKE_JS_SRC}
)
target_compile_features(onnx_light_viewer PRIVATE cxx_std_20)
target_compile_definitions(onnx_light_viewer PRIVATE
  NAPI_VERSION=8
  ONNX_LIGHT_VERSION="${ONNX_LIGHT_VERSION}"
)
target_include_directories(onnx_light_viewer PRIVATE ${CMAKE_JS_INC} "${_genai_headers}")
target_link_libraries(onnx_light_viewer PRIVATE
  onnx_light::lib_onnx_op onnx_light::lib_onnx_shape
  onnx_light::lib_onnx_patterns
  onnx_light::lib_onnx_manipulations
  onnx_light::lib_onnx_kernels
  lib_onnx_light_cpu_kernels
  onnx_light::lib_onnx_backend_test ${CMAKE_JS_LIB} ${CMAKE_DL_LIBS}
)
set(_output "${CMAKE_CURRENT_SOURCE_DIR}/dist/native")
set_target_properties(onnx_light_viewer PROPERTIES
  PREFIX ""
  SUFFIX ".node"
  LIBRARY_OUTPUT_DIRECTORY "${_output}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${_output}"
  RUNTIME_OUTPUT_DIRECTORY "${_output}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_output}"
)
if(MSVC)
  set_property(TARGET onnx_light_viewer PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
  target_compile_options(onnx_light_viewer PRIVATE /EHsc /W4)
  if(CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(
      COMMAND "${CMAKE_AR}" "/def:${CMAKE_JS_NODELIB_DEF}" "/out:${CMAKE_JS_NODELIB_TARGET}" ${CMAKE_STATIC_LINKER_FLAGS}
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
else()
  target_compile_options(onnx_light_viewer PRIVATE -Wall -Wextra -Werror)
endif()

add_library(test_locale SHARED test/locale-addon.cc ${CMAKE_JS_SRC})
target_compile_features(test_locale PRIVATE cxx_std_20)
target_include_directories(test_locale PRIVATE ${CMAKE_JS_INC})
target_link_libraries(test_locale PRIVATE ${CMAKE_JS_LIB})
set_target_properties(test_locale PROPERTIES PREFIX "" SUFFIX ".node")
if(MSVC)
  set_property(TARGET test_locale PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
  target_compile_options(test_locale PRIVATE /EHsc)
endif()

add_library(test_model_storage SHARED test/model-storage-addon.cc native/model_tensors.cc ${CMAKE_JS_SRC})
target_compile_features(test_model_storage PRIVATE cxx_std_20)
target_include_directories(test_model_storage PRIVATE ${CMAKE_JS_INC})
target_link_libraries(test_model_storage PRIVATE
  onnx_light::lib_onnx_op onnx_light::lib_onnx_shape onnx_light::lib_onnx_patterns
  onnx_light::lib_onnx_kernels ${CMAKE_JS_LIB})
set_target_properties(test_model_storage PROPERTIES PREFIX "" SUFFIX ".node")
if(MSVC)
  set_property(TARGET test_model_storage PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
  target_compile_options(test_model_storage PRIVATE /EHsc)
endif()

file(MAKE_DIRECTORY "${_output}")
file(DOWNLOAD
  "https://raw.githubusercontent.com/xadupre/onnx-light/${ONNX_LIGHT_VERSION}/LICENSE"
  "${_output}/onnx-light.LICENSE"
  EXPECTED_HASH "SHA256=d495ab111fc496d1cd0291b4451da584d15c64b8af0895655e9fa52f801ea2c4"
)
configure_file(node_modules/node-addon-api/LICENSE.md "${_output}/node-addon-api.LICENSE" COPYONLY)
file(DOWNLOAD
  "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/v${ONNX_RUNTIME_GENAI_VERSION}/LICENSE"
  "${_output}/onnxruntime-genai.LICENSE"
  EXPECTED_HASH "SHA256=c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"
  TLS_VERIFY ON
)
configure_file("${onnx_light_cpu_SOURCE_DIR}/LICENSE" "${_output}/onnx-light-cpu.LICENSE" COPYONLY)
configure_file("${onnx_light_cpu_SOURCE_DIR}/ml_cpu_compute/LICENSE" "${_output}/ml-cpu-compute.LICENSE" COPYONLY)
file(COPY "${onnx_light_cpp_SOURCE_DIR}/share/licenses/" DESTINATION "${_output}/licenses")
