cmake_minimum_required(VERSION 3.24)
project(onnx_light_viewer LANGUAGES C CXX)

if(MSVC)
  # Override cmake-js's static CRT before creating CPU dependency targets.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

include(FetchContent)

set(ONNX_LIGHT_VERSION "0.2.2")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_asset "onnx-light-cpp-linux-x86_64.tar.gz")
  set(_sha256 "00ac4f7e7638e2dd99dc41ca8a084b31a1432185ab8353bfd73220a20d0d617b")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(_asset "onnx-light-cpp-linux-aarch64.tar.gz")
  set(_sha256 "a92bb0fd7cfb099facf986f4f664d0189e0069989a99cdfda9634d3622d4ac69")
elseif(APPLE)
  set(_asset "onnx-light-cpp-macos-universal2.tar.gz")
  set(_sha256 "abe970602fb9bcb5cea68deb55ef03ae9c255f90da7e17a154c9ee8ef5784989")
elseif(WIN32 AND NODE_ARCH STREQUAL "x64")
  set(_asset "onnx-light-cpp-windows-AMD64.zip")
  set(_sha256 "afdb4e39172ee5789cc22f3d171a95460ad5bb7464fef4d531a8c7938c06ea77")
elseif(WIN32 AND NODE_ARCH STREQUAL "arm64")
  set(_asset "onnx-light-cpp-windows-ARM64.zip")
  set(_sha256 "492acad1768573d3e5b92228167ded53b618f4b07402cf59154d0dff5cc7f94c")
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

set(ONNX_LIGHT_CPU_VERSION "0.2.1.1")
set(ONNX_LIGHT_CPU_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_CPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_CPU_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_CPU_INSTALL OFF CACHE BOOL "" FORCE)
set(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  onnx_light_cpu
  URL "https://github.com/xadupre/onnx-light-cpu/archive/refs/tags/${ONNX_LIGHT_CPU_VERSION}.tar.gz"
  URL_HASH "SHA256=ec9ad7a8d58fd24716f4c61d9cf3fdb8bf5d791e7fbea7cbbdb78b8f1e78ac0c"
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
  native/graph_snapshot.cc
  native/tensor_preview.cc
  native/tensor_statistics.cc
  native/prompt.cc
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
