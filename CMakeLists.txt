cmake_minimum_required(VERSION 3.24)
project(onnx_light_viewer LANGUAGES C CXX)

if(MSVC)
  # Override cmake-js's static CRT before creating CPU dependency targets.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

include(FetchContent)

set(ONNX_LIGHT_VERSION "0.2.1")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_asset "onnx-light-cpp-linux-x86_64.tar.gz")
  set(_sha256 "a11ef7f0942627bcf31353f5856e998a114b35eea7c460d67bab9ca60fd7628a")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(_asset "onnx-light-cpp-linux-aarch64.tar.gz")
  set(_sha256 "48ea7f63428c771fa0b239c39a21cd7957135c20f1ff4727d3ac64aec4c6058c")
elseif(APPLE)
  set(_asset "onnx-light-cpp-macos-universal2.tar.gz")
  set(_sha256 "1d76b8460a9a7d0ca6436906ffdf597644c2e0c56786a0ff057d33128bb0f4d3")
elseif(WIN32 AND NODE_ARCH STREQUAL "x64")
  set(_asset "onnx-light-cpp-windows-AMD64.zip")
  set(_sha256 "18c6f896a2342106891a15dee41d8119cc0af7d39b839eae3808762431fee0c0")
elseif(WIN32 AND NODE_ARCH STREQUAL "arm64")
  set(_asset "onnx-light-cpp-windows-ARM64.zip")
  set(_sha256 "0b226d8a3e3b9563f02efd9ef1bb4e6e3e2a331576e02f22caa8c100bee47952")
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
  ${CMAKE_JS_SRC}
)
target_compile_features(onnx_light_viewer PRIVATE cxx_std_20)
target_compile_definitions(onnx_light_viewer PRIVATE
  NAPI_VERSION=8
  ONNX_LIGHT_VERSION="${ONNX_LIGHT_VERSION}"
)
target_include_directories(onnx_light_viewer PRIVATE ${CMAKE_JS_INC})
target_link_libraries(onnx_light_viewer PRIVATE
  onnx_light::lib_onnx_op onnx_light::lib_onnx_shape
  onnx_light::lib_onnx_patterns
  onnx_light::lib_onnx_kernels
  lib_onnx_light_cpu_kernels
  onnx_light::lib_onnx_backend_test ${CMAKE_JS_LIB}
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
configure_file("${onnx_light_cpu_SOURCE_DIR}/LICENSE" "${_output}/onnx-light-cpu.LICENSE" COPYONLY)
configure_file("${onnx_light_cpu_SOURCE_DIR}/ml_cpu_compute/LICENSE" "${_output}/ml-cpu-compute.LICENSE" COPYONLY)
file(COPY "${onnx_light_cpp_SOURCE_DIR}/share/licenses/" DESTINATION "${_output}/licenses")
