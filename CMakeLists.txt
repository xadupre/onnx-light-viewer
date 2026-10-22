cmake_minimum_required(VERSION 3.24)
project(onnx_light_viewer LANGUAGES C CXX)

include(FetchContent)

set(ONNX_LIGHT_VERSION "0.1.35")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_asset "onnx-light-cpp-linux-x86_64.tar.gz")
  set(_sha256 "c17c71a215eff43a290f31252310bdd85862f8eb430dcd8c0820bbad099df0c6")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(_asset "onnx-light-cpp-linux-aarch64.tar.gz")
  set(_sha256 "64ad5921ed29263900de4bc95f3d7426b4d0f7040ca377217c908dabd924ded0")
elseif(APPLE)
  set(_asset "onnx-light-cpp-macos-universal2.tar.gz")
  set(_sha256 "c9552bde2486bbdf23dd689b88d66363ff174e11e758f1ea8360284adff6e78b")
elseif(WIN32 AND NODE_ARCH STREQUAL "x64")
  set(_asset "onnx-light-cpp-windows-AMD64.zip")
  set(_sha256 "0d448d5f7ec40881b394880e7355d21aeb97c6d4640e8fd2d15df64a264859b4")
elseif(WIN32 AND NODE_ARCH STREQUAL "arm64")
  set(_asset "onnx-light-cpp-windows-ARM64.zip")
  set(_sha256 "86e8d31486676d0c5aabd0df98ec9e07f35e9bb10425736cb56432fda0e1999a")
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

add_library(onnx_light_viewer SHARED native/addon.cc ${CMAKE_JS_SRC})
target_compile_features(onnx_light_viewer PRIVATE cxx_std_20)
target_compile_definitions(onnx_light_viewer PRIVATE
  NAPI_VERSION=8
  ONNX_LIGHT_VERSION="${ONNX_LIGHT_VERSION}"
)
target_include_directories(onnx_light_viewer PRIVATE ${CMAKE_JS_INC})
target_link_libraries(onnx_light_viewer PRIVATE
  onnx_light::lib_onnx_op onnx_light::lib_onnx_shape
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

file(MAKE_DIRECTORY "${_output}")
file(DOWNLOAD
  "https://raw.githubusercontent.com/xadupre/onnx-light/${ONNX_LIGHT_VERSION}/LICENSE"
  "${_output}/onnx-light.LICENSE"
  EXPECTED_HASH "SHA256=d495ab111fc496d1cd0291b4451da584d15c64b8af0895655e9fa52f801ea2c4"
)
configure_file(node_modules/node-addon-api/LICENSE.md "${_output}/node-addon-api.LICENSE" COPYONLY)
file(COPY "${onnx_light_cpp_SOURCE_DIR}/share/licenses/" DESTINATION "${_output}/licenses")
