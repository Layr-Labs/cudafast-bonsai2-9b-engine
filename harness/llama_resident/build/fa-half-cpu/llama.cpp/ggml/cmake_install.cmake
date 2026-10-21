# Install script for directory: /Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/llama.cpp/ggml/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/llama.cpp/ggml/src/libggml.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.a")
    execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-cpu.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-alloc.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-backend.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-blas.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-cann.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-cpp.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-cuda.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-opt.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-metal.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-rpc.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-virtgpu.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-sycl.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-vulkan.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-webgpu.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-zendnn.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/ggml-openvino.h"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/llama.cpp/ggml/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/llama.cpp/ggml/src/libggml-base.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.a")
    execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/llama.cpp/ggml/ggml-config.cmake"
    "/Users/dmitryzvada/Projects/RnPLab/2026/cfw-fa-half/harness/llama_resident/build/fa-half-cpu/llama.cpp/ggml/ggml-config-version.cmake"
    )
endif()

