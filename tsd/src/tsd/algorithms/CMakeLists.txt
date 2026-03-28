## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_algorithms)

project_add_library(STATIC)

# CPU backend (always present)
project_sources(PRIVATE
  cpu/toneMap.cpp
  cpu/autoExposure.cpp
  cpu/outputTransform.cpp
  cpu/visualizeAOV.cpp
  cpu/outline.cpp
  cpu/clearBuffers.cpp
  cpu/convertColorBuffer.cpp
  cpu/depthCompositeFrame.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/../..
)

project_link_libraries(PUBLIC tsd_core)

# CPU internal: TBB or serial
if (TSD_USE_TBB)
  find_package(TBB REQUIRED)
  project_link_libraries(PRIVATE TBB::tbb)
  project_compile_definitions(PRIVATE ENABLE_TBB)
endif()

# GPU backend: CUDA
if (TSD_USE_CUDA)
  find_package(thrust REQUIRED HINTS ${CUDAToolkit_LIBRARY_DIR}/cmake)
  project_link_libraries(PUBLIC CUDA::cudart Thrust::Thrust)
  project_compile_definitions(PUBLIC TSD_ALGORITHMS_HAS_CUDA)

  set(TSD_ALGORITHMS_CUDA_SOURCES
    cuda/toneMap.cu
    cuda/autoExposure.cu
    cuda/outputTransform.cu
    cuda/visualizeAOV.cu
    cuda/outline.cu
    cuda/clearBuffers.cu
    cuda/convertColorBuffer.cu
    cuda/depthCompositeFrame.cu
  )
  project_sources(PRIVATE ${TSD_ALGORITHMS_CUDA_SOURCES})
  set_source_files_properties(${TSD_ALGORITHMS_CUDA_SOURCES}
    PROPERTIES
      COMPILE_OPTIONS "--extended-lambda;--expt-relaxed-constexpr;-Xcudafe=--diag_suppress=20012;-Wno-deprecated-gpu-targets"
      LANGUAGE CUDA
  )
endif()

# GPU backend: Metal
if (TSD_USE_METAL)
  if (NOT APPLE)
    message(FATAL_ERROR "TSD_USE_METAL requires macOS")
  endif()

  if (TSD_USE_CUDA)
    message(FATAL_ERROR "TSD_USE_CUDA and TSD_USE_METAL are mutually exclusive")
  endif()

  anari_sdk_fetch_project(
    NAME metal_cpp
    URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
  )
  project_include_directories(PUBLIC ${metal_cpp_LOCATION})
  project_compile_definitions(PUBLIC TSD_ALGORITHMS_HAS_METAL)

  # Embed shader source for runtime compilation (no xcrun/Xcode needed)
  set(TSD_METAL_SHADER_SOURCES
    # Math headers first (dependency order — inlined into MSL source)
    math/device_macros.h
    math/vec_types.h
    math/math_compat.h
    math/color.h
    math/tonemap_curves.h
    # Compute shaders
    metal/shaders/clearBuffers.metal
    metal/shaders/toneMap.metal
    metal/shaders/convertColorBuffer.metal
    metal/shaders/outputTransform.metal
    metal/shaders/autoExposure.metal
    metal/shaders/visualizeAOV.metal
    metal/shaders/outline.metal
    metal/shaders/composite.metal
  )

  # Build absolute paths for the manifest and DEPENDS
  set(TSD_METAL_SHADER_PATHS)
  foreach(SRC ${TSD_METAL_SHADER_SOURCES})
    list(APPEND TSD_METAL_SHADER_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/${SRC})
  endforeach()

  set(SHADER_MANIFEST ${CMAKE_CURRENT_BINARY_DIR}/metal_shader_manifest.txt)
  string(REPLACE ";" "\n" _shader_list "${TSD_METAL_SHADER_PATHS}")
  file(WRITE ${SHADER_MANIFEST} "${_shader_list}\n")

  set(GENERATED_SHADER_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/GeneratedShaderSource.h)

  add_custom_command(
    OUTPUT ${GENERATED_SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND}
      -DSHADER_MANIFEST=${SHADER_MANIFEST}
      -DOUTPUT_FILE=${GENERATED_SHADER_HEADER}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedShaders.cmake
    DEPENDS ${TSD_METAL_SHADER_PATHS}
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedShaders.cmake
    COMMENT "Embedding Metal shader source for runtime compilation"
  )

  project_sources(PRIVATE
    metal/MetalContext.mm
    metal/runtime.cpp
    metal/clearBuffers.cpp
    metal/toneMap.cpp
    metal/convertColorBuffer.cpp
    metal/outputTransform.cpp
    metal/autoExposure.cpp
    metal/visualizeAOV.cpp
    metal/outline.cpp
    ${METALLIB_C}
    metal/toneMap.cpp
    metal/convertColorBuffer.cpp
    metal/outputTransform.cpp
    metal/autoExposure.cpp
    metal/visualizeAOV.cpp
    metal/outline.cpp
    ${GENERATED_SHADER_HEADER}
  )

  project_include_directories(PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
  find_library(IOSURFACE_FRAMEWORK IOSurface REQUIRED)
  project_link_libraries(PUBLIC
    ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
    ${COREVIDEO_FRAMEWORK} ${IOSURFACE_FRAMEWORK})
endif()
