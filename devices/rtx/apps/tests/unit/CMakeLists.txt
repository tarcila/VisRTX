# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Host-only unit tests for the pure analytic primitive solvers. No CUDA/OptiX/GPU
# — these run on CPU and exercise devices/rtx/device/gpu/intersectPrimitives.h
# directly, so the intersection math is testable without a device.

project(rtx_test_Intersectors LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_Intersectors.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(${PROJECT_NAME} PRIVATE visrtx::glm)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::Intersectors COMMAND ${PROJECT_NAME})

project(rtx_test_IntersectorsOracle LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_IntersectorsOracle.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(${PROJECT_NAME} PRIVATE visrtx::glm)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::IntersectorsOracle COMMAND ${PROJECT_NAME})

project(rtx_test_IsosurfaceLinear LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_IsosurfaceLinear.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(${PROJECT_NAME} PRIVATE visrtx::glm)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::IsosurfaceLinear COMMAND ${PROJECT_NAME})

# lightPickPower operates on the GPU LightGPUData struct, so this host test pulls
# the OptiX/CUDA/nanovdb/anari headers its definition transitively needs.
project(rtx_test_LightPickPower LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_LightPickPower.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
  ${OptiX_ROOT_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE
  visrtx::glm
  anari::anari_headers
  CUDA::cudart)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::LightPickPower COMMAND ${PROJECT_NAME})

# The shared rect/ring area-light leaves (ADR 0009). lightGeometry.h is kept
# CUDA-free precisely so it can be tested here; like lightPickPower it still
# needs the OptiX/CUDA/nanovdb/anari headers that GPU struct definitions pull in.
project(rtx_test_LightGeometry LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_LightGeometry.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
  ${OptiX_ROOT_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE
  visrtx::glm
  anari::anari_headers
  CUDA::cudart)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::LightGeometry COMMAND ${PROJECT_NAME})

# The analytic ray/rect solver against an independent fp64 two-triangle oracle —
# the triangulated proxy ADR 0009 rejected as a representation, used here only as
# a test oracle.
project(rtx_test_LightGeometryOracle LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_LightGeometryOracle.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
  ${OptiX_ROOT_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE
  visrtx::glm
  anari::anari_headers
  CUDA::cudart)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::LightGeometryOracle COMMAND ${PROJECT_NAME})

# Shared production ordinal seam (no GPU initialization).
add_executable(rtx_test_SampleOrdinal test_SampleOrdinal.cpp)
target_include_directories(rtx_test_SampleOrdinal PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
  ${OptiX_ROOT_DIR}/include)
target_link_libraries(rtx_test_SampleOrdinal PRIVATE
  CUDA::cudart visrtx::glm anari::anari_headers)
target_compile_definitions(rtx_test_SampleOrdinal PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_compile_features(rtx_test_SampleOrdinal PRIVATE cxx_std_17)
add_test(NAME rtx::SampleOrdinal COMMAND rtx_test_SampleOrdinal)

# Compile verbatim production shader expressions at the agreed host test seam.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME rtx::SobolTableHeader
  COMMAND Python3::Interpreter -B
    ${CMAKE_CURRENT_LIST_DIR}/sobol/test_table_header.py)
set_tests_properties(rtx::SobolTableHeader PROPERTIES TIMEOUT 30)
set(draw_order_header ${CMAKE_CURRENT_BINARY_DIR}/shader_expressions.h)
add_custom_command(OUTPUT ${draw_order_header}
  COMMAND Python3::Interpreter ${CMAKE_CURRENT_LIST_DIR}/sobol/shader_expressions.py
    ${CMAKE_CURRENT_LIST_DIR}/../../../../.. ${draw_order_header}
  DEPENDS sobol/shader_expressions.py
    ../../../device/material/shaders/PhysicallyBasedShader_ptx.cu
    ../../../device/material/shaders/MDLShader_ptx.cu
    ../../../device/gpu/gpu_util.h
    ../../../device/gpu/sampleLight.h
  VERBATIM)
add_executable(rtx_test_DrawOrder test_DrawOrder.cpp ${draw_order_header})
target_include_directories(rtx_test_DrawOrder PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
  ${OptiX_ROOT_DIR}/include)
target_link_libraries(rtx_test_DrawOrder PRIVATE
  CUDA::cudart visrtx::glm anari::anari_headers)
target_compile_definitions(rtx_test_DrawOrder PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_compile_features(rtx_test_DrawOrder PRIVATE cxx_std_17)
add_test(NAME rtx::DrawOrder COMMAND rtx_test_DrawOrder)

add_executable(rtx_test_LightDrawOrder test_LightDrawOrder.cu)
target_include_directories(rtx_test_LightDrawOrder PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
  ${OptiX_ROOT_DIR}/include)
target_link_libraries(rtx_test_LightDrawOrder PRIVATE
  CUDA::cudart visrtx::glm anari::anari_headers)
target_compile_definitions(rtx_test_LightDrawOrder PRIVATE GLM_ENABLE_EXPERIMENTAL)
set_target_properties(rtx_test_LightDrawOrder PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(rtx_test_LightDrawOrder PRIVATE
  --use_fast_math --expt-relaxed-constexpr)
add_test(NAME rtx::LightDrawOrder COMMAND rtx_test_LightDrawOrder)
set_tests_properties(rtx::DrawOrder rtx::LightDrawOrder PROPERTIES TIMEOUT 60)

# Production Sobol sampler against independently sourced Joe-Kuo parameters.
add_executable(rtx_test_Sobol test_Sobol.cpp)
add_executable(rtx_test_SobolCUDA test_SobolCUDA.cu)
foreach(sobol_target rtx_test_Sobol rtx_test_SobolCUDA)
  target_include_directories(${sobol_target} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../device)
  target_link_libraries(${sobol_target} PRIVATE CUDA::cudart)
  target_compile_features(${sobol_target} PRIVATE cxx_std_17)
  set_target_properties(${sobol_target} PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(${sobol_target} PRIVATE
    SOBOL_PARAMETER_FILE="${CMAKE_CURRENT_LIST_DIR}/sobol/joe-kuo.txt"
    SOBOL_VECTOR_FILE="${CMAKE_CURRENT_LIST_DIR}/sobol/vectors.txt"
    SOBOL_SEED_VECTOR_FILE="${CMAKE_CURRENT_LIST_DIR}/sobol/seed-vectors.txt")
endforeach()
# Match the RTX PTX's arithmetic configuration.
target_compile_options(rtx_test_SobolCUDA PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr>)
add_test(NAME rtx::Sobol COMMAND rtx_test_Sobol)
add_test(NAME rtx::SobolCUDA COMMAND rtx_test_SobolCUDA)
set_tests_properties(rtx::Sobol rtx::SobolCUDA PROPERTIES TIMEOUT 60)

# Production sampler contracts, compiled separately so Release cannot mask the
# assertion-enabled gate (and assertions cannot mask the NDEBUG gate).
if(UNIX)
  function(configure_sampler_contract_test target mode timeout)
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}/../../../device
      ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
      ${OptiX_ROOT_DIR}/include)
    target_link_libraries(${target} PRIVATE
      CUDA::cudart visrtx::glm anari::anari_headers)
    target_compile_definitions(${target} PRIVATE GLM_ENABLE_EXPERIMENTAL)
    target_compile_features(${target} PRIVATE cxx_std_17)
    if(mode STREQUAL "Assert")
      target_compile_options(${target} PRIVATE -UNDEBUG)
    else()
      target_compile_definitions(${target} PRIVATE NDEBUG)
    endif()
    string(REPLACE "rtx_test_" "rtx::" test_name ${target})
    add_test(NAME ${test_name} COMMAND ${target})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT ${timeout})
  endfunction()

  foreach(mode Assert NDEBUG)
    add_executable(rtx_test_SamplerContracts${mode} test_SamplerContracts.cpp)
    configure_sampler_contract_test(rtx_test_SamplerContracts${mode} ${mode} 120)

    add_executable(rtx_test_SamplerContractsCUDA${mode} test_SamplerContractsCUDA.cu)
    configure_sampler_contract_test(rtx_test_SamplerContractsCUDA${mode} ${mode} 240)
    set_target_properties(rtx_test_SamplerContractsCUDA${mode} PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(rtx_test_SamplerContractsCUDA${mode} PRIVATE
      --use_fast_math --expt-relaxed-constexpr)
  endforeach()
endif()

# Shared optical coefficients and typed bindings, independent of OptiX/ANARI.
add_executable(rtx_test_VolumeOptics test_VolumeOptics.cpp)
target_include_directories(rtx_test_VolumeOptics PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(rtx_test_VolumeOptics PRIVATE visrtx::glm)
target_compile_features(rtx_test_VolumeOptics PRIVATE cxx_std_17)
add_test(NAME rtx::VolumeOptics COMMAND rtx_test_VolumeOptics)

# Run identical cases on CUDA, also with the production PTX fast-math setting.
foreach(variant IN ITEMS CUDA CUDAFastMath)
  set(target rtx_test_VolumeOptics${variant})
  add_executable(${target} test_VolumeOptics.cu)
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../device)
  target_link_libraries(${target} PRIVATE visrtx::glm CUDA::cudart)
  set_target_properties(${target} PROPERTIES CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
  target_compile_options(${target} PRIVATE --expt-relaxed-constexpr)
  if(variant STREQUAL "CUDAFastMath")
    target_compile_options(${target} PRIVATE --use_fast_math)
  endif()
  add_test(NAME rtx::VolumeOptics${variant} COMMAND ${target})
  set_tests_properties(rtx::VolumeOptics${variant} PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 60)
endforeach()

# Calibrated production blackbody against independent direct spectral data.
add_executable(rtx_test_Blackbody test_Blackbody.cpp)
add_executable(rtx_test_BlackbodyCUDA test_Blackbody.cu)
add_executable(rtx_test_BlackbodyCUDAFastMath test_Blackbody.cu)
foreach(target rtx_test_Blackbody rtx_test_BlackbodyCUDA rtx_test_BlackbodyCUDAFastMath)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../device)
  target_link_libraries(${target} PRIVATE visrtx::glm CUDA::cudart)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_compile_definitions(${target} PRIVATE
    BLACKBODY_ORACLE="${CMAKE_CURRENT_LIST_DIR}/../../../device/gpu/blackbody/data/reference.txt")
  string(REPLACE "rtx_test_" "rtx::" test_name ${target})
  add_test(NAME ${test_name} COMMAND ${target})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
endforeach()
target_compile_options(rtx_test_BlackbodyCUDAFastMath PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
add_test(NAME rtx::BlackbodyOracle COMMAND Python3::Interpreter -B
  ${CMAKE_CURRENT_LIST_DIR}/../../../../../tools/principled_volume/blackbody_reference.py)
set_tests_properties(rtx::BlackbodyOracle PROPERTIES TIMEOUT 60)
add_test(NAME rtx::BlackbodyInstall COMMAND ${CMAKE_COMMAND}
  -DBUILD_DIR=${CMAKE_BINARY_DIR} -DCONFIG=$<CONFIG>
  -DSOURCE_DIR=${CMAKE_CURRENT_LIST_DIR}/../../../device/gpu/blackbody/data
  -DDATA_DIR=${CMAKE_INSTALL_DATAROOTDIR}
  -P ${CMAKE_CURRENT_LIST_DIR}/blackbody/checkInstall.cmake)
set_tests_properties(rtx::BlackbodyInstall PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)

# Isolated reference optics.
add_executable(rtx_test_VolumeOpticsReference test_VolumeOpticsReference.cpp)
add_executable(rtx_test_VolumeOpticsReferenceCUDA test_VolumeOpticsReference.cu)
add_executable(rtx_test_VolumeOpticsReferenceCUDAFastMath test_VolumeOpticsReference.cu)
foreach(target rtx_test_VolumeOpticsReference rtx_test_VolumeOpticsReferenceCUDA
    rtx_test_VolumeOpticsReferenceCUDAFastMath)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../device)
  target_link_libraries(${target} PRIVATE visrtx::glm CUDA::cudart)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
  target_compile_definitions(${target} PRIVATE
    VOLUME_LOG_ORACLE="${CMAKE_CURRENT_LIST_DIR}/volumeReference/log-oracle.txt")
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
  elseif(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/fp:strict>)
  endif()
  string(REPLACE "rtx_test_" "rtx::" test_name ${target})
  add_test(NAME ${test_name} COMMAND ${target})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
endforeach()
target_compile_options(rtx_test_VolumeOpticsReferenceCUDAFastMath PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
add_test(NAME rtx::VolumeLogOracle
  COMMAND Python3::Interpreter -B ${CMAKE_CURRENT_LIST_DIR}/volumeReference/oracle.py)
set_tests_properties(rtx::VolumeLogOracle PROPERTIES TIMEOUT 30)

# Raw-range/revision preparation and controlled reference bindings.
add_executable(rtx_test_ReferenceFieldBindings test_ReferenceFieldBindings.cpp)
add_executable(rtx_test_ReferenceFieldBindingsCUDA test_ReferenceFieldBindings.cu)
add_executable(rtx_test_ReferenceFieldBindingsCUDAFastMath test_ReferenceFieldBindings.cu)
foreach(target rtx_test_ReferenceFieldBindings rtx_test_ReferenceFieldBindingsCUDA
    rtx_test_ReferenceFieldBindingsCUDAFastMath)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../device)
  target_link_libraries(${target} PRIVATE visrtx::glm CUDA::cudart)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
  elseif(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/fp:strict>)
  endif()
  string(REPLACE "rtx_test_" "rtx::" test_name ${target})
  add_test(NAME ${test_name} COMMAND ${target})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
endforeach()
target_compile_options(rtx_test_ReferenceFieldBindingsCUDAFastMath PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Real dense CUDA textures and NanoVDB Vec3f trees through typed bindings.
add_executable(rtx_test_StoredFieldSampler
  test_StoredFieldSampler.cu
  ${CMAKE_CURRENT_LIST_DIR}/../../../device/spatial_field/FieldTexture.cu
  ${CMAKE_CURRENT_LIST_DIR}/../../../device/utility/MemoryAllocation.cpp)
target_include_directories(rtx_test_StoredFieldSampler PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
  ${OptiX_ROOT_DIR}/include)
target_link_libraries(rtx_test_StoredFieldSampler PRIVATE
  visrtx::glm anari::anari_headers CUDA::cudart)
target_compile_definitions(rtx_test_StoredFieldSampler PRIVATE GLM_ENABLE_EXPERIMENTAL)
set_target_properties(rtx_test_StoredFieldSampler PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
target_compile_options(rtx_test_StoredFieldSampler PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
add_test(NAME rtx::StoredFieldSampler COMMAND rtx_test_StoredFieldSampler)
set_tests_properties(rtx::StoredFieldSampler PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

# Concrete field snapshots after public ANARI commit/update, then direct CUDA
# sampling at the private reference seam (no new public sampling property).
add_executable(rtx_test_StoredFieldReference test_StoredFieldReference.cu
  ${CMAKE_CURRENT_LIST_DIR}/../../../device/utility/MemoryAllocation.cpp)
target_include_directories(rtx_test_StoredFieldReference PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include)
target_link_libraries(rtx_test_StoredFieldReference PRIVATE
  anari_library_visrtx anari::anari_static anari::helium visrtx::glm
  OptiX::OptiX CUDA::cudart)
target_compile_definitions(rtx_test_StoredFieldReference PRIVATE GLM_ENABLE_EXPERIMENTAL)
set_target_properties(rtx_test_StoredFieldReference PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
target_compile_options(rtx_test_StoredFieldReference PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr;--extended-lambda>)
add_test(NAME rtx::StoredFieldReference COMMAND rtx_test_StoredFieldReference)
set_tests_properties(rtx::StoredFieldReference PROPERTIES TIMEOUT 120
  ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:anari_library_visrtx>:$ENV{LD_LIBRARY_PATH}")

# Public commits with direct inspection/execution of the private GPU payload.
add_executable(rtx_test_PrincipledVolumeState test_PrincipledVolumeState.cu
  ${CMAKE_CURRENT_LIST_DIR}/../../../device/utility/MemoryAllocation.cpp)
target_include_directories(rtx_test_PrincipledVolumeState PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include)
target_link_libraries(rtx_test_PrincipledVolumeState PRIVATE
  anari_library_visrtx anari::anari_static anari::helium visrtx::glm
  OptiX::OptiX CUDA::cudart)
target_compile_definitions(rtx_test_PrincipledVolumeState PRIVATE GLM_ENABLE_EXPERIMENTAL)
set_target_properties(rtx_test_PrincipledVolumeState PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
target_compile_options(rtx_test_PrincipledVolumeState PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr;--extended-lambda>)
add_test(NAME rtx::PrincipledVolumeState COMMAND rtx_test_PrincipledVolumeState)
set_tests_properties(rtx::PrincipledVolumeState PROPERTIES TIMEOUT 120
  ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:anari_library_visrtx>:$ENV{LD_LIBRARY_PATH}")

# The production preview step selector, including representability fallbacks.
add_executable(rtx_test_PreviewLattice test_PreviewLattice.cpp)
add_executable(rtx_test_PreviewLatticeCUDA test_PreviewLattice.cu)
add_executable(rtx_test_PreviewLatticeCUDAFastMath test_PreviewLattice.cu)
foreach(target rtx_test_PreviewLattice rtx_test_PreviewLatticeCUDA rtx_test_PreviewLatticeCUDAFastMath)
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../device
    ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include)
  target_link_libraries(${target} PRIVATE visrtx::glm anari::anari_headers OptiX::OptiX CUDA::cudart)
  target_compile_definitions(${target} PRIVATE GLM_ENABLE_EXPERIMENTAL)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  string(REPLACE "rtx_test_" "rtx::" test_name ${target})
  add_test(NAME ${test_name} COMMAND ${target})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
endforeach()
target_compile_options(rtx_test_PreviewLatticeCUDAFastMath PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# Counted aggregate traversal and optical consumption, independent analytic oracles.
add_executable(rtx_test_AggregateVolume test_AggregateVolume.cpp)
add_executable(rtx_test_AggregateVolumeCUDA test_AggregateVolume.cpp test_AggregateVolume.cu)
add_executable(rtx_test_AggregateVolumeCUDAFastMath test_AggregateVolume.cpp test_AggregateVolume.cu)
target_compile_definitions(rtx_test_AggregateVolumeCUDA PRIVATE AGGREGATE_CUDA)
target_compile_definitions(rtx_test_AggregateVolumeCUDAFastMath PRIVATE AGGREGATE_CUDA)
foreach(target rtx_test_AggregateVolume rtx_test_AggregateVolumeCUDA rtx_test_AggregateVolumeCUDAFastMath)
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../device
    ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include)
  target_link_libraries(${target} PRIVATE visrtx::glm anari::anari_headers OptiX::OptiX CUDA::cudart)
  target_compile_definitions(${target} PRIVATE GLM_ENABLE_EXPERIMENTAL)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  string(REPLACE "rtx_test_" "rtx::" test_name ${target})
  add_test(NAME ${test_name} COMMAND ${target})
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 120)
endforeach()
target_compile_options(rtx_test_AggregateVolumeCUDAFastMath PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

add_executable(rtx_test_WorldVolumeReference test_WorldVolumeReference.cu
  ${CMAKE_CURRENT_LIST_DIR}/../../../device/utility/MemoryAllocation.cpp)
target_include_directories(rtx_test_WorldVolumeReference PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include)
target_link_libraries(rtx_test_WorldVolumeReference PRIVATE
  anari_library_visrtx anari::anari_static anari::helium visrtx::glm OptiX::OptiX CUDA::cudart)
target_compile_definitions(rtx_test_WorldVolumeReference PRIVATE GLM_ENABLE_EXPERIMENTAL)
set_target_properties(rtx_test_WorldVolumeReference PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH})
target_compile_options(rtx_test_WorldVolumeReference PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr;--extended-lambda>)
add_test(NAME rtx::WorldVolumeReference COMMAND rtx_test_WorldVolumeReference)
set_tests_properties(rtx::WorldVolumeReference PROPERTIES TIMEOUT 120
  ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:anari_library_visrtx>:$ENV{LD_LIBRARY_PATH}")

# Standard-library-only executor tests.
find_package(Threads REQUIRED)
add_executable(rtx_test_TaskExecutor
  test_TaskExecutor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../../device/utility/TaskExecutor.cpp)
target_include_directories(rtx_test_TaskExecutor PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(rtx_test_TaskExecutor PRIVATE Threads::Threads)
target_compile_features(rtx_test_TaskExecutor PRIVATE cxx_std_17)
add_test(NAME rtx::TaskExecutor COMMAND rtx_test_TaskExecutor)
add_test(NAME rtx::TaskExecutorSerial COMMAND rtx_test_TaskExecutor --serial)
add_test(NAME rtx::TaskExecutorReserveFailure
  COMMAND rtx_test_TaskExecutor --reserve-failure)
set_tests_properties(rtx::TaskExecutor rtx::TaskExecutorSerial
  rtx::TaskExecutorReserveFailure PROPERTIES TIMEOUT 30)

if (VISRTX_ENABLE_MDL_SUPPORT)
  # Exercise the real registry directly (its symbols are private to the device
  # DSO). Reuse the device's embedded MDL/PTX, without creating a GPU device.
  set(mdl_device_binary_dir ${CMAKE_CURRENT_BINARY_DIR}/../../../device)
  set(mdl_registry_resources
    ${mdl_device_binary_dir}/mdl/default.mdl.cpp
    ${mdl_device_binary_dir}/mdl/physically_based.mdl.cpp)
  set_source_files_properties(${mdl_registry_resources} PROPERTIES GENERATED TRUE)
  add_executable(rtx_test_MdlMaterialRegistry
    test_MdlMaterialRegistry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/mdl/MaterialRegistry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/utility/TaskExecutor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/mdl/ptx.cpp
    ${mdl_registry_resources})
  target_link_libraries(rtx_test_MdlMaterialRegistry PRIVATE
    anari_library_visrtx libmdl fmt::fmt nonstd::scope-lite
    visrtx::glm OptiX::OptiX anari::helium
    CUDA::cudart Threads::Threads)
  target_include_directories(rtx_test_MdlMaterialRegistry PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include)
  target_compile_definitions(rtx_test_MdlMaterialRegistry PRIVATE USE_MDL)
  target_compile_features(rtx_test_MdlMaterialRegistry PRIVATE cxx_std_17)
  add_test(NAME rtx::MdlMaterialRegistry COMMAND rtx_test_MdlMaterialRegistry)
  set_tests_properties(rtx::MdlMaterialRegistry PROPERTIES TIMEOUT 120)

  project(rtx_test_MdlEmissionClassifier LANGUAGES CXX)

  add_executable(${PROJECT_NAME} test_MdlEmissionClassifier.cpp)
  # libmdl's Core.h includes <fmt/core.h> but links fmt PRIVATE, so a consumer
  # that only links libmdl must pull fmt itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE libmdl fmt::fmt)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

  add_test(NAME rtx::MdlEmissionClassifier COMMAND ${PROJECT_NAME})
  # The test SKIPs (not fails) when the MDL SDK shared library is not loadable.
  set_tests_properties(rtx::MdlEmissionClassifier PROPERTIES SKIP_RETURN_CODE 77)

  project(rtx_test_MdlSceneDataDescriptor LANGUAGES CXX)
  add_executable(${PROJECT_NAME} test_MdlSceneDataDescriptor.cpp)
  target_link_libraries(${PROJECT_NAME} PRIVATE libmdl fmt::fmt)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
  add_test(NAME rtx::MdlSceneDataDescriptor COMMAND ${PROJECT_NAME})
  set_tests_properties(rtx::MdlSceneDataDescriptor PROPERTIES SKIP_RETURN_CODE 77)

  # Binding resolution uses only owned host data: no CUDA or SDK runtime calls.
  project(rtx_test_MdlSceneDataInstance LANGUAGES CXX)
  add_executable(${PROJECT_NAME}
    test_MdlSceneDataInstance.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/mdl/SceneDataInstance.cpp)
  target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../device)
  target_link_libraries(${PROJECT_NAME} PRIVATE libmdl)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
  add_test(NAME rtx::MdlSceneDataInstance COMMAND ${PROJECT_NAME})
endif()
