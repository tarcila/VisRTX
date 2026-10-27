# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Install into a fresh build-local prefix: stale files cannot mask omissions.
if(IS_ABSOLUTE "${DATA_DIR}")
  message(FATAL_ERROR "Blackbody install test requires a relative install dataroot")
endif()
string(RANDOM LENGTH 16 suffix)
set(prefix "${BUILD_DIR}/blackbody-install-${suffix}")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}"
  --config "${CONFIG}" --prefix "${prefix}" --component blackbody-data
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Blackbody component install failed: ${output}\n${error}")
endif()
foreach(name ATTRIBUTION.md LICENSE.md tables.h reference.txt
    CIE_std_illum_D65.csv CIE_std_illum_D65.csv_metadata_v2.json
    CIE_xyz_1931_2deg.csv CIE_xyz_1931_2deg.csv_metadata.json)
  set(installed "${prefix}/${DATA_DIR}/VisRTX/blackbody/${name}")
  if(NOT EXISTS "${installed}")
    message(FATAL_ERROR "Missing installed blackbody file: ${installed}")
  endif()
  file(SHA256 "${SOURCE_DIR}/${name}" expected)
  file(SHA256 "${installed}" actual)
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "Installed blackbody content differs: ${name}")
  endif()
endforeach()
message(STATUS "Eight blackbody data/attribution files installed byte-for-byte")
file(REMOVE_RECURSE "${prefix}")
