## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(animated_particles)
add_subdirectory(animated_volume)
add_subdirectory(array_instancing)
add_subdirectory(time_series)
option(BUILD_VISKORES_DEMO "Build Viskores demo viewer" OFF)
if (BUILD_VISKORES_DEMO)
  add_subdirectory(viskores)
endif()
