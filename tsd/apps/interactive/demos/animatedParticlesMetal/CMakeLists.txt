## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (NOT TSD_USE_METAL)
  return()
endif()

project(tsdDemoAnimatedParticlesMetal)
project_add_executable(
  particle_system.cpp
  SimulationControls.cpp
  tsdDemoAnimatedParticlesMetal.cpp
)
project_link_libraries(tsd_ui_imgui tsd_algorithms)
