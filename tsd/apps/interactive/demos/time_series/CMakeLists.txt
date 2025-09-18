## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdDemoTimeSeries)
project_add_executable(AnimationControls.cpp tsdDemoTimeSeries.cpp)
project_link_libraries(tsd_ui_imgui)
