## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdMultiDeviceViewer)
project_add_executable(
  MultiDeviceSceneRenderPass.cpp
  multiDeviceViewer.cpp
  MultiDeviceViewport.cpp
)
project_link_libraries(PUBLIC tsd_ui_imgui)
