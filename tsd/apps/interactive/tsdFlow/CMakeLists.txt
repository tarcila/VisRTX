## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdFlow)
project_add_executable(tsdFlow.cpp)
project_link_libraries(PUBLIC tsd_ui_imgui tsd_graph_nodes)
