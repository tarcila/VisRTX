## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_metal)

project_add_library(STATIC)
project_sources(PRIVATE interop.cpp)
project_include_directories(PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
project_compile_definitions(PUBLIC TSD_HAS_METAL)
