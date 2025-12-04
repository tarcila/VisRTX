## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_nanovdb LANGUAGES CXX)
project_add_library(INTERFACE)
project_include_directories(INTERFACE ${CMAKE_CURRENT_LIST_DIR})
if (TSD_USE_TBB)
  find_package(TBB REQUIRED)
  project_link_libraries(INTERFACE TBB::tbb)
  project_compile_definitions(INTERFACE NANOVDB_USE_TBB)
endif()
