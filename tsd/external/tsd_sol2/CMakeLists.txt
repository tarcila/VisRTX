## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

# Sol3 (sol2 v3.x) - Modern C++ Lua Bindings
# Header-only library, downloads sol3 headers

include(FetchContent)

FetchContent_Declare(
  sol2
  GIT_REPOSITORY https://github.com/ThePhD/sol2.git
  GIT_TAG v3.5.0
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(sol2)

project(tsd_ext_sol2 LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${sol2_SOURCE_DIR}/include>
)

# Link against Lua
target_link_libraries(${PROJECT_NAME} INTERFACE tsd_ext_lua)

# Sol3 configuration
target_compile_definitions(${PROJECT_NAME}
  INTERFACE
    SOL_ALL_SAFETIES_ON=1
    SOL_USING_CXX_LUA=0
)
