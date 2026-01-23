## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

include(FetchContent)

FetchContent_Declare(
  lua
  URL https://www.lua.org/ftp/lua-5.4.8.tar.gz
  URL_HASH SHA256=4f18ddae154e793e46eeab727c59ef1c0c0c2b744e7b94219710d76f530629ae
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(lua)

# Lua source files (core library, no standalone interpreter)
set(LUA_CORE_SOURCES
  ${lua_SOURCE_DIR}/src/lapi.c
  ${lua_SOURCE_DIR}/src/lcode.c
  ${lua_SOURCE_DIR}/src/lctype.c
  ${lua_SOURCE_DIR}/src/ldebug.c
  ${lua_SOURCE_DIR}/src/ldo.c
  ${lua_SOURCE_DIR}/src/ldump.c
  ${lua_SOURCE_DIR}/src/lfunc.c
  ${lua_SOURCE_DIR}/src/lgc.c
  ${lua_SOURCE_DIR}/src/llex.c
  ${lua_SOURCE_DIR}/src/lmem.c
  ${lua_SOURCE_DIR}/src/lobject.c
  ${lua_SOURCE_DIR}/src/lopcodes.c
  ${lua_SOURCE_DIR}/src/lparser.c
  ${lua_SOURCE_DIR}/src/lstate.c
  ${lua_SOURCE_DIR}/src/lstring.c
  ${lua_SOURCE_DIR}/src/ltable.c
  ${lua_SOURCE_DIR}/src/ltm.c
  ${lua_SOURCE_DIR}/src/lundump.c
  ${lua_SOURCE_DIR}/src/lvm.c
  ${lua_SOURCE_DIR}/src/lzio.c
)

# Lua auxiliary library
set(LUA_AUX_SOURCES
  ${lua_SOURCE_DIR}/src/lauxlib.c
  ${lua_SOURCE_DIR}/src/lbaselib.c
  ${lua_SOURCE_DIR}/src/lcorolib.c
  ${lua_SOURCE_DIR}/src/ldblib.c
  ${lua_SOURCE_DIR}/src/liolib.c
  ${lua_SOURCE_DIR}/src/lmathlib.c
  ${lua_SOURCE_DIR}/src/loadlib.c
  ${lua_SOURCE_DIR}/src/loslib.c
  ${lua_SOURCE_DIR}/src/lstrlib.c
  ${lua_SOURCE_DIR}/src/ltablib.c
  ${lua_SOURCE_DIR}/src/lutf8lib.c
  ${lua_SOURCE_DIR}/src/linit.c
)

project(tsd_ext_lua LANGUAGES C)

add_library(${PROJECT_NAME} STATIC
  ${LUA_CORE_SOURCES}
  ${LUA_AUX_SOURCES}
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${lua_SOURCE_DIR}/src>
)

# Platform-specific definitions
if(UNIX AND NOT APPLE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE LUA_USE_LINUX)
  target_link_libraries(${PROJECT_NAME} PRIVATE m dl)
elseif(APPLE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE LUA_USE_MACOSX)
  target_link_libraries(${PROJECT_NAME} PRIVATE m)
elseif(WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE LUA_USE_WINDOWS)
endif()

# Suppress warnings from Lua code
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(${PROJECT_NAME} PRIVATE -w)
elseif(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W0)
endif()
