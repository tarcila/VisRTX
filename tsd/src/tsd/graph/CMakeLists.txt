project(tsd_graph)

project_add_library(STATIC)

project_sources(PRIVATE
  Version.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(PUBLIC tsd_core)
