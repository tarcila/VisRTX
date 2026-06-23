project(tsd_graph_nodes)

project_add_library(STATIC)

project_sources(PRIVATE
  BuiltinNodes.cpp
  DemoGraph.cpp
  DisplayMask.cpp
  GraphEditModel.cpp
  GenerateNoiseVolume.cpp
  ScalarRange.cpp
  TransferFunction.cpp
  DisplayVolume.cpp
  BoundingBox.cpp
  DisplaySurface.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(PUBLIC tsd_graph)
