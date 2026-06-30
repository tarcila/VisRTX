project(tsd_graph_nodes)

project_add_library(STATIC)

project_sources(PRIVATE
  BuiltinNodes.cpp
  DemoGraph.cpp
  GraphLayout.cpp
  DisplayMask.cpp
  GraphEditModel.cpp
  GenerateNoiseVolume.cpp
  GenerateGyroid.cpp
  GenerateTurbulence.cpp
  GenerateMetaballs.cpp
  ScalarRange.cpp
  TransferFunction.cpp
  DisplayVolume.cpp
  BoundingBox.cpp
  DisplaySurface.cpp
  DisplayLight.cpp
  DisplayTransform.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(PUBLIC tsd_graph)

# IsosurfaceExtract requires Viskores. Auto-detected and optional: when Viskores
# is absent the node source, its link deps, and its TSD_GRAPH_NODES_HAVE_VISKORES
# guard are all omitted, leaving the rest of the library unchanged.
find_package(Viskores 1.1.0 QUIET)
if(Viskores_FOUND)
  target_sources(tsd_graph_nodes PRIVATE IsosurfaceExtract.cpp CrossSection.cpp)
  target_link_libraries(tsd_graph_nodes PUBLIC viskores::filter_contour viskores::cont)
  target_compile_definitions(tsd_graph_nodes PUBLIC TSD_GRAPH_NODES_HAVE_VISKORES)
endif()
