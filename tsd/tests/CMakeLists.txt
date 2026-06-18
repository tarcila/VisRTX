## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (NOT BUILD_TESTING)
  return()
endif()

project(tsdTests LANGUAGES CXX)
project_add_executable(
  test_App.cpp
  test_AnimationManager.cpp
  test_AnimationArchive.cpp
  test_AnimationManagerArchive.cpp
  test_ArchiveCompatibility.cpp
  test_CameraRendererArchive.cpp
  test_Algorithms.cpp
  test_AnyArray.cpp
  test_Array.cpp
  test_ComponentSerialization.cpp
  test_DataTree.cpp
  test_FlatMap.cpp
  test_Forest.cpp
  test_Geometry.cpp
  test_graph_ConversionRegistry.cpp
  test_graph_Parameter.cpp
  test_graph_PortType.cpp
  test_graph_Residency.cpp
  test_graph_Smoke.cpp
  test_graph_Node.cpp
  test_graph_NodeRegistry.cpp
  test_graph_TransferRegistry.cpp
  test_graph_Value.cpp
  test_graph_GraphLinks.cpp
  test_graph_DeviceKeying.cpp
  test_graph_Dirty.cpp
  test_graph_Evaluator.cpp
  test_graph_ImplicitOps.cpp
  test_graph_NonCacheable.cpp
  test_graph_TestBackendTransfer.cpp
  test_graph_AsyncEval.cpp
  test_graph_AsyncCancel.cpp
  test_graph_AsyncSupersede.cpp
  test_graph_AsyncError.cpp
  test_graph_Renderable.cpp
  test_bridge_Mask.cpp
  test_bridge_RenderSurface.cpp
  test_bridge_RenderVolume.cpp
  test_bridge_Rebuild.cpp
  test_Importers.cpp
  test_LayerSubtreeArchive.cpp
  test_Manipulator.cpp
  test_ObjectPool.cpp
  test_Material.cpp
  test_Math.cpp
  test_Object.cpp
  test_ObjectArchive.cpp
  test_ObjectUsePtr.cpp
  test_Parameter.cpp
  test_Scene.cpp
  test_SceneArchive.cpp
  test_Token.cpp
)
if (TARGET tsd_scivis_studio_model)
  target_sources(${PROJECT_NAME} PRIVATE
    test_ProjectAssetTransaction.cpp
    test_SciVisStudio.cpp
  )
endif()
project_link_libraries(
PRIVATE
  tsd_algorithms
  tsd_animation
  tsd_app
  tsd_graph
  tsd_io
  tsd_rendering
  tsd_scene
  tsd_ext_catch2
)
if (TARGET tsd_scivis_studio_model)
  target_link_libraries(${PROJECT_NAME} PRIVATE tsd_scivis_studio_model)
endif()

add_test(NAME tsd::AnimationManager COMMAND ${PROJECT_NAME} "[AnimationManager]")
add_test(NAME tsd::App COMMAND ${PROJECT_NAME} "[App]")
add_test(NAME tsd::AnimationArchive COMMAND ${PROJECT_NAME} "[AnimationArchive]")
add_test(NAME tsd::AnimationManagerArchive
  COMMAND ${PROJECT_NAME} "[AnimationManagerArchive]")
add_test(NAME tsd::ArchiveCompatibility
  COMMAND ${PROJECT_NAME} "[ArchiveCompatibility]")
add_test(NAME tsd::CameraArchive COMMAND ${PROJECT_NAME} "[CameraArchive]")
add_test(NAME tsd::RendererArchive COMMAND ${PROJECT_NAME} "[RendererArchive]")
add_test(NAME tsd::Algorithms    COMMAND ${PROJECT_NAME} "[Algorithms]"   )
add_test(NAME tsd::AnyArray      COMMAND ${PROJECT_NAME} "[AnyArray]"     )
add_test(NAME tsd::Array         COMMAND ${PROJECT_NAME} "[Array]"        )
add_test(NAME tsd::ComponentSerialization
  COMMAND ${PROJECT_NAME} "[ComponentSerialization]")
add_test(NAME tsd::DataTree      COMMAND ${PROJECT_NAME} "[DataTree]"     )
add_test(NAME tsd::FlatMap       COMMAND ${PROJECT_NAME} "[FlatMap]"      )
add_test(NAME tsd::Forest        COMMAND ${PROJECT_NAME} "[Forest]"       )
add_test(NAME tsd::Geometry      COMMAND ${PROJECT_NAME} "[Geometry]"     )
add_test(NAME tsd::graph::ConversionRegistry COMMAND ${PROJECT_NAME} "[graph-convert]")
add_test(NAME tsd::graph::Parameter COMMAND ${PROJECT_NAME} "[graph-param]")
add_test(NAME tsd::graph::PortType  COMMAND ${PROJECT_NAME} "[graph-porttype]")
add_test(NAME tsd::graph::Residency COMMAND ${PROJECT_NAME} "[graph-residency]")
add_test(NAME tsd::graph::Smoke COMMAND ${PROJECT_NAME} "[graph-smoke]")
add_test(NAME tsd::graph::TransferRegistry COMMAND ${PROJECT_NAME} "[graph-transfer]")
add_test(NAME tsd::graph::Node  COMMAND ${PROJECT_NAME} "[graph-node]")
add_test(NAME tsd::graph::NodeRegistry COMMAND ${PROJECT_NAME} "[graph-noderegistry]")
add_test(NAME tsd::graph::Value COMMAND ${PROJECT_NAME} "[graph-value]")
add_test(NAME tsd::graph::GraphLinks COMMAND ${PROJECT_NAME} "[graph-links]")
add_test(NAME tsd::graph::DeviceKeying COMMAND ${PROJECT_NAME} "[graph-devicekeying]")
add_test(NAME tsd::graph::Dirty COMMAND ${PROJECT_NAME} "[graph-dirty]")
add_test(NAME tsd::graph::Evaluator COMMAND ${PROJECT_NAME} "[graph-eval]")
add_test(NAME tsd::graph::ImplicitOps COMMAND ${PROJECT_NAME} "[graph-implicitops]")
add_test(NAME tsd::graph::NonCacheable COMMAND ${PROJECT_NAME} "[graph-noncache]")
add_test(NAME tsd::graph::TestBackend COMMAND ${PROJECT_NAME} "[graph-testbackend]")
add_test(NAME tsd::graph::Async COMMAND ${PROJECT_NAME} "[graph-async]")
add_test(NAME tsd::graph::AsyncCancel COMMAND ${PROJECT_NAME} "[graph-asynccancel]")
add_test(NAME tsd::graph::AsyncSupersede COMMAND ${PROJECT_NAME} "[graph-supersede]")
add_test(NAME tsd::graph::AsyncError COMMAND ${PROJECT_NAME} "[graph-asyncerror]")
add_test(NAME tsd::graph::Renderable COMMAND ${PROJECT_NAME} "[graph-renderable]")
add_test(NAME tsd::rendering::BridgeMask COMMAND ${PROJECT_NAME} "[bridge-mask]")
set_tests_properties(tsd::rendering::BridgeMask PROPERTIES TIMEOUT 300)
add_test(NAME tsd::rendering::BridgeRenderSurface COMMAND ${PROJECT_NAME} "[bridge-render-surface]")
set_tests_properties(tsd::rendering::BridgeRenderSurface PROPERTIES TIMEOUT 300)
add_test(NAME tsd::rendering::BridgeRenderVolume COMMAND ${PROJECT_NAME} "[bridge-render-volume]")
set_tests_properties(tsd::rendering::BridgeRenderVolume PROPERTIES TIMEOUT 300)
add_test(NAME tsd::rendering::BridgeRebuild COMMAND ${PROJECT_NAME} "[bridge-rebuild]")
set_tests_properties(tsd::rendering::BridgeRebuild PROPERTIES TIMEOUT 300)
add_test(NAME tsd::Importers     COMMAND ${PROJECT_NAME} "[Importers]"    )
add_test(NAME tsd::LayerSubtreeArchive
  COMMAND ${PROJECT_NAME} "[LayerSubtreeArchive]")
add_test(NAME tsd::Manipulator   COMMAND ${PROJECT_NAME} "[Manipulator]"  )
add_test(NAME tsd::ObjectPool    COMMAND ${PROJECT_NAME} "[ObjectPool]"   )
add_test(NAME tsd::Material      COMMAND ${PROJECT_NAME} "[Material]"     )
add_test(NAME tsd::Math          COMMAND ${PROJECT_NAME} "[Math]"         )
add_test(NAME tsd::Object        COMMAND ${PROJECT_NAME} "[Object]"       )
add_test(NAME tsd::ObjectArchive COMMAND ${PROJECT_NAME} "[ObjectArchive]")
add_test(NAME tsd::ObjectUsePtr  COMMAND ${PROJECT_NAME} "[ObjectUsePtr]" )
add_test(NAME tsd::Parameter     COMMAND ${PROJECT_NAME} "[Parameter]"    )
add_test(NAME tsd::Scene         COMMAND ${PROJECT_NAME} "[Scene]"        )
add_test(NAME tsd::SceneArchive  COMMAND ${PROJECT_NAME} "[SceneArchive]" )
add_test(NAME tsd::Token         COMMAND ${PROJECT_NAME} "[Token]"        )
if (TARGET tsd_scivis_studio_model)
  add_test(NAME tsd::SciVisStudio COMMAND ${PROJECT_NAME} "[SciVisStudio]")
endif()
