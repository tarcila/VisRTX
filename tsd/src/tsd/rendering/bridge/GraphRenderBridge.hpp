// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/Renderable.hpp"
#include "tsd/rendering/index/RenderIndexAllLayers.hpp"
#include "tsd/scene/Scene.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tsd::rendering {

class GraphRenderBridge
{
 public:
  GraphRenderBridge(tsd::graph::Graph &graph,
      tsd::graph::Evaluator &eval,
      tsd::core::Token deviceName,
      anari::Device device,
      int numViewports);
  ~GraphRenderBridge();

  GraphRenderBridge(const GraphRenderBridge &) = delete;
  GraphRenderBridge &operator=(const GraphRenderBridge &) = delete;

  void setDisplay(tsd::graph::NodeId node, uint64_t viewportMask, bool enabled);
  void removeDisplay(tsd::graph::NodeId node);
  void setDisplayTransform(tsd::graph::NodeId node, const tsd::math::mat4 &xfm);

  std::vector<const tsd::scene::Layer *> layersForViewport(int i) const;

  void update();

  // Rebuild viewport i's RenderIndex on a new device. No-op if d is already
  // this viewport's device. The render scene is device-agnostic; the index
  // maintains its own per-device ANARI handle cache.
  void setViewportDevice(int i, tsd::core::Token deviceName, anari::Device d);

  anari::World world(int viewport) const;
  int numViewports() const
  {
    return int(m_indices.size());
  }

  // Read access to the generated render scene (for the layer debug panel).
  // The scene is rebuilt each frame from the graph; do not retain references.
  tsd::scene::Scene &renderScene()
  {
    return m_renderScene;
  }

  // Number of render-Scene objects + arrays; for tests/diagnostics.
  size_t renderSceneObjectCount() const;

 private:
  struct Display
  {
    uint64_t mask{0};
    bool enabled{true};
    tsd::scene::Layer *layer{nullptr};
    uint64_t lastVersion{0};
    bool realized{false};
    tsd::math::mat4 transform{tsd::math::IDENTITY_MAT4};
  };

  void addDefaultLight();
  // Layers a viewport's index includes: masked display layers + the light
  // layer.
  std::vector<const tsd::scene::Layer *> indexLayers(int i) const;
  // Human-readable name shared by a display's layer and its render object,
  // matching the graph node's editor title: "<typeName> #<nodeId>".
  std::string displayName(tsd::graph::NodeId node) const;
  void rebuildLayer(tsd::graph::NodeId node, Display &d);
  void clearLayerObjects(tsd::scene::Layer *layer);
  void buildSurface(tsd::scene::Layer *layer,
      const tsd::graph::Renderable &r,
      const std::string &name);
  void buildVolume(tsd::scene::Layer *layer,
      const tsd::graph::Renderable &r,
      const std::string &name);
  void applyParams(
      tsd::scene::Object &obj, const tsd::graph::RenderableParams &p);

  tsd::graph::Graph &m_graph;
  tsd::graph::Evaluator &m_eval;
  tsd::core::Token m_deviceName;
  anari::Device m_device;

  // Authoritative per-viewport device + name. m_device/m_deviceName above are
  // only the ctor defaults (used to seed these and for device-agnostic layer
  // creation); per-viewport device is m_viewportDevices[i].
  std::vector<anari::Device> m_viewportDevices;
  std::vector<tsd::core::Token> m_viewportDeviceNames;

  tsd::scene::Scene m_renderScene;
  tsd::scene::Layer *m_lightsLayer{nullptr}; // default light; in every viewport
  std::map<tsd::graph::NodeId, Display> m_displays;
  std::vector<std::unique_ptr<RenderIndexAllLayers>> m_indices;
};

} // namespace tsd::rendering
