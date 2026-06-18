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

  std::vector<const tsd::scene::Layer *> layersForViewport(int i) const;

  void update();

  anari::World world(int viewport) const;
  int numViewports() const
  {
    return int(m_indices.size());
  }

 private:
  struct Display
  {
    uint64_t mask{0};
    bool enabled{true};
    tsd::scene::Layer *layer{nullptr};
    uint64_t lastVersion{0};
    bool realized{false};
  };

  void rebuildLayer(tsd::graph::NodeId node, Display &d);
  void buildSurface(tsd::scene::Layer *layer, const tsd::graph::Renderable &r);
  void buildVolume(tsd::scene::Layer *layer, const tsd::graph::Renderable &r);
  void applyParams(
      tsd::scene::Object &obj, const tsd::graph::RenderableParams &p);

  tsd::graph::Graph &m_graph;
  tsd::graph::Evaluator &m_eval;
  tsd::core::Token m_deviceName;
  anari::Device m_device;

  tsd::scene::Scene m_renderScene;
  std::map<tsd::graph::NodeId, Display> m_displays;
  std::vector<std::unique_ptr<RenderIndexAllLayers>> m_indices;
};

} // namespace tsd::rendering
