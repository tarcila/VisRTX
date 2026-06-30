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

  // Per-viewport camera headlight. mode On/Off force it; Auto enables it only
  // when no plain light is masked to that viewport.
  struct HeadlightState
  {
    enum class Mode
    {
      Auto,
      On,
      Off
    } mode{Mode::Auto};
    tsd::math::float3 direction{0.f, 0.f, -1.f}; // light travel dir (world)
    tsd::math::float3 color{1.f, 1.f, 1.f};
    float irradiance{1.f};
  };
  void setViewportHeadlight(int viewport, const HeadlightState &s);
  // Whether viewport i's headlight is currently injected (for tests + UI).
  bool headlightActive(int viewport) const;

 private:
  struct Display
  {
    uint64_t mask{0};
    bool enabled{true};
    tsd::scene::Layer *layer{nullptr};
    uint64_t lastVersion{0};
    bool realized{false};
    bool isLight{
        false}; // last build emitted a light (headlight auto uses this)
    tsd::math::mat4 transform{tsd::math::IDENTITY_MAT4};
  };

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
  void buildLight(tsd::scene::Layer *layer,
      const tsd::graph::Renderable &r,
      const std::string &name);
  void applyParams(
      tsd::scene::Object &obj, const tsd::graph::RenderableParams &p);

  bool viewportHasPlainLight(int i) const;
  void releaseHeadlight(int i);

  struct Headlight
  {
    anari::Light light{nullptr};
    anari::Instance instance{nullptr};
    bool active{false}; // currently injected via setExternalInstances
  };
  std::vector<Headlight> m_headlights; // sized to numViewports

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
  std::map<tsd::graph::NodeId, Display> m_displays;
  std::vector<std::unique_ptr<RenderIndexAllLayers>> m_indices;
};

} // namespace tsd::rendering
