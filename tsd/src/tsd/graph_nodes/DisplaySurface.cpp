// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph_nodes/DisplayMask.hpp"

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;

struct DisplaySurface : Node
{
  ParameterList params;
  DisplaySurface()
  {
    params.set(tsd::core::Token("viewportMask"), kDefaultViewportMask);
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("DisplaySurface");
    i.category = Token("sink");
    i.inputs.push_back({Token("in"), PortType{portSurface()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portRenderable()}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto s = std::static_pointer_cast<SurfaceData>(
        ctx.input(Token("in"), hostResidency()).payload);
    if (!s) {
      ctx.fail("DisplaySurface: missing surface input");
      return;
    }
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Surface;
    r->primSubtype = s->geomSubtype;
    r->prim = s->prim;
    r->appearance = s->appearance;

    Value out;
    out.type = PortType{portRenderable()};
    out.residency = hostResidency();
    out.payload = r;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerDisplaySurface(NodeRegistry &reg)
{
  reg.registerType(Token("DisplaySurface"),
      [] { return std::make_unique<DisplaySurface>(); });
}

} // namespace tsd::graph_nodes
