// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::Field;
using float2 = tsd::core::math::float2;

namespace {

struct KnownField : Node // 2x2x1 field with values 0,1,2,3
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("KnownField");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto f = std::make_shared<Field>();
    f->dims = tsd::core::math::uint3(2u, 2u, 1u);
    f->data = tsd::core::AnyArray(ANARI_FLOAT32, 4);
    for (int k = 0; k < 4; ++k)
      f->data.get<float>(k) = float(k);
    Value v;
    v.type = PortType{Token("field")};
    v.residency = hostResidency();
    v.payload = f;
    ctx.setOutput(Token("out"), v);
  }
};

NodeId addBuiltin(Graph &g, const char *type)
{
  static NodeRegistry reg = [] {
    NodeRegistry r;
    tsd::graph_nodes::registerBuiltinNodes(r);
    return r;
  }();
  return g.addNode(reg.create(Token(type)));
}

} // namespace

SCENARIO("ScalarRange computes field min/max", "[nodes-range]")
{
  Graph g;
  auto src = g.addNode(std::make_unique<KnownField>());
  auto sr = addBuiltin(g, "ScalarRange");
  g.connect(src, Token("out"), sr, Token("in"));
  Evaluator e(g);

  WHEN("pulled")
  {
    REQUIRE(e.pull(sr));
    auto out = e.output(sr, Token("out"), hostResidency());
    REQUIRE(out != nullptr);
    auto r = std::static_pointer_cast<float2>(out->payload);
    THEN("the range is {0,3}")
    {
      REQUIRE(r->x == 0.f);
      REQUIRE(r->y == 3.f);
    }
  }
}
