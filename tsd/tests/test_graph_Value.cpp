// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Value.hpp"
// std
#include <memory>
#include <vector>

using tsd::graph::hostResidency;
using tsd::graph::PortType;
using tsd::graph::Residency;
using tsd::graph::Value;

SCENARIO("tsd::graph::Value holds a typed, residency-tagged payload",
    "[graph-value]")
{
  GIVEN("a default Value")
  {
    Value v;
    THEN("it is invalid")
    {
      REQUIRE_FALSE(v.valid());
    }
  }

  GIVEN("a Value wrapping a host float buffer")
  {
    auto buf =
        std::make_shared<std::vector<float>>(std::vector<float>{1, 2, 3});
    Value v;
    v.type = PortType{tsd::core::Token("array")};
    v.residency = hostResidency();
    v.payload = buf;
    v.producerNodeId = 7;
    v.version = 42;

    THEN("it is valid and exposes its payload")
    {
      REQUIRE(v.valid());
      auto out = std::static_pointer_cast<std::vector<float>>(v.payload);
      REQUIRE(out->at(1) == 2.0f);
      REQUIRE(v.version == 42);
      REQUIRE(v.residency == hostResidency());
    }
    THEN("contentTag is unset by default")
    {
      REQUIRE_FALSE(v.contentTag.has_value());
    }
  }
}
