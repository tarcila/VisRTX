// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/PortType.hpp"

using tsd::graph::PortType;
using tsd::graph::PortTypeRegistry;

SCENARIO("tsd::graph::PortTypeRegistry registration", "[graph-porttype]")
{
  GIVEN("a fresh registry")
  {
    PortTypeRegistry reg;
    WHEN("a type is registered")
    {
      auto field = reg.registerType("spatialField");
      THEN("it reports as registered")
      {
        REQUIRE(reg.isRegistered(tsd::core::Token("spatialField")));
      }
      THEN("re-registering the same name yields an equal PortType")
      {
        auto field2 = reg.registerType("spatialField");
        REQUIRE(field == field2);
      }
      THEN("a different type is not equal")
      {
        auto range = reg.registerType("range");
        REQUIRE(field != range);
      }
    }
    THEN("an unregistered name reports false")
    {
      REQUIRE_FALSE(reg.isRegistered(tsd::core::Token("nope")));
    }
  }
}
