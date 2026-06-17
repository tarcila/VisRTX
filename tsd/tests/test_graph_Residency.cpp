// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Residency.hpp"

using tsd::graph::hostResidency;
using tsd::graph::Residency;

SCENARIO("tsd::graph::Residency equality", "[graph-residency]")
{
  GIVEN("the host residency")
  {
    auto h = hostResidency();
    THEN("its backend is \"host\" and deviceId is -1")
    {
      REQUIRE(h.backend == tsd::core::Token("host"));
      REQUIRE(h.deviceId == -1);
    }
    THEN("two host residencies compare equal")
    {
      REQUIRE(h == hostResidency());
    }
  }

  GIVEN("two cuda residencies on different devices")
  {
    Residency d0{tsd::core::Token("cuda"), 0};
    Residency d1{tsd::core::Token("cuda"), 1};
    THEN("they are not equal")
    {
      REQUIRE(d0 != d1);
    }
  }
}
