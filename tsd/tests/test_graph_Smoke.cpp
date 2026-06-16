// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Version.hpp"

SCENARIO("tsd::graph links and runs", "[graph-smoke]")
{
  GIVEN("the engine version")
  {
    THEN("it is at least 1")
    {
      REQUIRE(tsd::graph::engineVersion() >= 1);
    }
  }
}
