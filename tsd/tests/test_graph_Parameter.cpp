// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Parameter.hpp"

using tsd::core::Token;
using tsd::graph::ParameterList;

SCENARIO("tsd::graph::ParameterList stores values and hashes", "[graph-param]")
{
  GIVEN("a parameter list with two values")
  {
    ParameterList p;
    p.set(Token("iso"), 0.5f);
    p.set(Token("count"), 3);

    THEN("values read back by name")
    {
      REQUIRE(p.get<float>(Token("iso")) == 0.5f);
      REQUIRE(p.get<int>(Token("count")) == 3);
    }
    THEN("a missing param falls back to the default")
    {
      REQUIRE(p.getOr<float>(Token("missing"), -1.0f) == -1.0f);
    }
    THEN("hash is stable for identical content")
    {
      ParameterList q;
      q.set(Token("iso"), 0.5f);
      q.set(Token("count"), 3);
      REQUIRE(p.hash() == q.hash());
    }
    THEN("hash changes when a value changes")
    {
      auto before = p.hash();
      p.set(Token("iso"), 0.6f);
      REQUIRE(p.hash() != before);
    }
  }
}
