// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/TransferRegistry.hpp"
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using tsd::graph::hostResidency;
using tsd::graph::PortType;
using tsd::graph::Residency;
using tsd::graph::TransferRegistry;
using tsd::graph::Value;

SCENARIO("tsd::graph::TransferRegistry lookup and apply", "[graph-transfer]")
{
  PortType arrayT{Token("array")};
  Residency testDev{Token("test"), 0};

  GIVEN("a registry with a host->test transfer for arrays")
  {
    TransferRegistry reg;
    reg.registerTransfer(
        arrayT,
        Token("host"),
        Token("test"),
        [](const Value &src, const Residency &target) {
          Value out = src; // copy metadata
          out.residency = target; // mark moved
          return out;
        },
        [](const Value &src) -> size_t {
          auto b = std::static_pointer_cast<std::vector<float>>(src.payload);
          return b->size() * sizeof(float);
        });

    WHEN("looking up host->test for arrays")
    {
      const auto *e = reg.find(arrayT, Token("host"), Token("test"));
      THEN("it is found")
      {
        REQUIRE(e != nullptr);
      }
      THEN("applying it retags residency and estimates cost")
      {
        auto buf = std::make_shared<std::vector<float>>(4);
        Value src;
        src.type = arrayT;
        src.residency = hostResidency();
        src.payload = buf;
        REQUIRE(e->estimateBytes(src) == 4 * sizeof(float));
        auto moved = e->fn(src, testDev);
        REQUIRE(moved.residency == testDev);
      }
    }

    WHEN("looking up an unregistered direction")
    {
      const auto *e = reg.find(arrayT, Token("test"), Token("host"));
      THEN("it is not found")
      {
        REQUIRE(e == nullptr);
      }
    }
  }
}
