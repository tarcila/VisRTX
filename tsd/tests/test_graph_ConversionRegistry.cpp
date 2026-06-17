// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/ConversionRegistry.hpp"
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using tsd::graph::ConversionRegistry;
using tsd::graph::hostResidency;
using tsd::graph::PortType;
using tsd::graph::Value;

SCENARIO("tsd::graph::ConversionRegistry lookup and apply", "[graph-convert]")
{
  PortType i32{Token("i32array")};
  PortType f32{Token("f32array")};

  GIVEN("a registry with an i32array->f32array conversion")
  {
    ConversionRegistry reg;
    reg.registerConversion(
        i32,
        f32,
        [](const Value &src) {
          auto in = std::static_pointer_cast<std::vector<int>>(src.payload);
          auto out = std::make_shared<std::vector<float>>();
          for (int x : *in)
            out->push_back(static_cast<float>(x));
          Value v = src;
          v.type = PortType{Token("f32array")};
          v.payload = out;
          return v;
        },
        [](const Value &src) -> size_t {
          return std::static_pointer_cast<std::vector<int>>(src.payload)
              ->size();
        });

    WHEN("converting an i32 buffer")
    {
      auto in = std::make_shared<std::vector<int>>(std::vector<int>{1, 2, 3});
      Value src;
      src.type = i32;
      src.residency = hostResidency();
      src.payload = in;

      const auto *e = reg.find(i32, f32);
      THEN("the conversion exists and produces floats")
      {
        REQUIRE(e != nullptr);
        REQUIRE(e->estimateElements(src) == 3);
        auto out = e->fn(src);
        REQUIRE(out.type == f32);
        auto fb = std::static_pointer_cast<std::vector<float>>(out.payload);
        REQUIRE(fb->at(2) == 3.0f);
      }
    }

    WHEN("looking up a missing conversion")
    {
      THEN("it is not found")
      {
        REQUIRE(reg.find(f32, i32) == nullptr);
      }
    }
  }
}
