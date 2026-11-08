// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "gpu/mdlTracking.h"

using namespace visrtx;
void require(bool value, const char *why)
{
  if (!value)
    throw std::runtime_error(why);
}
int main()
{
  try {
    const float tiny32 = std::ldexp(1.f, -25), tiny64 = std::ldexp(1.f, -80);
    struct Case
    {
      float a, s, m;
      bool accepted;
    };
    const float maximum = std::numeric_limits<float>::max();
    for (auto c : {Case{0, 0, 0, true},
             Case{.25f, .5f, .75f, true},
             Case{.5f, .5f, 1, true},
             Case{1, tiny32, 1, false},
             Case{1, tiny64, 1, false},
             Case{1, tiny32, std::nextafter(1.f, 2.f), true},
             Case{1, tiny64, std::nextafter(1.f, 2.f), true},
             Case{.25f, .5f, std::nextafter(.75f, 0.f), false},
             Case{maximum, maximum, maximum, false},
             Case{tiny64, 0, tiny64, true}}) {
      for (bool swap : {false, true}) {
        MdlTrackingChannel v;
        require(
            makeMdlTrackingChannel(swap ? c.s : c.a, swap ? c.a : c.s, c.m, v)
                == c.accepted,
            "exact widened bound comparison");
      }
    }
    // Exact rational reference: M=1, a=1/4, s=1/2. Boundaries in
    // integer RNG space are 2^22 and 3*2^22. No production sums/PDFs
    // participate in the expected categorical answer.
    constexpr uint32_t quarter = 1u << 22;
    for (bool swap : {false, true}) {
      MdlTrackingChannel v;
      require(
          makeMdlTrackingChannel(swap ? .5f : .25f, swap ? .25f : .5f, 1, v),
          "valid");
      for (uint32_t k : {0u,
               quarter - 1,
               quarter,
               quarter + 1,
               3 * quarter - 1,
               3 * quarter,
               3 * quarter + 1,
               (1u << 24) - 1}) {
        const auto expected = k < quarter
            ? (swap ? MdlTrackingEvent::SCATTER : MdlTrackingEvent::ABSORB)
            : k < 3 * quarter
            ? (swap ? MdlTrackingEvent::ABSORB : MdlTrackingEvent::SCATTER)
            : MdlTrackingEvent::NULL_EVENT;
        require(
            mdlTrackingEvent(v, 1, k) == expected, "ordered exact boundary");
      }
    }
    for (float M : {std::numeric_limits<float>::denorm_min(),
             .75f,
             1.f,
             std::nextafter(1.f, 2.f),
             maximum})
      for (uint32_t k : {0u, 1u, quarter, quarter + 1, (1u << 24) - 1}) {
        const long double reference =
            static_cast<long double>(k) * M / 16777216.L;
        require(static_cast<long double>(mdlScaledVariate(M, k)) == reference,
            "scaled product must be exactly representable");
      }
    MdlTrackingChannel tie;
    require(makeMdlTrackingChannel(.5f, .5f, 1, tie), "tie accepted");
    require(mdlTrackingEvent(tie, 1, quarter) == MdlTrackingEvent::ABSORB,
        "absorption first on ties");
    require(mdlTrackingEvent(tie, 1, 2 * quarter) == MdlTrackingEvent::SCATTER,
        "tie boundary belongs to second event");
    for (bool swap : {false, true}) {
      MdlTrackingChannel v;
      require(makeMdlTrackingChannel(swap ? tiny64 : 1,
                  swap ? 1 : tiny64,
                  std::nextafter(1.f, 2.f),
                  v),
          "disparate accepted");
      require(v.smaller == double(tiny64), "retain tiny smaller coefficient");
      require(mdlTrackingEvent(v, std::nextafter(1.f, 2.f), 0)
              == (swap ? MdlTrackingEvent::ABSORB : MdlTrackingEvent::SCATTER),
          "zero variate preserves tiny event in either order");
    }
    // Independent exact comparison without a sum: if the larger operand
    // exceeds half M, Sterbenz's lemma makes M-larger exactly representable.
    // Otherwise both operands are <= M/2, which proves the bound directly.
    uint32_t state = 60419;
    auto finiteFloat = [&]() {
      state = 1664525u * state + 1013904223u;
      uint32_t bits = state & 0x7fffffffu;
      if (bits >= 0x7f800000u)
        bits = 0x7f7fffffu;
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    };
    for (int i = 0; i < 10000; ++i) {
      const float a = finiteFloat(), s = finiteFloat(), M = finiteFloat();
      const long double larger = a >= s ? a : s, smaller = a >= s ? s : a;
      const bool expected = larger <= M
          && (larger <= static_cast<long double>(M) / 2
              || smaller <= static_cast<long double>(M) - larger);
      MdlTrackingChannel channel;
      require(makeMdlTrackingChannel(a, s, M, channel) == expected,
          "randomized independent exact bound oracle");
    }
    puts(
        "MDL widened bounds and ordered categories: pass (10000 oracle cases)");
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
