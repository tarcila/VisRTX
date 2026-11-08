// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cfloat>
#include "gpu_decl.h"

namespace visrtx {
struct MdlTrackingChannel
{
  double extinction;
  double smaller;
  bool absorptionFirst;
};
enum class MdlTrackingEvent
{
  ABSORB,
  SCATTER,
  NULL_EVENT
};

namespace detail {
VISRTX_HOST_DEVICE double mdlAddRn(double a, double b)
{
#ifdef __CUDA_ARCH__
  return __dadd_rn(a, b);
#else
  volatile double result = a + b;
  return result;
#endif
}
VISRTX_HOST_DEVICE double mdlSubRn(double a, double b)
{
#ifdef __CUDA_ARCH__
  return __dsub_rn(a, b);
#else
  volatile double result = a - b;
  return result;
#endif
}
} // namespace detail

VISRTX_HOST_DEVICE bool makeMdlTrackingChannel(
    float a, float s, float majorant, MdlTrackingChannel &channel)
{
  if (!(a >= 0.f && a <= FLT_MAX && s >= 0.f && s <= FLT_MAX && majorant >= 0.f
          && majorant <= FLT_MAX))
    return false;
  const double t = detail::mdlAddRn(double(a), double(s));
  const double z = detail::mdlSubRn(t, double(a));
  const double residual =
      detail::mdlAddRn(detail::mdlSubRn(double(a), detail::mdlSubRn(t, z)),
          detail::mdlSubRn(double(s), z));
  if (t > double(majorant) || (t == double(majorant) && residual > 0.))
    return false;
  channel.extinction = t;
  channel.absorptionFirst = a <= s;
  channel.smaller = double(a <= s ? a : s);
  return true;
}

// k is uniform on [0,2^24). Its product with an FP32 M needs at most
// 48 significant bits, so this FP64 threshold variate is exact.
VISRTX_HOST_DEVICE double mdlScaledVariate(float majorant, uint32_t k)
{
  return (double(k) / 16777216.) * double(majorant);
}
VISRTX_HOST_DEVICE MdlTrackingEvent mdlTrackingEvent(
    const MdlTrackingChannel &channel, float majorant, uint32_t k)
{
  const double z = mdlScaledVariate(majorant, k);
  if (z >= channel.extinction)
    return MdlTrackingEvent::NULL_EVENT;
  const bool first = z < channel.smaller;
  return first == channel.absorptionFirst ? MdlTrackingEvent::ABSORB
                                          : MdlTrackingEvent::SCATTER;
}
} // namespace visrtx
