// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <anari/anari_cpp.hpp>
#include <stdexcept>

// Shared public-API failure assertion. No device internals or test-only
// transport controls: a completed failed attempt must publish no channels.
inline void requireFailedMdlFrame(anari::Device d, anari::Frame frame)
{
  int count = -1;
  if (!anariGetProperty(d,
          frame,
          "numSamples",
          ANARI_INT32,
          &count,
          sizeof(count),
          ANARI_WAIT)
      || count != 0)
    throw std::runtime_error("failed accumulation must have zero samples");
  for (const char *channel : {"channel.color",
           "channel.depth",
           "channel.normal",
           "channel.albedo",
           "channel.primitiveId",
           "channel.objectId",
           "channel.instanceId",
           "channel.colorCUDA",
           "channel.depthCUDA",
           "channel.normalCUDA",
           "channel.albedoCUDA",
           "channel.primitiveIdCUDA",
           "channel.objectIdCUDA",
           "channel.instanceIdCUDA"}) {
    unsigned w = 9, h = 9;
    ANARIDataType type = ANARI_FLOAT32;
    if (anariMapFrame(d, frame, channel, &w, &h, &type) || w || h
        || type != ANARI_UNKNOWN)
      throw std::runtime_error("failed channel must be null/zero/UNKNOWN");
  }
}
