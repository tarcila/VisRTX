/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "gpu/shading_api.h"

namespace visrtx {

enum class RayType
{
  PRIMARY
};

DECLARE_FRAME_DATA(frameData)

RT_PROGRAM void __closesthit__()
{
  // no-op
}

RT_PROGRAM void __miss__()
{
  // no-op
}

RT_PROGRAM void __raygen__()
{
  /////////////////////////////////////////////////////////////////////////////
  // TODO: clean this up! need to split out Ray/RNG, don't need screen samples
  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;
  auto ray = makePrimaryRay(ss);
  /////////////////////////////////////////////////////////////////////////////

  accumResults(frameData.fb,
      ss.pixel,
      vec4(ray.dir, 1.f),
      1.f,
      ray.dir,
      -ray.dir,
      ~0u,
      ~0u,
      ~0u);
}

} // namespace visrtx
