/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

// Raw geometry-attribute decoding, shared between the OptiX shading headers
// (evalMaterialParameters.h) and plain-CUDA consumers (OMM bake). Must stay
// free of OptiX device intrinsics.

#include "gpu_objects.h"
#include "gpu_util.h"

#include "utility/AnariTypeHelpers.h"

namespace visrtx {

VISRTX_DEVICE bool isPopulated(const AttributeData &ap)
{
  return ap.numChannels > 0;
}

template <typename T>
VISRTX_DEVICE const T *typedOffset(const void *mem, uint32_t offset)
{
  return ((const T *)mem) + offset;
}

template <typename ELEMENT_T>
VISRTX_DEVICE vec4 getAttributeValue_ufixed(
    const AttributeData &attr, uint32_t offset)
{
  constexpr float m = float(static_cast<ELEMENT_T>(0xFFFFFFFF));
  vec4 retval(0.f, 0.f, 0.f, 1.f);
  switch (attr.numChannels) {
  case 4:
    retval.w =
        *typedOffset<ELEMENT_T>(attr.data, attr.numChannels * offset + 3) / m;
    [[fallthrough]];
  case 3:
    retval.z =
        *typedOffset<ELEMENT_T>(attr.data, attr.numChannels * offset + 2) / m;
    [[fallthrough]];
  case 2:
    retval.y =
        *typedOffset<ELEMENT_T>(attr.data, attr.numChannels * offset + 1) / m;
    [[fallthrough]];
  case 1:
    retval.x =
        *typedOffset<ELEMENT_T>(attr.data, attr.numChannels * offset + 0) / m;
    [[fallthrough]];
  default:
    break;
  }

  return retval;
}

VISRTX_DEVICE vec4 getAttributeValue_f32(
    const AttributeData &attr, uint32_t offset)
{
  switch (attr.numChannels) {
  case 1:
    return vec4(*typedOffset<float>(attr.data, offset), 0.f, 0.f, 1.f);
  case 2:
    return vec4(*typedOffset<vec2>(attr.data, offset), 0.f, 1.f);
  case 3:
    return vec4(*typedOffset<vec3>(attr.data, offset), 1.f);
  case 4:
    return *typedOffset<vec4>(attr.data, offset);
  default:
    break;
  }

  return vec4(0.f, 0.f, 0.f, 1.f);
}

VISRTX_DEVICE vec4 getAttributeValue(
    const AttributeData &attr, uint32_t offset, const vec4 &uniformFallback)
{
  if (attr.data == nullptr || offset == 0xFFFFFFFF)
    return uniformFallback;

  if (isFloat32(attr.type))
    return getAttributeValue_f32(attr, offset);
  else if (isFixed8(attr.type))
    return getAttributeValue_ufixed<uint8_t>(attr, offset);
  else if (isSrgb8(attr.type))
    return convertLinearToSRGB(getAttributeValue_ufixed<uint8_t>(attr, offset));
  else if (isFixed16(attr.type))
    return getAttributeValue_ufixed<uint16_t>(attr, offset);
  else if (isFixed32(attr.type))
    return getAttributeValue_ufixed<uint32_t>(attr, offset);

  return uniformFallback;
}

} // namespace visrtx
