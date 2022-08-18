/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "array/Array.h"
// std
#include <array>
#include <type_traits>

namespace visrtx {

template <int SIZE>
using texel_t = std::array<uint8_t, SIZE>;
using texel1 = texel_t<1>;
using texel2 = texel_t<2>;
using texel3 = texel_t<3>;
using texel4 = texel_t<4>;

bool isFloat(ANARIDataType format);
int numANARIChannels(ANARIDataType format);
int bytesPerChannel(ANARIDataType format);
int countCudaChannels(const cudaChannelFormatDesc &desc);
cudaTextureAddressMode stringToAddressMode(const std::string &str);

template <int SIZE, typename IN_VEC_T>
inline texel_t<SIZE> makeTexelFromFloat(IN_VEC_T v)
{
  v *= 255;
  texel_t<SIZE> retval;
  auto *in = (float *)&v;
  for (int i = 0; i < SIZE; i++)
    retval[i] = uint8_t(in[i]);
  return retval;
}

template <int IN_NC, typename IN_VEC_T>
inline void transformToStagingBuffer(Array &image, uint8_t *stagingBuffer)
{
  constexpr int NC = IN_NC == 3 ? 4 : IN_NC;
  using texel = texel_t<NC>;

  auto *begin = image.dataAs<IN_VEC_T>();
  auto *end = begin + image.totalSize();
  std::transform(begin, end, (texel *)stagingBuffer, [](IN_VEC_T &v) {
    if constexpr (std::is_same_v<IN_VEC_T, vec3>)
      return makeTexelFromFloat<4>(vec4(v, 1.f));
    else
      return makeTexelFromFloat<NC>(v);
  });
}

} // namespace visrtx