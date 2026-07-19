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

#include "MDL.h"

namespace visrtx {

struct PhysicallyBasedMDL : public MDL
{
  PhysicallyBasedMDL(DeviceGlobalState *d);

  void commitParameters() override;

  bool emissionIsConstant() const override;
  bool emissionIsSampleable() const override;
  vec3 emissionAverage() const override;

  // Opacity Micromap support (ADR 0009): the wrapper's cutout is
  // ResolveAlphaInput(mode, cutoff, opacity * baseColor.alpha) with raw uv0
  // texture lookups, so the spec is bakeable with rawSamplerLookups set.
  MaterialAlphaSpec alphaSpec() const override;
  helium::TimeStamp alphaStateStamp() const override;

 private:
  void translateAndRemoveParameter(std::string_view paramName);

  // Captured by mirroring the parameter translation (see commitParameters):
  // a freshly committed pre-translate `emissive` key determines the binding;
  // the post-translate `emissive.value`/`emissive.texture` keys cover later
  // commits that don't re-set it (no MDL introspection).
  // A nonzero constant is a Geometry Light with an exact Pick Power; a bound
  // sampler is one with the live sampler-mean as Pick Power, the device
  // evaluating the compiled EDF at the synthetic next-event hit (ADR 0006).
  bool m_emissionIsConstant{false};
  vec3 m_emissionRadiance{0.f};
  helium::ChangeObserverPtr<Sampler> m_emissiveSampler;

  // Alpha bindings for the Opacity Micromap bake view, captured by mirroring
  // the parameter translation (same rule as the emissive capture above).
  AlphaMode m_alphaMode{AlphaMode::OPAQUE};
  float m_alphaCutoff{0.5f};
  float m_alphaOpacity{1.f};
  vec4 m_alphaTransmission{0.f};
  helium::ChangeObserverPtr<Sampler> m_alphaColorSampler;
  helium::ChangeObserverPtr<Sampler> m_alphaOpacitySampler;
  helium::ChangeObserverPtr<Sampler> m_alphaTransmissionSampler;
};

} // namespace visrtx
