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

#include "RegisteredObject.h"
#include "sampler/Sampler.h"

namespace visrtx {

// OMM bake view of a material's Opacity Function:
//   alpha = colorAlpha.w * opacity.x, post-processed by mode/cutoff.
// bakeable=false when the material cannot express its alpha this way (MDL,
// unknown subtypes) — such materials never get Opacity Micromaps.
struct MaterialAlphaSpec
{
  bool bakeable{false};
  AlphaMode mode{AlphaMode::OPAQUE};
  float cutoff{0.5f};
  MaterialParameter colorAlpha{};
  MaterialParameter opacity{};
  // Not part of alpha itself, but it feeds isFullyOpaque (and thus the
  // per-surface DISABLE_ANYHIT geometry flag), so it must invalidate GASes.
  MaterialParameter transmission{vec4(0.f)};
  // MDL-backed materials sample textures raw: fixed uv0, no ANARI in/out
  // transforms. The bake bounds that function instead of the ANARI sampler's.
  bool rawSamplerLookups{false};
};

bool operator==(const MaterialParameter &a, const MaterialParameter &b);
bool operator==(const MaterialAlphaSpec &a, const MaterialAlphaSpec &b);

struct Material : public RegisteredObject<MaterialGPUData>
{
  Material(DeviceGlobalState *d);
  ~Material() = default;

  static Material *createInstance(
      std::string_view subtype, DeviceGlobalState *d);

  // Emissive Surface support, decided from committed material state alone, never
  // from rendering.
  // - emissionIsConstant: emission is a nonzero constant (not attribute- or
  //   sampler-bound); emissionAverage() IS that constant, exactly, and the NEE
  //   fast-path uses it to skip a per-sample material eval.
  // - emissionIsSampleable: emission is not provably zero (constant, sampler, or
  //   attribute bound with a nonzero average). This is the Geometry Light gate,
  //   broadened past constant emission in Stage 2.
  // - emissionAverage: mean emitted radiance, used only to size Pick Power
  //   (variance, never bias); zero average ⇒ zero pick weight.
  virtual bool emissionIsConstant() const;
  virtual bool emissionIsSampleable() const;
  virtual vec3 emissionAverage() const;

  // Opacity Micromap support (ADR 0009). alphaSpec() describes the Opacity
  // Function for the bake; alphaStateStamp() advances whenever the baked
  // classification could change (spec edits, and for sampler-bound alpha the
  // sampler's own re-finalization) — Group uses it to scope GAS rebuilds to
  // affected surfaces.
  virtual MaterialAlphaSpec alphaSpec() const;
  virtual helium::TimeStamp alphaStateStamp() const;

 protected:
  // Bump the world's light-set timestamp iff this material's Geometry Light
  // eligibility or average radiance changed since the last commit, so emissive
  // edits rebuild the light set while ordinary edits (roughness, color) stay
  // free. Subclasses call this once their emission state is resolved: native
  // PBR at the end of commitParameters(); the MDL family at the end of
  // finalize(), where the compile-time emission classification is known.
  void refreshEmissionLightSet();

  // Bump the BLAS timestamps iff this material's Opacity Function changed, so
  // alpha edits rebake OMMs / rebuild owning GASes while ordinary edits
  // (roughness, color) stay free. Subclasses with a bakeable alpha call this
  // once per commit with their current spec.
  void refreshAlphaState(const MaterialAlphaSpec &spec);

 private:
  bool m_emissionWasSampleable{false};
  vec3 m_lastEmissionAverage{0.f};
  MaterialAlphaSpec m_lastAlphaSpec{};
  helium::TimeStamp m_alphaStamp{0};
  // alphaStateStamp() as of the last refreshAlphaState() — detects bound
  // samplers re-finalizing between commits with an otherwise unchanged spec.
  helium::TimeStamp m_lastAlphaObservedStamp{0};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline MaterialAlphaSpec Material::alphaSpec() const
{
  return {};
}

inline helium::TimeStamp Material::alphaStateStamp() const
{
  return m_alphaStamp;
}

// Inlined helper functions ///////////////////////////////////////////////////

template <typename T>
inline void populateMaterialParameter(
    MaterialParameter &mp, T value, Sampler *sampler, const std::string &attrib)
{
  if (sampler && sampler->isValid()) {
    mp.type = MaterialParameterType::SAMPLER;
    mp.sampler = sampler->index();
  } else if (!attrib.empty()) {
    mp.type = MaterialParameterType::ATTRIBUTE;
    mp.attribute = attributeFromString(attrib);
  } else {
    mp.type = MaterialParameterType::VALUE;
    mp.value = vec4(value);
  }
}

inline AlphaMode alphaModeFromString(const std::string &s)
{
  if (s == "blend")
    return AlphaMode::BLEND;
  else if (s == "mask")
    return AlphaMode::MASK;
  else
    return AlphaMode::OPAQUE;
}

// Inline VALUE with channel ≥ 1 — no sampler/attribute can lower it.
inline bool isStaticOne(const MaterialParameter &mp, int channel = 0)
{
  return mp.type == MaterialParameterType::VALUE && mp.value[channel] >= 1.0f;
}

// Inline VALUE clamped to 0. Used to gate the static-opaque shortcut.
inline bool isStaticZero(const MaterialParameter &mp, int channel = 0)
{
  return mp.type == MaterialParameterType::VALUE && mp.value[channel] <= 0.0f;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Material *, ANARI_MATERIAL);
