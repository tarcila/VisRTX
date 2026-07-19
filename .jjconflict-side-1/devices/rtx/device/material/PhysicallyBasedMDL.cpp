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

#include "PhysicallyBasedMDL.h"

#include <anari/anari_cpp/Traits.h>
#include <anari/frontend/anari_enums.h>

#include <string_view>

using namespace std::string_view_literals;

namespace visrtx {

PhysicallyBasedMDL::PhysicallyBasedMDL(DeviceGlobalState *d)
    : MDL(d),
      m_emissiveSampler(this),
      m_alphaColorSampler(this),
      m_alphaOpacitySampler(this),
      m_alphaTransmissionSampler(this)
{
  setParam("source",
      ANARI_STRING,
      "::visrtx::physically_based::physically_based_material");
}

void PhysicallyBasedMDL::translateAndRemoveParameter(std::string_view name)
{
  std::string nameS(name);
  auto asAny = getParamDirect(nameS);
  switch (asAny.type()) {
  case ANARI_FLOAT32:
  case ANARI_FLOAT32_VEC2:
  case ANARI_FLOAT32_VEC3:
  case ANARI_FLOAT32_VEC4: {
    setParamDirect(nameS + ".value", asAny);
    removeParam(nameS + ".texture");
    break;
  }
  case ANARI_SAMPLER:
    setParamDirect(nameS + ".texture", asAny);
    removeParam(nameS + ".value");
    break;
  default:
    // Do nothing...
    return;
  }

  removeParam(nameS);
}

void PhysicallyBasedMDL::commitParameters()
{
  // Limitation: attribute-bound inputs are not supported by the MDL wrapper;
  // only .value / .texture bindings are translated below.

  // Translate all supported parameters to their matching .value or .texture if they are
  // variant inputs.
  translateAndRemoveParameter("opacity"sv);
  translateAndRemoveParameter("baseColor"sv);
  translateAndRemoveParameter("metallic"sv);
  translateAndRemoveParameter("roughness"sv);
  translateAndRemoveParameter("normal"sv);
  translateAndRemoveParameter("emissive"sv);
  translateAndRemoveParameter("occlusion"sv);
  translateAndRemoveParameter("specular"sv);
  translateAndRemoveParameter("specularColor"sv);
  translateAndRemoveParameter("clearcoat"sv);
  translateAndRemoveParameter("clearcoatRoughness"sv);
  translateAndRemoveParameter("clearcoatNormal"sv);
  translateAndRemoveParameter("transmission"sv);
  translateAndRemoveParameter("thickness"sv);
  translateAndRemoveParameter("sheenColor"sv);
  translateAndRemoveParameter("sheenRoughness"sv);
  translateAndRemoveParameter("iridescence"sv);
  translateAndRemoveParameter("iridescenceThickness"sv);

  // Capture the emissive binding by mirroring the translation above. Under
  // helium snapshot-commit, getters read the snapshot taken at
  // anariCommitParameters() while setParam*() writes staging — so the
  // post-translate keys the translation just wrote are INVISIBLE to getters
  // until the next flush. A freshly set pre-translate `emissive` (visible in
  // this flush's snapshot) therefore fully determines the binding; only when
  // absent do the post-translate keys persisted by earlier flushes apply —
  // which also keeps the capture alive on later commits that don't re-set
  // `emissive` (the translation consumed the pre-translate key).
  // Constant iff a nonzero inline color is bound; a bound sampler goes through
  // the EDF path with a live sampler-mean Pick Power (see emissionAverage).
  // Note: unsetting `emissive` leaves the last translated key in place — a
  // pre-existing translation-lifecycle trait shared by the MDL argument block,
  // so the light stays consistent with the still-glowing surface.
  {
    vec3 radiance(0.f);
    if (auto any = getParamDirect("emissive"); any.type() != ANARI_UNKNOWN) {
      m_emissiveSampler =
          any.type() == ANARI_SAMPLER ? any.getObject<Sampler>() : nullptr;
      if (any.type() == ANARI_FLOAT32_VEC3)
        radiance = any.get<vec3>();
      else if (any.type() == ANARI_FLOAT32_VEC4)
        radiance = vec3(any.get<vec4>());
    } else {
      m_emissiveSampler = getParamObject<Sampler>("emissive.texture");
      if (!m_emissiveSampler) {
        vec4 v(0.f);
        getParam("emissive.value", ANARI_FLOAT32_VEC4, &v);
        getParam("emissive.value", ANARI_FLOAT32_VEC3, &v);
        radiance = vec3(v);
      }
    }
    m_emissionRadiance = radiance;
    m_emissionIsConstant = glm::any(glm::greaterThan(radiance, vec3(0.f)));
  }

  // Purposefully exclude the following from the above translation, as they are
  // raw parameters and are already set correctly.
  // alphaMode: enum
  // alphaCutoff: float
  // ior: float
  // attenuationDistance: float
  // attenuationColor: color
  // iridescenceIor: float

  // Translate alphaMode to its matching integer value, capturing the decoded
  // mode on the way: under helium snapshot-commit the setParam below writes
  // STAGING, so a getParam("alphaMode") in this same flush would still read
  // the pre-translate string (or nothing) from the committed snapshot.
  if (auto alphaModeAny = getParamDirect("alphaMode");
      alphaModeAny.type() == ANARI_STRING) {
    m_alphaMode = alphaModeFromString(alphaModeAny.getString());
    setParam("alphaMode", static_cast<int>(m_alphaMode));
  } else {
    // Not freshly set as a string: an int from the app, or the int this
    // translation persisted in an earlier flush.
    m_alphaMode = static_cast<AlphaMode>(
        getParam<int>("alphaMode", int(AlphaMode::OPAQUE)));
  }

  // Capture the alpha bindings for the Opacity Micromap bake view. Getters
  // read the committed snapshot while the translation above wrote staging, so
  // the post-translate keys it just produced are INVISIBLE until the next
  // flush. Mirror the translation instead: a pre-translate key freshly set by
  // the app (visible in this flush's snapshot) fully determines the binding;
  // only when absent do the post-translate keys persisted by earlier flushes
  // apply.
  auto captureAlphaBinding = [&](const char *name,
                                 const char *textureKey,
                                 const char *valueKey,
                                 float defaultValue) {
    struct
    {
      Sampler *sampler{nullptr};
      float value;
    } b{nullptr, defaultValue};
    if (auto any = getParamDirect(name); any.type() != ANARI_UNKNOWN) {
      if (any.type() == ANARI_SAMPLER)
        b.sampler = any.getObject<Sampler>();
      else if (any.type() == ANARI_FLOAT32)
        b.value = any.get<float>();
      // vec3/vec4 color values carry no alpha for the bake view — keep the
      // default, matching the wrapper's texture-over-value precedence.
      return b;
    }
    if (auto *s = getParamObject<Sampler>(textureKey); s)
      b.sampler = s;
    else if (valueKey)
      b.value = getParam<float>(valueKey, defaultValue);
    return b;
  };

  m_alphaCutoff = getParam<float>("alphaCutoff", 0.5f);
  m_alphaColorSampler =
      captureAlphaBinding("baseColor", "baseColor.texture", nullptr, 1.f)
          .sampler;
  const auto opacity =
      captureAlphaBinding("opacity", "opacity.texture", "opacity.value", 1.f);
  m_alphaOpacitySampler = opacity.sampler;
  m_alphaOpacity = opacity.value;
  const auto transmission = captureAlphaBinding(
      "transmission", "transmission.texture", "transmission.value", 0.f);
  m_alphaTransmissionSampler = transmission.sampler;
  m_alphaTransmission = vec4(transmission.value);
  refreshAlphaState(alphaSpec());

  MDL::commitParameters();
  // The light-set refresh runs from MDL::finalize (after the emission
  // classification is resolved), not here.
}

bool PhysicallyBasedMDL::emissionIsConstant() const
{
  return m_emissionIsConstant;
}

bool PhysicallyBasedMDL::emissionIsSampleable() const
{
  return glm::any(glm::greaterThan(emissionAverage(), vec3(0.f)));
}

vec3 PhysicallyBasedMDL::emissionAverage() const
{
  // Sampler-bound: LIVE mean texel (the sampler may finalize after this
  // material's commit — capturing the mean at commit would read a stale or
  // default value). Variance-only Pick Power either way; the compiled EDF at
  // the synthetic hit supplies the true per-point radiance, exactly as on the
  // path-hit deposit. An all-black texture is a zero mean -> not sampleable,
  // matching native PBR.
  if (m_emissiveSampler && m_emissiveSampler->isValid())
    return vec3(m_emissiveSampler->averageValue());
  return m_emissionRadiance;
}

MaterialAlphaSpec PhysicallyBasedMDL::alphaSpec() const
{
  MaterialAlphaSpec spec;
  spec.bakeable = true;
  spec.rawSamplerLookups = true;
  spec.mode = m_alphaMode;
  spec.cutoff = m_alphaCutoff;

  // The wrapper's ResolveBaseColorInput: texture bound -> alpha = lookup.w,
  // otherwise constant 1 (a plain color has no alpha).
  if (m_alphaColorSampler.get() && m_alphaColorSampler->isValid()) {
    spec.colorAlpha.type = MaterialParameterType::SAMPLER;
    spec.colorAlpha.sampler = m_alphaColorSampler->index();
  } else {
    spec.colorAlpha = MaterialParameter(vec4(1.f));
  }

  if (m_alphaOpacitySampler.get() && m_alphaOpacitySampler->isValid()) {
    spec.opacity.type = MaterialParameterType::SAMPLER;
    spec.opacity.sampler = m_alphaOpacitySampler->index();
  } else {
    spec.opacity = MaterialParameter(vec4(m_alphaOpacity));
  }

  if (m_alphaTransmissionSampler.get()
      && m_alphaTransmissionSampler->isValid()) {
    spec.transmission.type = MaterialParameterType::SAMPLER;
    spec.transmission.sampler = m_alphaTransmissionSampler->index();
  } else {
    spec.transmission = MaterialParameter(m_alphaTransmission);
  }

  return spec;
}

helium::TimeStamp PhysicallyBasedMDL::alphaStateStamp() const
{
  auto t = Material::alphaStateStamp();
  if (m_alphaColorSampler.get())
    t = std::max(t, m_alphaColorSampler->lastFinalized());
  if (m_alphaOpacitySampler.get())
    t = std::max(t, m_alphaOpacitySampler->lastFinalized());
  if (m_alphaTransmissionSampler.get())
    t = std::max(t, m_alphaTransmissionSampler->lastFinalized());
  return t;
}

} // namespace visrtx
