// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "reclassifyAlphaModes.hpp"

#include "tsd/scene/objects/Array.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/Sampler.hpp"

#include <cstdint>
#include <optional>

namespace tsd::scene {

// One 8-bit step of slack on either end: content-pipeline round-tripping
// rarely preserves exact 0/1 alpha.
constexpr float ALPHA_TRANSPARENT_MAX = 1.5f / 255.f;
constexpr float ALPHA_OPAQUE_MIN = 1.f - 1.5f / 255.f;
// Fraction of texels that must be (near-)binary before rewriting blend to
// mask; the remainder is edge anti-aliasing the mask cutoff resolves.
constexpr float BINARY_TEXEL_FRACTION = 0.99f;
constexpr float RECLASSIFY_MASK_CUTOFF = 0.5f;

namespace {

struct AlphaFactor
{
  float constant{1.f}; // when sampler == nullptr
  Sampler *sampler{nullptr};
  int channel{0}; // effective channel after the sampler's out-transform
};

// Decode one texel to float4 with the ANARI fill convention (missing color
// channels -> 0, missing alpha -> 1). Alpha of sRGB formats is linear, and
// this heuristic never needs color-exactness, so raw normalization suffices.
bool decodeTexel(
    const Array &a, anari::DataType type, size_t i, math::float4 &out)
{
  auto fixed = [&](const uint8_t *base, int nc, float scale) {
    out = math::float4(0.f, 0.f, 0.f, 1.f);
    for (int c = 0; c < nc; c++)
      out[c] = float(base[i * nc + c]) * scale;
  };
  switch (type) {
  case ANARI_FLOAT32:
    out = math::float4(a.dataAs<float>()[i], 0.f, 0.f, 1.f);
    return true;
  case ANARI_FLOAT32_VEC2: {
    auto v = a.dataAs<math::float2>()[i];
    out = math::float4(v.x, v.y, 0.f, 1.f);
    return true;
  }
  case ANARI_FLOAT32_VEC3: {
    auto v = a.dataAs<math::float3>()[i];
    out = math::float4(v, 1.f);
    return true;
  }
  case ANARI_FLOAT32_VEC4:
    out = a.dataAs<math::float4>()[i];
    return true;
  case ANARI_UFIXED8:
    fixed(a.dataAs<uint8_t>(), 1, 1.f / 255.f);
    return true;
  case ANARI_UFIXED8_VEC2:
    fixed(a.dataAs<uint8_t>(), 2, 1.f / 255.f);
    return true;
  case ANARI_UFIXED8_VEC3:
  case ANARI_UFIXED8_RGB_SRGB:
    fixed(a.dataAs<uint8_t>(), 3, 1.f / 255.f);
    return true;
  case ANARI_UFIXED8_VEC4:
  case ANARI_UFIXED8_RGBA_SRGB:
    fixed(a.dataAs<uint8_t>(), 4, 1.f / 255.f);
    return true;
  default:
    return false;
  }
}

// Alpha-relevant view of one material parameter; nullopt when the parameter
// cannot be conservatively evaluated (attribute-driven, exotic sampler).
std::optional<AlphaFactor> resolveFactor(
    Material &m, Token name, int channel)
{
  AlphaFactor f;
  f.channel = channel;

  if (auto *sampler = m.parameterValueAsObject<Sampler>(name); sampler) {
    if (sampler->subtype() != tokens::sampler::image2D)
      return std::nullopt;
    // A zero out-transform row makes the channel texel-independent (the glTF
    // importer builds baseColor samplers this way: alpha forced to the out
    // offset) — treat as the constant it is.
    math::mat4 outTransform = math::IDENTITY_MAT4;
    math::float4 outOffset(0.f);
    if (auto v = sampler->parameterValueAs<math::mat4>("outTransform"); v)
      outTransform = *v;
    if (auto v = sampler->parameterValueAs<math::float4>("outOffset"); v)
      outOffset = *v;
    bool rowIsZero = true;
    for (int j = 0; j < 4; j++)
      rowIsZero &= outTransform[j][channel] == 0.f;
    if (rowIsZero) {
      f.constant = outOffset[channel];
    } else {
      f.sampler = sampler;
    }
    return f;
  }

  const auto *p = m.parameter(name);
  if (!p)
    return f; // parameter default: alpha contribution 1
  const auto &v = p->value();
  if (v.holdsObject())
    return std::nullopt; // non-sampler object (attribute string, etc.)
  if (v.type() == ANARI_FLOAT32)
    f.constant = v.get<float>();
  else if (v.type() == ANARI_FLOAT32_VEC4)
    f.constant = v.get<math::float4>()[channel];
  else if (v.type() == ANARI_FLOAT32_VEC3)
    f.constant = 1.f; // no alpha channel
  else if (v.type() == ANARI_STRING)
    return std::nullopt; // attribute-driven
  else
    return std::nullopt;
  return f;
}

// Effective alpha of one texel through the sampler's affine out-transform.
float sampleAlpha(const Array &img,
    anari::DataType type,
    size_t i,
    int channel,
    const math::mat4 &outTransform,
    const math::float4 &outOffset)
{
  math::float4 t(0.f, 0.f, 0.f, 1.f);
  decodeTexel(img, type, i, t);
  return math::mul(outTransform, t)[channel] + outOffset[channel];
}

} // namespace

ReclassifyAlphaResult reclassifyAlphaModes(Scene &scene)
{
  ReclassifyAlphaResult result;
  const bool debug = getenv("TSD_RECLASSIFY_DEBUG") != nullptr;

  // numberOfObjects() counts LIVE materials, but getObject() indexes pool
  // SLOTS (erased materials leave holes) — walk slots until every live
  // material has been visited.
  const size_t numMaterials = scene.numberOfObjects(ANARI_MATERIAL);
  for (size_t i = 0, seen = 0; seen < numMaterials; i++) {
    auto m = scene.getObject<Material>(i);
    if (!m)
      continue;
    seen++;
    const bool isPBR = m->subtype() == tokens::material::physicallyBased
        || m->subtype() == tokens::material::physicallyBasedMDL;
    const bool isMatte = m->subtype() == tokens::material::matte;
    if (!isPBR && !isMatte)
      continue;

    const auto *modeParam = m->parameter("alphaMode");
    if (!modeParam || modeParam->value().type() != ANARI_STRING
        || modeParam->value().getString() != "blend")
      continue;

    result.examined++;

    const Token colorName = isPBR ? Token("baseColor") : Token("color");
    auto colorAlpha = resolveFactor(*m, colorName, 3);
    auto opacity = resolveFactor(*m, Token("opacity"), 0);
    if (!colorAlpha || !opacity) {
      if (debug)
        printf("[reclassify] '%s': unresolvable factor (color=%d opacity=%d)\n",
            m->name().c_str(), int(bool(colorAlpha)), int(bool(opacity)));
      continue;
    }
    // At most one texture may drive alpha; two independent grids have no
    // common texel domain to histogram.
    if (colorAlpha->sampler && opacity->sampler)
      continue;

    const AlphaFactor &tex = colorAlpha->sampler ? *colorAlpha : *opacity;
    const float constFactor =
        colorAlpha->sampler ? opacity->constant : colorAlpha->constant;

    if (!tex.sampler) {
      // Fully constant alpha: only the (mis-exported) opaque case is safe to
      // rewrite; fractional constants are genuine blends.
      if (colorAlpha->constant * opacity->constant >= ALPHA_OPAQUE_MIN) {
        m->setParameter("alphaMode", "opaque");
        result.toOpaque++;
      }
      continue;
    }

    auto *img = tex.sampler->parameterValueAsObject<Array>("image");
    if (!img || img->size() == 0) {
      if (debug)
        printf("[reclassify] '%s': no image array\n", m->name().c_str());
      continue;
    }
    if (!img->isHost()) {
      // decodeTexel dereferences the array's data pointer on the CPU.
      if (debug)
        printf("[reclassify] '%s': device-memory image, skipping\n",
            m->name().c_str());
      continue;
    }
    const auto type = img->elementType();
    {
      math::float4 probe;
      if (!decodeTexel(*img, type, 0, probe)) {
        if (debug)
          printf("[reclassify] '%s': unsupported texel type %d\n",
              m->name().c_str(), int(type));
        continue; // unsupported texel format
      }
    }

    math::mat4 outTransform = math::IDENTITY_MAT4;
    math::float4 outOffset(0.f);
    if (auto v = tex.sampler->parameterValueAs<math::mat4>("outTransform"); v)
      outTransform = *v;
    if (auto v = tex.sampler->parameterValueAs<math::float4>("outOffset"); v)
      outOffset = *v;

    // Full-texture CPU scan (per-texel out-transform): an opt-in import-time
    // pass, not a render-loop cost. The histogram judges texels only — a
    // clampToBorder sampler with a transparent border can still render
    // transparent at uv edges after an all-opaque rewrite; acceptable for an
    // opt-in heuristic that already rewrites author intent.
    const size_t n = img->size();
    size_t nTransparent = 0, nOpaque = 0;
    for (size_t t = 0; t < n; t++) {
      const float a = constFactor
          * sampleAlpha(*img, type, t, tex.channel, outTransform, outOffset);
      if (a <= ALPHA_TRANSPARENT_MAX)
        nTransparent++;
      else if (a >= ALPHA_OPAQUE_MIN)
        nOpaque++;
    }

    if (nOpaque == n) {
      m->setParameter("alphaMode", "opaque");
      result.toOpaque++;
    } else if (nTransparent > 0
        && float(nTransparent + nOpaque) >= BINARY_TEXEL_FRACTION * float(n)) {
      m->setParameter("alphaMode", "mask");
      m->setParameter("alphaCutoff", RECLASSIFY_MASK_CUTOFF);
      result.toMask++;
    }
  }

  return result;
}

} // namespace tsd::scene
