// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/algorithms/reclassifyAlphaModes.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/Sampler.hpp"

#include <cstdint>
#include <vector>

using namespace tsd::scene;

namespace {

MaterialRef makeBlendMaterial(Scene &scene, const std::vector<uint8_t> &alpha)
{
  auto mat = scene.createObject<Material>(tokens::material::physicallyBased);
  mat->setParameter("alphaMode", "blend");

  auto img = scene.createArray(ANARI_UFIXED8_VEC4, alpha.size());
  std::vector<uint8_t> texels(alpha.size() * 4);
  for (size_t i = 0; i < alpha.size(); i++) {
    texels[i * 4 + 0] = 128;
    texels[i * 4 + 1] = 200;
    texels[i * 4 + 2] = 64;
    texels[i * 4 + 3] = alpha[i];
  }
  img->setData(texels.data());

  auto sampler = scene.createObject<Sampler>(tokens::sampler::image2D);
  sampler->setParameterObject("image", *img);
  // glTF-importer style: route texel alpha into the sampler's x output.
  sampler->setParameter("outTransform",
      tsd::math::mat4({0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 1}));
  mat->setParameterObject("opacity", *sampler);
  return mat;
}

std::string alphaMode(const MaterialRef &m)
{
  return m->parameter("alphaMode")->value().getString();
}

} // namespace

SCENARIO("tsd::scene::reclassifyAlphaModes", "[reclassifyAlphaModes]")
{
  GIVEN("A scene with blend materials of varying alpha content")
  {
    Scene scene;

    // Binary cutout: half fully transparent, half fully opaque.
    std::vector<uint8_t> cutout(64);
    for (size_t i = 0; i < cutout.size(); i++)
      cutout[i] = i < 32 ? 0 : 255;
    auto cutoutMat = makeBlendMaterial(scene, cutout);

    // Fully opaque alpha channel.
    std::vector<uint8_t> opaque(64, 255);
    auto opaqueMat = makeBlendMaterial(scene, opaque);

    // Genuine gradient blend: must stay untouched.
    std::vector<uint8_t> gradient(64);
    for (size_t i = 0; i < gradient.size(); i++)
      gradient[i] = uint8_t(i * 4);
    auto gradientMat = makeBlendMaterial(scene, gradient);

    // Constant-opaque blend (no texture at all).
    auto constMat =
        scene.createObject<Material>(tokens::material::physicallyBased);
    constMat->setParameter("alphaMode", "blend");
    constMat->setParameter("opacity", 1.f);

    WHEN("reclassifyAlphaModes runs")
    {
      const auto result = reclassifyAlphaModes(scene);

      THEN("binary cutout becomes mask with cutoff 0.5")
      {
        REQUIRE(alphaMode(cutoutMat) == "mask");
        REQUIRE(cutoutMat->parameter("alphaCutoff") != nullptr);
        REQUIRE(cutoutMat->parameter("alphaCutoff")->value().get<float>()
            == 0.5f);
      }
      THEN("opaque-textured and constant-opaque blends become opaque")
      {
        REQUIRE(alphaMode(opaqueMat) == "opaque");
        REQUIRE(alphaMode(constMat) == "opaque");
      }
      THEN("a genuine gradient blend is left untouched")
      {
        REQUIRE(alphaMode(gradientMat) == "blend");
      }
      THEN("the result tallies match")
      {
        // >= : the scene's default material is also examined (and, being a
        // constant-opaque blend, rewritten to opaque)
        REQUIRE(result.examined >= 4);
        REQUIRE(result.toMask == 1);
        REQUIRE(result.toOpaque >= 2);
      }
    }
  }
}
