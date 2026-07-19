// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/scene/Scene.hpp"

namespace tsd::scene {

// Alpha Classification (opt-in): reclassifies 'blend' materials whose
// effective alpha (baseColor.a x opacity) is provably binary — exporters
// routinely mark cut-out foliage as blend, which defeats binary-visibility
// acceleration downstream (e.g. Opacity Micromaps in device BVHs).
//   alpha ~1 everywhere                              -> "opaque"
//   nearly all texels near {0,1}, some transparent   -> "mask" (cutoff 0.5)
// Materials with genuinely fractional alpha are left untouched. Never run
// implicitly: it rewrites author intent.
struct ReclassifyAlphaResult
{
  size_t examined{0};
  size_t toMask{0};
  size_t toOpaque{0};
};

ReclassifyAlphaResult reclassifyAlphaModes(Scene &scene);

} // namespace tsd::scene
