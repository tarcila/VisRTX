// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __METAL_VERSION__
using metal::clamp;
using metal::exp2;
using metal::log2;
using metal::max;
using metal::min;
using metal::pow;
#else
#include <algorithm>
#include <cmath>
namespace tsd::algorithms::math {
using std::clamp;
using std::exp2;
using std::log2;
using std::max;
using std::min;
using std::pow;
} // namespace tsd::algorithms::math
#endif
