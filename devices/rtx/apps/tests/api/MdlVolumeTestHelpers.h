// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "MdlVolumeTestChecks.h"

// Keep the quality fixture's spelling while sharing the landed failure checks.
inline void requireFailedMdlFrame(anari::Device d, anari::Frame frame)
{
  checkFailedMdlFrame(d, frame);
}
