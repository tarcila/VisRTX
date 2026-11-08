// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// The default vdf has no directional_bias expression. Only the elemental
// anisotropic_vdf bundle links the scalar evaluator; neither generates a VDF.
#define VISRTX_MDL_VOLUME_BIAS
#include "MDLVolume_ptx.cu"
