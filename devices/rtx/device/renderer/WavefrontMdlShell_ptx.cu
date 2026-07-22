/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Wavefront MDL shade shell (ticket 10, slice 1 — infrastructure). This is
// compiled to RELOCATABLE PTX at build time; at MDL material commit the device
// nvJitLinks this shell against that material's PTX blob (which defines mdlInit
// and the mdlBsdf_* / mdlEmission_* entries) plus the texture runtime, producing
// one loadable cubin per material. This first slice keeps the kernel minimal —
// it only references the MDL init entry so the per-material link must resolve a
// real MDL symbol. The full State / argBlock setup and BSDF/emission evaluation
// (modelled on MDLShader_ptx.cu) land in the next slice.

// The MDL material PTX exports these as extern "C" .visible .func symbols; the
// exact signature does not matter for the link (name resolution) at this stage.
extern "C" __device__ void mdlInit(
    void *state, const void *res, const char *argBlock);

extern "C" __global__ void wavefrontMdlShade(
    unsigned int liveSlots, int *sentinel)
{
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= liveSlots)
    return;

  // Force a reference to the MDL init entry so nvJitLink must resolve it against
  // the per-material PTX (never taken at runtime).
  if (i == 0xffffffffu)
    mdlInit(nullptr, nullptr, nullptr);

  if (i == 0 && sentinel)
    *sentinel = 1;
}
