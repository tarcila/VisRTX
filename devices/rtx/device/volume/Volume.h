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
#include "SourceImportance.h"
#include "gpu/aggregateVolumeData.h"

namespace visrtx {

struct Volume : public RegisteredObject<VolumeGPUData>
{
  Volume(DeviceGlobalState *d, VolumeType type = VolumeType::TF1D);
  ~Volume() = default;

  void commitParameters() override;
  void markFinalized() override;

  bool isVisible() const;
  // Intrinsic kind is available even before successful finalization.
  VolumeType volumeType() const;

  // Internal preparation seam; callers flush commits before requesting a view.
  // UNKNOWN means unavailable, not a zero-contribution substitute.
  virtual ReferenceVolumeGPUData referenceVolume();
  ReferenceVolumeGPUData withSourceImportance(ReferenceVolumeGPUData view);

  OptixBuildInput buildInput() const;

  static Volume *createInstance(std::string_view subtype, DeviceGlobalState *d);

 protected:
  VolumeGPUData gpuData() const override;

 private:
  SourceImportance m_sourceImportance;
  mutable DeviceBuffer m_aabbsBuffer;
  mutable CUdeviceptr m_aabbsBufferPtr{};
  uint32_t m_id{~0u};
  bool m_visible{true};
  const VolumeType m_type;
};

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Volume *, ANARI_VOLUME);
