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

#include "SDF.h"
// glm
#include <glm/glm.hpp>
// std
#include <algorithm>

static_assert(
    sizeof(visrtx::SDFPrimitive) == 72, "SDFPrimitive layout must be 72 bytes");
static_assert(offsetof(visrtx::SDFPrimitive, neighboursIndex) == 56,
    "SDFPrimitive::neighboursIndex must be at offset 56");

namespace visrtx {

// Compute a conservative AABB for an SDF primitive on the host (CPU).
static box3 computeSDFPrimitiveAABB(const SDFPrimitive &g)
{
  const float disp = g.userParams.x; // displacement amplitude padding
  const float r0 = glm::max(0.f, g.r0);
  const float r1 = glm::max(0.f, g.r1);

  switch (static_cast<SDFType>(g.type)) {
  case SDFType::SPHERE:
  case SDFType::CUT_SPHERE:
    return box3(g.p0 - (r0 + disp), g.p0 + (r0 + disp));

  case SDFType::TORUS:
    return box3(g.p0 - (r0 + r1 + disp), g.p0 + (r0 + r1 + disp));

  case SDFType::ELLIPSOID:
    // p1 stores radii vector for the ellipsoid
    return box3(g.p0 - (g.p1 + disp), g.p0 + (g.p1 + disp));

  case SDFType::PILL:
  case SDFType::CONE_PILL:
  case SDFType::CONE_PILL_SIGMOID:
  case SDFType::CONE:
  case SDFType::VESICA:
  default: {
    const float maxR = glm::max(r0, r1) + disp;
    return box3(glm::min(g.p0, g.p1) - maxR, glm::max(g.p0, g.p1) + maxR);
  }
  }
}

SDF::SDF(DeviceGlobalState *d) : Geometry(d), m_sdf(this), m_neighbour(this) {}

SDF::~SDF() = default;

void SDF::commitParameters()
{
  Geometry::commitParameters();
  m_sdf = getParamObject<Array1D>("primitive.sdf");
  m_neighbour = getParamObject<Array1D>("primitive.neighbor");
  m_epsilon = getParam<float>("epsilon", 1e-5f);
  m_nbMarchIterations = static_cast<uint32_t>(
      glm::max(getParam<int32_t>("nbMarchIterations", 16), 1));
  m_blendFactor = getParam<float>("blendFactor", 1.f);
  m_blendLerpFactor = getParam<float>("blendLerpFactor", 0.5f);
  m_omega = glm::clamp(getParam<float>("omega", 1.f), 0.f, 1.f);
  m_distanceFromCamera = getParam<float>("distanceFromCamera", 100.f);
  m_noiseFactor = glm::clamp(getParam<float>("noiseFactor", 0.f), 0.f, 1.f);
}

void SDF::finalize()
{
  if (!m_sdf) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'primitive.sdf' on sdf geometry");
    return;
  }

  // primitive.sdf must be an ANARI_UINT8 array of N * sizeof(SDFPrimitive)
  // bytes.
  m_numPrimitives = m_sdf->size() / sizeof(SDFPrimitive);

  reportMessage(ANARI_SEVERITY_DEBUG,
      "finalizing sdf geometry with %zu primitives",
      m_numPrimitives);

  const auto *geometries =
      reinterpret_cast<const SDFPrimitive *>(m_sdf->data());

  m_aabbs.resize(m_numPrimitives);
  auto *aabbsHost = m_aabbs.dataHost();
  for (size_t i = 0; i < m_numPrimitives; i++) {
    aabbsHost[i] = computeSDFPrimitiveAABB(geometries[i]);

    // Expand AABB by the smooth-min blend radius so OptiX fires the
    // intersection test for rays passing through the blending zone at
    // junctions between adjacent primitives. Without this, blend zones that
    // lie outside the tight per-primitive AABB produce holes in the geometry.
    if (geometries[i].numNeighbours > 0) {
      const float r0 = glm::max(0.f, geometries[i].r0);
      const float r1 = glm::max(
          0.f, geometries[i].r1 >= 0.f ? geometries[i].r1 : geometries[i].r0);
      const float k =
          glm::mix(glm::min(r0, r1), glm::max(r0, r1), m_blendLerpFactor)
          * m_blendFactor;
      aabbsHost[i].lower -= k;
      aabbsHost[i].upper += k;
    }

    // Expand AABB for noise displacement. The noise can pull the surface
    // outward by up to amplitude = noiseFactor * r0 * 0.1.
    if (m_noiseFactor > 0.f) {
      const float noiseAmplitude =
          m_noiseFactor * glm::max(0.f, geometries[i].r0) * 0.1f;
      aabbsHost[i].lower -= noiseAmplitude;
      aabbsHost[i].upper += noiseAmplitude;
    }
  }

  m_aabbs.upload();
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.dataDevice();

  upload();
}

bool SDF::isValid() const
{
  return bool(m_sdf);
}

void SDF::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  buildInput.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  buildInput.customPrimitiveArray.numPrimitives = m_numPrimitives;

  static uint32_t buildInputFlags[1] = {
      OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
  buildInput.customPrimitiveArray.flags = buildInputFlags;
  buildInput.customPrimitiveArray.numSbtRecords = 1;
}

int SDF::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

GeometryGPUData SDF::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::SDF;

  auto &sdf = retval.sdf;
  sdf.geometries =
      reinterpret_cast<const SDFPrimitive *>(m_sdf->data(AddressSpace::GPU));
  sdf.neighbours =
      m_neighbour ? m_neighbour->beginAs<uint64_t>(AddressSpace::GPU) : nullptr;
  sdf.numGeometries = static_cast<uint32_t>(m_numPrimitives);
  sdf.epsilon = m_epsilon;
  sdf.nbMarchIterations = m_nbMarchIterations;
  sdf.blendFactor = m_blendFactor;
  sdf.blendLerpFactor = m_blendLerpFactor;
  sdf.omega = m_omega;
  sdf.distanceFromCamera = m_distanceFromCamera;
  sdf.noiseFactor = m_noiseFactor;

  return retval;
}

} // namespace visrtx
