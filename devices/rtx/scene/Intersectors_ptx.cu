/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/gpu_math.h"
#include "gpu/shading_api.h"

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

RT_FUNCTION void reportIntersection(float t, const vec3 &normal, float u)
{
  optixReportIntersection(t,
      0,
      bit_cast<uint32_t>(u),
      bit_cast<uint32_t>(normal.x),
      bit_cast<uint32_t>(normal.y),
      bit_cast<uint32_t>(normal.z));
}

RT_FUNCTION void reportIntersection(float t)
{
  optixReportIntersection(t, 0, bit_cast<uint32_t>(0.f));
}

RT_FUNCTION void reportIntersectionVolume(const box1 &t)
{
  const auto rd = optixGetObjectRayDirection();
  optixReportIntersection(t.lower,
      0,
      bit_cast<uint32_t>(t.upper),
      bit_cast<uint32_t>(rd.x),
      bit_cast<uint32_t>(rd.y),
      bit_cast<uint32_t>(rd.z));
}

// Primitive intersectors /////////////////////////////////////////////////////

RT_FUNCTION void intersectSphere(const GeometryGPUData &geometryData)
{
  const auto &sphereData = geometryData.sphere;

  const auto primID =
      sphereData.indices ? sphereData.indices[ray::primID()] : ray::primID();

  const auto center = sphereData.centers[primID];
  const auto radius =
      sphereData.radii ? sphereData.radii[primID] : sphereData.radius;

  const vec3 d = ray::localDirection();
  const float rd2 = 1.f / dot(d, d);
  const vec3 CO = center - ray::localOrigin();
  const float projCO = dot(CO, d) * rd2;
  const vec3 perp = CO - projCO * d;
  const float l2 = glm::dot(perp, perp);
  const float r2 = radius * radius;
  if (l2 > r2)
    return;
  const float td = glm::sqrt((r2 - l2) * rd2);
  reportIntersection(projCO - td);
}

RT_FUNCTION void intersectCylinder(const GeometryGPUData &geometryData)
{
  const auto &cylinderData = geometryData.cylinder;

  const uvec2 pidx = cylinderData.indices ? cylinderData.indices[ray::primID()]
                                          : (2 * ray::primID() + uvec2(0, 1));

  const auto p0 = cylinderData.vertices[pidx.x];
  const auto p1 = cylinderData.vertices[pidx.y];

  const float radius =
      glm::abs(cylinderData.radii ? cylinderData.radii[ray::primID()]
                                  : cylinderData.radius);

  const vec3 ro = ray::localOrigin();
  const vec3 rd = ray::localDirection();

  vec3 ca = p1 - p0;
  vec3 oc = ro - p0;

  float caca = glm::dot(ca, ca);
  float card = glm::dot(ca, rd);
  float caoc = glm::dot(ca, oc);

  float a = caca - card * card;
  float b = caca * glm::dot(oc, rd) - caoc * card;
  float c = caca * glm::dot(oc, oc) - caoc * caoc - radius * radius * caca;
  float h = b * b - a * c;

  if (h < 0.f)
    return;

  h = glm::sqrt(h);
  float d = (-b - h) / a;

  float y = caoc + d * card;
  if (y > 0.f && y < caca) {
    auto n = (oc + d * rd - ca * y / caca) / radius;
    reportIntersection(d, n, position(y, box1(0.f, caca)));
  }

  d = ((y < 0.f ? 0.f : caca) - caoc) / card;

  if (glm::abs(b + a * d) < h) {
    auto n = ca * glm::sign(y) / caca;
    reportIntersection(d, n, y < 0.f ? 0.f : 1.f);
  }
}

RT_FUNCTION void intersectVolume()
{
  auto &hit = ray::rayData<VolumeHit>();
  if (hit.volID == ray::primID() && hit.instID == ray::instID())
    return;

  const auto &ss = ray::screenSample();
  const auto &frameData = *ss.frameData;
  const auto &volumeData = ray::volumeData(frameData);
  const auto &bounds = volumeData.bounds;
  const vec3 mins =
      (bounds.lower - ray::localOrigin()) * (1.f / ray::localDirection());
  const vec3 maxs =
      (bounds.upper - ray::localOrigin()) * (1.f / ray::localDirection());
  const vec3 nears = glm::min(mins, maxs);
  const vec3 fars = glm::max(mins, maxs);

  box1 t(glm::compMax(nears), glm::compMin(fars));

  if (t.lower < t.upper) {
    const box1 rayt{ray::tmin(), ray::tmax()};
    t.lower = clamp(t.lower, rayt);
    t.upper = clamp(t.upper, rayt);
    reportIntersectionVolume(t);
  }
}

// Generic geometry dispatch //////////////////////////////////////////////////

RT_FUNCTION void intersectGeometry()
{
  const auto &ss = ray::screenSample();
  const auto &frameData = *ss.frameData;
  const auto &surfaceData = ray::surfaceData(frameData);
  const auto &geometryData = getGeometryData(frameData, surfaceData.geometry);

  switch (geometryData.type) {
  case GeometryType::SPHERE:
    intersectSphere(geometryData);
    break;
  case GeometryType::CYLINDER:
    intersectCylinder(geometryData);
    break;
  }
}

// Main intersection dispatch /////////////////////////////////////////////////

RT_PROGRAM void __intersection__()
{
  if (ray::isIntersectingSurfaces())
    intersectGeometry();
  else
    intersectVolume();
}

} // namespace visrtx
