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

// SDF distance functions /////////////////////////////////////////////////////

VISRTX_DEVICE float sdfSphere(const vec3 &p, const SDFPrimitive &g)
{
  return glm::length(p - g.p0) - g.r0;
}

VISRTX_DEVICE float sdfPill(const vec3 &p, const SDFPrimitive &g)
{
  const vec3 ba = g.p1 - g.p0;
  const float h =
      glm::clamp(glm::dot(p - g.p0, ba) / glm::dot(ba, ba), 0.f, 1.f);
  return glm::length(p - g.p0 - h * ba) - g.r0;
}

VISRTX_DEVICE float sdfConePill(const vec3 &p, const SDFPrimitive &g)
{
  // Inigo Quilez's rounded cone (r0 >= r1, enforced by host factory functions)
  const vec3 ba = g.p1 - g.p0;
  const float l2 = glm::dot(ba, ba);
  const float rr = g.r0 - g.r1;
  const float a2 = l2 - rr * rr;
  const float il2 = 1.f / l2;
  const vec3 pa = p - g.p0;
  const float y = glm::dot(pa, ba);
  const float z = y - l2;
  const vec3 xv = pa * l2 - ba * y;
  const float x2 = glm::dot(xv, xv);
  const float y2 = y * y * l2;
  const float z2 = z * z * l2;
  const float k = glm::sign(rr) * rr * rr * x2;
  if (glm::sign(z) * a2 * z2 > k)
    return glm::sqrt(x2 + z2) * il2 - g.r1;
  if (glm::sign(y) * a2 * y2 < k)
    return glm::sqrt(x2 + y2) * il2 - g.r0;
  return (glm::sqrt(x2 * a2 * il2) + y * rr) * il2 - g.r0;
}

VISRTX_DEVICE float sdfCappedCone(const vec3 &p, const SDFPrimitive &g)
{
  // Inigo Quilez's capped/truncated cone
  const vec3 bav = g.p1 - g.p0;
  const float rba = g.r1 - g.r0;
  const float baba = glm::dot(bav, bav);
  const float papa = glm::dot(p - g.p0, p - g.p0);
  const float paba = glm::dot(p - g.p0, bav) / baba;
  const float x = glm::sqrt(glm::max(0.f, papa - paba * paba * baba));
  const float cax = glm::max(0.f, x - ((paba < 0.5f) ? g.r0 : g.r1));
  const float cay = glm::abs(paba - 0.5f) - 0.5f;
  const float k = rba * rba + baba;
  const float f = glm::clamp((rba * (x - g.r0) + paba * baba) / k, 0.f, 1.f);
  const float cbx = x - g.r0 - f * rba;
  const float cby = paba - f;
  const float s = (cbx < 0.f && cay < 0.f) ? -1.f : 1.f;
  return s
      * glm::sqrt(
          glm::min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}

VISRTX_DEVICE float sdfTorus(const vec3 &p, const SDFPrimitive &g)
{
  // p0=center, r0=major radius, r1=minor radius; torus lies in the XZ plane
  const vec3 c = g.p0;
  const vec2 q =
      vec2(glm::length(vec2(p.x - c.x, p.z - c.z)) - g.r0, p.y - c.y);
  return glm::length(q) - g.r1;
}

VISRTX_DEVICE float sdfCutSphere(const vec3 &p, const SDFPrimitive &g)
{
  // Inigo Quilez's cut sphere: center p0, radius r0, cut plane at height r1
  const vec3 c = g.p0;
  const float h = g.r1;
  const float w = glm::sqrt(glm::max(0.f, g.r0 * g.r0 - h * h));
  const vec2 q = vec2(glm::length(vec2(p.x - c.x, p.z - c.z)), p.y - c.y);
  const float s =
      glm::max((h - g.r0) * q.x * q.x + w * w * (h + g.r0 - 2.f * q.y),
          h * q.x - w * q.y);
  return (s < 0.f) ? glm::length(q) - g.r0
      : (q.x < w)  ? h - q.y
                   : glm::length(q - vec2(w, h));
}

VISRTX_DEVICE float sdfVesica(const vec3 &p, const SDFPrimitive &g)
{
  // Inigo Quilez's 3D vesica segment: endpoints p0, p1, width r0
  const vec3 mid = 0.5f * (g.p0 + g.p1);
  const float l = glm::length(g.p1 - g.p0);
  const vec3 v = (g.p1 - g.p0) / l;
  const float y = glm::dot(p - mid, v);
  const vec2 q = vec2(glm::length(p - mid - y * v), glm::abs(y));
  const float r = 0.5f * l;
  const float d = 0.5f * (r * r - g.r0 * g.r0) / g.r0;
  const vec3 h =
      (r * q.x < d * (q.y - r)) ? vec3(0.f, r, 0.f) : vec3(-d, 0.f, d + g.r0);
  return glm::length(q - vec2(h.x, h.y)) - h.z;
}

VISRTX_DEVICE float sdfEllipsoid(const vec3 &p, const SDFPrimitive &g)
{
  // Approximate ellipsoid SDF; p0=center, p1=radii vector
  const vec3 r = g.p1;
  const float k0 = glm::length((p - g.p0) / r);
  const float k1 = glm::length((p - g.p0) / (r * r));
  return k0 * (k0 - 1.f) / k1;
}

// Evaluate the SDF distance for a single primitive (no blending).
VISRTX_DEVICE float sdfPrimitiveDist(const vec3 &p, const SDFPrimitive &g)
{
  switch (static_cast<SDFType>(g.type)) {
  case SDFType::SPHERE:
    return sdfSphere(p, g);
  case SDFType::PILL:
    return sdfPill(p, g);
  case SDFType::CONE_PILL:
  case SDFType::CONE_PILL_SIGMOID:
    return sdfConePill(p, g);
  case SDFType::CONE:
    return sdfCappedCone(p, g);
  case SDFType::TORUS:
    return sdfTorus(p, g);
  case SDFType::CUT_SPHERE:
    return sdfCutSphere(p, g);
  case SDFType::VESICA:
    return sdfVesica(p, g);
  case SDFType::ELLIPSOID:
    return sdfEllipsoid(p, g);
  default:
    return glm::length(p - g.p0) - g.r0;
  }
}

// Procedural displacement: organic membrane-like surface detail.
VISRTX_DEVICE float sdfDisplacement(const vec3 &p, const vec3 &userParams)
{
  const float A = userParams.x;
  const float f = userParams.y;
  return A
      * (0.7f * glm::sin(f * p.x * 0.72f) * glm::sin(f * p.y * 0.65f)
              * glm::sin(f * p.z * 0.81f)
          + 0.3f * glm::cos(p.x * 2.12f) * glm::cos(p.y * 2.23f)
              * glm::cos(p.z * 2.41f));
}

// Polynomial smooth-min for blending two SDF values.
VISRTX_DEVICE float sminPoly(float a, float b, float k)
{
  const float h = glm::clamp(0.5f + 0.5f * (b - a) / k, 0.f, 1.f);
  return glm::mix(b, a, h) - k * h * (1.f - h);
}

// Compute AABB for an SDF primitive on the GPU (mirrors host-side logic).
// padding expands the box uniformly on all sides to cover blend zones.
VISRTX_DEVICE box3 computeSDFAABB(const SDFPrimitive &g, float padding = 0.f)
{
  const float disp = g.userParams.x + padding;
  const float r0 = glm::max(0.f, g.r0);
  const float r1 = glm::max(0.f, g.r1);
  switch (static_cast<SDFType>(g.type)) {
  case SDFType::SPHERE:
  case SDFType::CUT_SPHERE:
    return box3(g.p0 - (r0 + disp), g.p0 + (r0 + disp));
  case SDFType::TORUS:
    return box3(g.p0 - (r0 + r1 + disp), g.p0 + (r0 + r1 + disp));
  case SDFType::ELLIPSOID:
    return box3(g.p0 - (g.p1 + disp), g.p0 + (g.p1 + disp));
  default: {
    const float maxR = glm::max(r0, r1) + disp;
    return box3(glm::min(g.p0, g.p1) - maxR, glm::max(g.p0, g.p1) + maxR);
  }
  }
}

// ---------------------------------------------------------------------------
// Smooth 3D value noise — hash lattice with trilinear smoothstep blending.
// Returns a value in [-1, 1].
// ---------------------------------------------------------------------------
VISRTX_DEVICE float _sdfHash(vec3 p)
{
  p = glm::fract(p * vec3(0.1031f, 0.1030f, 0.0973f));
  p += glm::dot(p, vec3(p.y + 33.33f, p.z + 33.33f, p.x + 33.33f));
  return glm::fract((p.x + p.y) * p.z) * 2.f - 1.f;
}

VISRTX_DEVICE float sdfNoise3D(const vec3 &p)
{
  const vec3 i = glm::floor(p);
  const vec3 f = glm::fract(p);
  const vec3 u = f * f * (3.f - 2.f * f); // smoothstep

  return glm::mix(
      glm::mix(
          glm::mix(
              _sdfHash(i + vec3(0, 0, 0)), _sdfHash(i + vec3(1, 0, 0)), u.x),
          glm::mix(
              _sdfHash(i + vec3(0, 1, 0)), _sdfHash(i + vec3(1, 1, 0)), u.x),
          u.y),
      glm::mix(
          glm::mix(
              _sdfHash(i + vec3(0, 0, 1)), _sdfHash(i + vec3(1, 0, 1)), u.x),
          glm::mix(
              _sdfHash(i + vec3(0, 1, 1)), _sdfHash(i + vec3(1, 1, 1)), u.x),
          u.y),
      u.z);
}

// Two-octave fBm for richer organic texture.
VISRTX_DEVICE float sdfONoise(const vec3 &p)
{
  return 0.6f * sdfNoise3D(p)
      + 0.4f * sdfNoise3D(p * 2.3f + vec3(1.7f, 9.2f, 3.5f));
}

// Full SDF evaluation with optional neighbour blending and displacement.
// Neighbours are evaluated without further blending (one level only).
VISRTX_DEVICE float sdfEval(
    const vec3 &p, uint32_t primIdx, const SDFGeometryData &data, float depth)
{
  const SDFPrimitive &g = data.geometries[primIdx];
  float d = sdfPrimitiveDist(p, g);

  if (g.userParams.x > 0.f && depth < data.distanceFromCamera)
    d += sdfDisplacement(p, g.userParams);

  if (g.numNeighbours > 0 && data.neighbours != nullptr) {
    const float rMin = glm::min(g.r0, g.r1 >= 0.f ? g.r1 : g.r0);
    const float rMax = glm::max(g.r0, g.r1 >= 0.f ? g.r1 : g.r0);
    const float k =
        glm::mix(rMin, rMax, data.blendLerpFactor) * data.blendFactor;
    for (uint8_t i = 0; i < g.numNeighbours; i++) {
      const uint64_t ni = data.neighbours[g.neighboursIndex + i];
      if (ni >= data.numGeometries)
        continue;
      const float nd = sdfPrimitiveDist(p, data.geometries[ni]);
      // Always blend even when nd < 0 (inside the neighbor). Skipping negative
      // neighbors caused a discontinuity in the SDF at the neighbor's surface
      // boundary, making the sphere tracer miss the blended junction surface.
      d = sminPoly(nd, d, k);
    }
  }

  // Organic noise displacement: amplitude = noiseFactor * 10% of r0,
  // frequency tuned so ~one bump spans the primitive radius.
  if (data.noiseFactor > 0.f) {
    const float r0 = glm::max(g.r0, 1e-3f);
    const float amplitude = data.noiseFactor * r0 * 0.1f;
    const float frequency = 2.f / r0;
    d += amplitude * sdfONoise(p * frequency);
  }

  return d;
}

VISRTX_DEVICE void intersectSDF(const GeometryGPUData &geometryData)
{
  const auto &sdfData = geometryData.sdf;
  const uint32_t primIdx = ray::primID();
  if (primIdx >= sdfData.numGeometries)
    return;

  const vec3 ro = ray::localOrigin();
  const vec3 rd = ray::localDirection();

  // Expand the sphere-tracing AABB by the blend radius and noise amplitude
  // so the tracer covers the full extent of the perturbed surface.
  float padding = 0.f;
  {
    const SDFPrimitive &g = sdfData.geometries[primIdx];
    if (g.numNeighbours > 0) {
      const float r0 = glm::max(0.f, g.r0);
      const float r1 = glm::max(0.f, g.r1 >= 0.f ? g.r1 : g.r0);
      padding +=
          glm::mix(glm::min(r0, r1), glm::max(r0, r1), sdfData.blendLerpFactor)
          * sdfData.blendFactor;
    }
    if (sdfData.noiseFactor > 0.f)
      padding += sdfData.noiseFactor * glm::max(0.f, g.r0) * 0.1f;
  }
  const box3 bounds = computeSDFAABB(sdfData.geometries[primIdx], padding);
  float t0, t1;
  if (!rayBoxIntersection(ro, rd, bounds, t0, t1))
    return;
  t0 = glm::max(t0, optixGetRayTmin());
  t1 = glm::min(t1, optixGetRayTmax());
  if (t0 >= t1)
    return;

  // Stable camera-to-primitive distance in WORLD space: p0/p1 are stored in
  // world space, so we must use the world-space ray origin here.
  // optixGetObjectRayOrigin() would be in object space which differs from
  // world space when the geometry is referenced through a transformed instance,
  // causing camDist to be wrong. World-space origin is always correct.
  const SDFPrimitive &prim0 = sdfData.geometries[primIdx];
  const vec3 primCentre =
      prim0.r1 >= 0.f ? 0.5f * (prim0.p0 + prim0.p1) : prim0.p0;
  const vec3 roWorld = make_vec3(optixGetWorldRayOrigin());
  const float camDist = glm::length(roWorld - primCentre);

  // Lipschitz correction: noise displacement adds a gradient whose magnitude
  // depends on amplitude * frequency * fBm gain. With two octaves (gains 0.6
  // and 0.4 at 2.3x frequency), the worst-case directional derivative is
  // approximately noiseFactor * 0.2 * (0.6 + 0.4*2.3) * smoothstep_peak
  // ≈ noiseFactor * 0.5. Dividing each step by (1 + G) keeps sphere tracing
  // conservative and prevents overshooting through the perturbed surface.
  const float lipschitzCorr = 1.f / (1.f + sdfData.noiseFactor * 0.5f);

  // All sdfEval calls use camDist as the LOD depth so distanceFromCamera
  // compares against a consistent, stable value.
  const float sdfSign =
      glm::sign(sdfEval(ro + t0 * rd, primIdx, sdfData, camDist));

  float t = t0;
  float stepLength = 0.f;
  float candidateT = t0;
  float bestError = 1e9f;

  for (uint32_t i = 0; i < sdfData.nbMarchIterations; i++) {
    const vec3 p = ro + t * rd;
    const float radius =
        glm::abs(sdfSign * sdfEval(p, primIdx, sdfData, camDist));

    // Scale step by Lipschitz correction so noise-induced gradient > 1 can't
    // cause the tracer to skip past the surface. Clamp to epsilon to prevent
    // stalling when sdfEval goes negative deep inside a blended neighbor.
    stepLength =
        glm::max(radius * sdfData.omega * lipschitzCorr, sdfData.epsilon);

    if (t > 0.f) {
      const float error = radius / t;
      if (error < bestError) {
        bestError = error;
        candidateT = t;
      }
      if (bestError < sdfData.epsilon)
        break;
    }
    if (t > t1)
      break;

    t += stepLength;
    if (t < t0)
      t = t0;
  }

  if (candidateT > t1 || bestError > sdfData.epsilon * 100.f)
    return;

  // Normal estimation via the tetrahedron technique (4 SDF evaluations).
  // Use 1% of the primitive's primary radius as the finite-difference offset
  // so the gradient is well-conditioned across all scene scales.
  const vec3 hp = ro + candidateT * rd;
  const float primScale = glm::max(sdfData.geometries[primIdx].r0, 1e-3f);
  const float e = primScale * 0.01f;
  const vec3 k0(1.f, -1.f, -1.f);
  const vec3 k1(-1.f, -1.f, 1.f);
  const vec3 k2(-1.f, 1.f, -1.f);
  const vec3 k3(1.f, 1.f, 1.f);
  const vec3 normal =
      glm::normalize(k0 * sdfEval(hp + e * k0, primIdx, sdfData, camDist)
          + k1 * sdfEval(hp + e * k1, primIdx, sdfData, camDist)
          + k2 * sdfEval(hp + e * k2, primIdx, sdfData, camDist)
          + k3 * sdfEval(hp + e * k3, primIdx, sdfData, camDist));

  reportIntersection(candidateT, normal, 0.f);
}
