/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu_debug.h"
#include "gpu_math.h"
#include "gpu_objects.h"
#include "uniformGrid.h"

namespace visrtx {

// Amanatides & Woo 3D grid traversal
template <typename Func>
VISRTX_DEVICE void amanatidesWoo3(Ray ray, ivec3 gridDims, box3 modelBounds, Func func)
{
  const float inf = 1e30f;
  // Compute grid cell size
  const vec3 gridSize = (modelBounds.upper - modelBounds.lower) / vec3(gridDims);
  // Compute initial cell
  ivec3 cell = projectOnGrid(ray.org, gridDims, modelBounds);

  // Compute step and out-of-bounds stop
  ivec3 step, stop;
  vec3 tMax, tDelta;
  for (int axis = 0; axis < 3; ++axis) {
    float dir = (&ray.dir.x)[axis];
    float org = (&ray.org.x)[axis];
    float cellMin = (&modelBounds.lower.x)[axis] + float((&cell.x)[axis]) * (&gridSize.x)[axis];
    float cellMax = cellMin + (&gridSize.x)[axis];
    if (dir > 0.f) {
      step[axis] = 1;
      stop[axis] = (&gridDims.x)[axis];
      tMax[axis] = (cellMax - org) / dir;
      tDelta[axis] = (&gridSize.x)[axis] / dir;
    } else if (dir < 0.f) {
      step[axis] = -1;
      stop[axis] = -1;
      tMax[axis] = (cellMin - org) / dir;
      tDelta[axis] = -(&gridSize.x)[axis] / dir;
    } else {
      step[axis] = 0;
      stop[axis] = (&cell.x)[axis];
      tMax[axis] = inf;
      tDelta[axis] = inf;
    }
  }

  float t = max(ray.t.lower, 0.f);
  float tEnd = ray.t.upper;
  while (cell.x >= 0 && cell.x < gridDims.x &&
         cell.y >= 0 && cell.y < gridDims.y &&
         cell.z >= 0 && cell.z < gridDims.z &&
         t < tEnd) {
    float tNext = min(tMax.x, min(tMax.y, tMax.z));
    float t1 = min(tNext, tEnd);
    if (!func(linearIndex(cell, gridDims), t, t1))
      return;
    // Advance to next cell
    if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
      cell.x += step.x;
      t = tMax.x;
      tMax.x += tDelta.x;
    } else if (tMax.y <= tMax.z) {
      cell.y += step.y;
      t = tMax.y;
      tMax.y += tDelta.y;
    } else {
      cell.z += step.z;
      t = tMax.z;
      tMax.z += tDelta.z;
    }
  }
}

typedef ivec3 GridIterationState;

template <typename Func>
VISRTX_DEVICE void dda3(Ray ray, ivec3 gridDims, box3 modelBounds, Func func)
{
  // Use a large value for infinity
  const float inf = 1e30f;
  const vec3 rcp_dir(
    ray.dir.x != 0.f ? 1.f / ray.dir.x : 0.f,
    ray.dir.y != 0.f ? 1.f / ray.dir.y : 0.f,
    ray.dir.z != 0.f ? 1.f / ray.dir.z : 0.f);

  const vec3 lo = (modelBounds.lower - ray.org) * rcp_dir;
  const vec3 hi = (modelBounds.upper - ray.org) * rcp_dir;

  const vec3 tnear = min(lo, hi);
  const vec3 tfar = max(lo, hi);

  ivec3 cellID = projectOnGrid(ray.org, gridDims, modelBounds);

  // Distance in world space to get from cell to cell
  const vec3 dist(
    ray.dir.x != 0.f ? (tfar.x - tnear.x) / float(gridDims.x) : inf,
    ray.dir.y != 0.f ? (tfar.y - tnear.y) / float(gridDims.y) : inf,
    ray.dir.z != 0.f ? (tfar.z - tnear.z) / float(gridDims.z) : inf);

  // Cell increment: if direction is zero, never step in that axis
  const ivec3 step = {
    ray.dir.x > 0.f ? 1 : (ray.dir.x < 0.f ? -1 : 0),
    ray.dir.y > 0.f ? 1 : (ray.dir.y < 0.f ? -1 : 0),
    ray.dir.z > 0.f ? 1 : (ray.dir.z < 0.f ? -1 : 0)};

  // Stop when we reach grid borders; if direction is zero, never stop in that axis
  const ivec3 stop = {
    ray.dir.x > 0.f ? gridDims.x : (ray.dir.x < 0.f ? -1 : cellID.x),
    ray.dir.y > 0.f ? gridDims.y : (ray.dir.y < 0.f ? -1 : cellID.y),
    ray.dir.z > 0.f ? gridDims.z : (ray.dir.z < 0.f ? -1 : cellID.z)};

  // Increment in world space; if direction is zero, set tnext to +inf so it never wins
  vec3 tnext = {
    ray.dir.x > 0.f ? tnear.x + float(cellID.x + 1) * dist.x :
      (ray.dir.x < 0.f ? tnear.x + float(gridDims.x - cellID.x) * dist.x : inf),
    ray.dir.y > 0.f ? tnear.y + float(cellID.y + 1) * dist.y :
      (ray.dir.y < 0.f ? tnear.y + float(gridDims.y - cellID.y) * dist.y : inf),
    ray.dir.z > 0.f ? tnear.z + float(cellID.z + 1) * dist.z :
      (ray.dir.z < 0.f ? tnear.z + float(gridDims.z - cellID.z) * dist.z : inf)};

  float t0 = max(ray.t.lower, 0.f);

  while (1) { // loop over grid cells
    const float t1 = min(compMin(tnext), ray.t.upper);
    if (debug()) {
      printf("  t1: %f\n", t0);
    }
    if (debug()) printf("DDA cell: (%i,%i,%i), ival: [%f,%f]\n",
                         cellID.x,cellID.y,cellID.z,t0,t1);
    if (!func(linearIndex(cellID, gridDims), t0, t1))
      return;

#if 0
      int axis = arg_min(tnext);
      tnext[axis] += dist[axis];
      cellID[axis] += step[axis];
      if (cellID[axis]==stop[axis]) {
        break;
      }
#else
    const float t_closest = compMin(tnext);
    if (tnext.x == t_closest) {
      tnext.x += dist.x;
      cellID.x += step.x;
      if (cellID.x == stop.x) {
        break;
      }
    }
    if (tnext.y == t_closest) {
      tnext.y += dist.y;
      cellID.y += step.y;
      if (cellID.y == stop.y) {
        break;
      }
    }
    if (tnext.z == t_closest) {
      tnext.z += dist.z;
      cellID.z += step.z;
      if (cellID.z == stop.z) {
        break;
      }
    }
#endif
    t0 = t1;
  }
}

} // namespace visrtx
