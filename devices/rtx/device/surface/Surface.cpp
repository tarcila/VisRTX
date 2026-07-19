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

#include "Surface.h"

#include <algorithm>

namespace visrtx {

Surface::Surface(DeviceGlobalState *d)
    : RegisteredObject<SurfaceGPUData>(ANARI_SURFACE, d)
{
  setRegistry(d->registry.surfaces);
}

void Surface::commitParameters()
{
  m_id = getParam<uint32_t>("id", ~0u);
  m_geometry = getParamObject<Geometry>("geometry");
  m_material = getParamObject<Material>("material");
  m_visible = getParam<bool>("visible", true);
}

void Surface::finalize()
{
  if (!m_material) {
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'material' on ANARISurface");
    return;
  }
  if (!m_geometry) {
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'geometry' on ANARISurface");
    return;
  }
  upload();
}

void Surface::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastSurfaceBLASChange = helium::newTimeStamp();
}

bool Surface::isValid() const
{
  return geometryIsValid() && materialIsValid();
}

Geometry *Surface::geometry()
{
  return m_geometry.ptr;
}

const Geometry *Surface::geometry() const
{
  return m_geometry.ptr;
}

Material *Surface::material()
{
  return m_material.ptr;
}

const Material *Surface::material() const
{
  return m_material.ptr;
}

bool Surface::isVisible() const
{
  return m_visible;
}

bool Surface::isSampleableEmitter() const
{
  return m_material && m_material->emissionIsSampleable() && m_geometry
      && m_geometry->isAreaSamplingSupported();
}

GeometryLight *Surface::ensureGeometryLight()
{
  if (!m_geometryLight) {
    m_geometryLight = new GeometryLight(deviceState());
    // RefCounted starts at PUBLIC=1; the IntrusivePtr owns the only reference we
    // need. Drop the birth PUBLIC ref so clearGeometryLight() actually frees the
    // light and releases its registry.lights slot (mirrors World.cpp's zero
    // group/instance).
    m_geometryLight->refDec(helium::RefType::PUBLIC);
  }
  return m_geometryLight.ptr;
}

void Surface::clearGeometryLight()
{
  m_geometryLight = nullptr;
}

GeometryLight *Surface::geometryLight() const
{
  return m_geometryLight.ptr;
}

OptixBuildInput Surface::buildInput() const
{
  OptixBuildInput obi = {};
  if (geometryIsValid())
    m_geometry->populateBuildInput(obi);

  // Fully Opaque surfaces skip any-hit entirely (shadow rays terminate in
  // traversal). Backface-culling tri/quad geometry still needs the primary
  // any-hit cull, so it keeps any-hit; active cut planes re-enable any-hit
  // per-ray via ENFORCE_ANYHIT instead of per-build.
  if (materialIsValid() && geometryIsValid()) {
    const auto &registry = deviceState()->registry;
    bool disableAnyhit =
        registry.materials.hostValue(m_material->index()).isFullyOpaque;
    if (disableAnyhit) {
      const auto &ggd = registry.geometries.hostValue(m_geometry->index());
      if (ggd.type == GeometryType::TRIANGLE)
        disableAnyhit = !ggd.tri.cullBackfaces;
      else if (ggd.type == GeometryType::QUAD)
        disableAnyhit = !ggd.quad.cullBackfaces;
    }
    if (disableAnyhit) {
      // Keep REQUIRE_SINGLE_ANYHIT_CALL: ENFORCE_ANYHIT (cut planes) turns
      // any-hit back on for exactly these surfaces, and the shadow programs
      // rely on single invocation per primitive.
      constexpr uint32_t flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
          | OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
      m_buildInputFlags[0] = flags;
      if (obi.type == OPTIX_BUILD_INPUT_TYPE_TRIANGLES)
        obi.triangleArray.flags = m_buildInputFlags;
      else if (obi.type == OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES)
        obi.customPrimitiveArray.flags = m_buildInputFlags;
      else if (obi.type == OPTIX_BUILD_INPUT_TYPE_CURVES)
        obi.curveArray.flag = flags;
    }
  }

  if (m_omm && m_omm->attached
      && obi.type == OPTIX_BUILD_INPUT_TYPE_TRIANGLES) {
    auto &om = obi.triangleArray.opacityMicromap;
    om.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
    om.opacityMicromapArray = (CUdeviceptr)m_omm->micromapArray.ptr();
    om.indexBuffer = (CUdeviceptr)m_omm->indexBuffer.ptr();
    om.indexSizeInBytes = sizeof(int32_t);
    om.indexStrideInBytes = 0;
    om.indexOffset = 0;
    om.numMicromapUsageCounts = m_omm->numUsage;
    om.micromapUsageCounts = m_omm->numUsage ? m_omm->usage : nullptr;
  }

  return obi;
}

helium::TimeStamp Surface::lastBLASInputChange() const
{
  auto t = lastFinalized();
  if (m_geometry)
    t = std::max(t, m_geometry->lastFinalized());
  if (m_material)
    t = std::max(t, m_material->alphaStateStamp());
  return t;
}

// Cache key mixes object identities with their finalize stamps: stamps are
// globally monotonic, so an allocator-recycled pointer can never resurrect a
// stale entry.
static uint64_t ommBakeCacheKey(const Geometry *g,
    const Material *m,
    helium::TimeStamp geomStamp,
    helium::TimeStamp alphaStamp,
    helium::TimeStamp configStamp)
{
  auto mix = [](uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
  };
  uint64_t h = 0;
  h = mix(h, uint64_t(reinterpret_cast<uintptr_t>(g)));
  h = mix(h, uint64_t(geomStamp));
  h = mix(h, uint64_t(reinterpret_cast<uintptr_t>(m)));
  h = mix(h, uint64_t(alphaStamp));
  h = mix(h, uint64_t(configStamp));
  return h;
}

bool Surface::ensureOpacityMicromap()
{
  if (!isValid())
    return false;
  auto &omm = deviceState()->omm;
  const auto inputStamp = std::max(lastBLASInputChange(), omm.lastChange);
  if (m_ommBakedAt >= inputStamp)
    return false;

  // Surfaces sharing (geometry, material alpha state) share one bake —
  // including negative verdicts, so a no-win pair is evaluated once.
  const auto key = ommBakeCacheKey(m_geometry.ptr,
      m_material.ptr,
      m_geometry->lastFinalized(),
      m_material->alphaStateStamp(),
      omm.lastChange);
  if (auto it = omm.bakeCache.find(key); it != omm.bakeCache.end()) {
    m_omm = it->second;
    m_ommBakedAt = helium::newTimeStamp();
    return false;
  }

  // Bake only once the inputs have been stable across OMM_SETTLE_PASSES
  // BLAS-rebuild passes: hosts that churn surface/geometry objects would
  // otherwise pay full bakes for micromaps that never get traversed.
  // Deferred surfaces render without OMM; the Group schedules follow-up
  // rebuilds until the settled bakes attach. Epochs (not per-call checks)
  // keep a surface shared by several Groups from "settling" within a pass.
  constexpr uint64_t OMM_SETTLE_PASSES = 2;
  const auto epoch = omm.rebuildEpoch;
  if (m_ommSeenStamp != inputStamp) {
    m_ommSeenStamp = inputStamp;
    m_ommSeenEpoch = epoch;
    m_omm.reset();
    return true;
  }
  if (epoch - m_ommSeenEpoch < OMM_SETTLE_PASSES)
    return true;

  auto baked = std::make_shared<OpacityMicromapBuffers>();
  bakeOpacityMicromaps(*baked, m_geometry.ptr, m_material.ptr, this);
  m_omm = std::move(baked);

  // Prune entries only the cache still references before inserting.
  constexpr size_t OMM_BAKE_CACHE_MAX = 512;
  if (omm.bakeCache.size() >= OMM_BAKE_CACHE_MAX) {
    for (auto it = omm.bakeCache.begin(); it != omm.bakeCache.end();) {
      if (it->second.use_count() == 1)
        it = omm.bakeCache.erase(it);
      else
        ++it;
    }
  }
  omm.bakeCache.emplace(key, m_omm);

  m_ommBakedAt = helium::newTimeStamp();
  return false;
}

bool Surface::geometryIsValid() const
{
  return m_geometry && m_geometry->isValid();
}

bool Surface::materialIsValid() const
{
  return m_material && m_material->isValid();
}

SurfaceGPUData Surface::gpuData() const
{
  SurfaceGPUData retval;
  retval.id = m_id;
  retval.geometry = geometry() ? geometry()->index() : -1;
  retval.material = material() ? material()->index() : -1;
  return retval;
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Surface *);
