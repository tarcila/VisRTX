// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <glm/gtc/matrix_inverse.hpp>
#include <stdexcept>
#include "PreviewShadowCache.h"
#include "PreviewShadowColumns.h"
#include "gpu/renderer/previewFloatLattice.h"
#include "spatial_field/StoredFieldPreviewSampler.h"

namespace visrtx {
namespace {
void checked(cudaError_t error)
{
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
template <typename T>
T readDevice(const T *pointer)
{
  T value;
  checked(cudaMemcpy(&value, pointer, sizeof(value), cudaMemcpyDeviceToHost));
  return value;
}
bool sameBox(const box3 &a, const box3 &b)
{
  return a.lower == b.lower && a.upper == b.upper;
}
bool lessBox(const box3 &a, const box3 &b)
{
  for (int c = 0; c < 3; ++c) {
    if (a.lower[c] != b.lower[c])
      return a.lower[c] < b.lower[c];
    if (a.upper[c] != b.upper[c])
      return a.upper[c] < b.upper[c];
  }
  return false;
}
PreviewShadowCacheKey contentKey(const PrincipledVolumeGPUData &volume,
    const VolumeOccurrenceGPUData &occurrence,
    vec3 direction,
    double rate)
{
  PreviewShadowCacheKey key;
  const auto &b = volume.bindings;
  key.densityRevision = volume.densityRevision;
  key.bounds = b.bounds;
  key.densityDomain = b.density.domain;
  key.densityScale = b.density.scale;
  key.densityOutside = b.density.outside;
  key.direction = direction;
  key.step = occurrence.volume.stepSize;
  key.rate = rate;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 4; ++col)
      key.worldToObject[row][col] = occurrence.worldToObject[row][col];
  // Only internal cuts affect this quadrature. Ignore redundant full-domain
  // roles and canonicalize duplicate cuts independently of binding order.
  for (int role = 0; role < 11; ++role) {
    const auto *domain = preview_volume::bindingDomain(b, role);
    if (!domain || optical_reference::domainContains(*domain, b.bounds))
      continue;
    box3 clipped(glm::max(domain->lower, b.bounds.lower),
        glm::min(domain->upper, b.bounds.upper));
    if (optical_reference::nonemptyDomain(clipped))
      key.cuts[key.cutCount++] = clipped;
  }
  std::sort(key.cuts.begin(), key.cuts.begin() + key.cutCount, lessBox);
  key.cutCount = size_t(
      std::unique(key.cuts.begin(), key.cuts.begin() + key.cutCount, sameBox)
      - key.cuts.begin());
  return key;
}
bool sameContent(const PreviewShadowCacheKey &a, const PreviewShadowCacheKey &b)
{
  if (!(a.densityRevision == b.densityRevision) || !sameBox(a.bounds, b.bounds)
      || !sameBox(a.densityDomain, b.densityDomain)
      || a.densityScale != b.densityScale
      || a.densityOutside != b.densityOutside || a.direction != b.direction
      || a.step != b.step || a.rate != b.rate || a.cutCount != b.cutCount)
    return false;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 4; ++col)
      if (a.worldToObject[row][col] != b.worldToObject[row][col])
        return false;
  for (size_t i = 0; i < a.cutCount; ++i)
    if (!sameBox(a.cuts[i], b.cuts[i]))
      return false;
  return true;
}
VISRTX_GLOBAL void buildPreviewShadow(float *output,
    uint32_t n,
    AggregateVolumeSceneGPUData scene,
    mat4 objectToWorld,
    vec3 direction,
    float rate)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = size_t(n) * n * n;
  if (i >= count)
    return;
  const auto &o = scene.occurrences[0];
  const auto &volume = *o.volume.principled;
  const auto &b = volume.bindings;
  const vec3 index(
      float(i % n), float((i / n) % n), float(i / (size_t(n) * n)));
  const vec3 objectPoint = b.bounds.lower
      + (index / float(n - 1)) * (b.bounds.upper - b.bounds.lower);
  const vec3 point = vec3(objectToWorld * vec4(objectPoint, 1.f));
  preview_volume::Ray ray;
  for (int c = 0; c < 3; ++c) {
    ray.origin[c] = point[c];
    ray.direction[c] = direction[c];
  }
  ray.upper = INFINITY;
  const auto prepared = practical_preview::prepareRay(scene, ray, rate);
  float integral = 0.f, cursor = 0.f;
  preview_volume::Interval span;
  while (preview_volume::nextSpan(scene, ray, cursor, span)) {
    PreviewDensityIntegral query;
    for (int c = 0; c < 3; ++c) {
      query.origin[c] = prepared.local.origin[c];
      query.direction[c] = prepared.local.direction[c];
    }
    query.lower = span.lower;
    query.upper = span.upper;
    query.anchor = prepared.full.lower;
    query.step = prepared.step;
    query.volumeDomain = b.bounds;
    query.fieldDomain = b.density.domain;
    query.scale = b.density.scale;
    query.outside = b.density.outside;
    query.allowTexture = false; // comparison builder retains software filtering
    integral +=
        preview_math::densityIntegral(volume.fields[b.density.field], query);
  }
  output[i] = integral;
}
} // namespace
PreviewShadowCache::PreviewShadowCache(
    uint32_t resolution, PreviewShadowBuilder builder)
    : m_resolution(resolution), m_builder(builder)
{
  if (resolution < 2 || resolution > 257)
    throw std::runtime_error("invalid internal shadow cache resolution");
}
PreviewShadowCache::~PreviewShadowCache()
{
  cudaFree(m_values);
}
uint64_t PreviewShadowCache::buildCount() const
{
  return m_builds;
}
PreviewShadowCacheGPUData PreviewShadowCache::prepare(
    const AggregateVolumeSceneGPUData &scene,
    const FrameGPUData &frame,
    uint64_t lightStamp,
    cudaStream_t stream)
{
  // Higher shadow rates bypass this fixed spatial cache rather than impose
  // an interpolation error floor on refinement. No public rate cap is added.
  if (!scene.complete || !scene.previewFloatEligible || scene.count != 1
      || frame.world.numLightInstances != 1 || frame.referenceShadowRate > 1.
      || size_t(frame.fb.size.x) * frame.fb.size.y < 4096) {
    m_data = {};
    m_attempted = false;
    m_contentValid = false;
    return m_data;
  }
  if (m_attempted && m_generation == scene.generation
      && m_lightStamp == lightStamp && m_rate == frame.referenceShadowRate)
    return m_data;
  const bool previousValid = m_contentValid;
  m_contentValid = false;
  m_data = {};
  m_generation = scene.generation;
  m_lightStamp = lightStamp;
  m_rate = frame.referenceShadowRate;
  m_attempted = true;
  checked(cudaStreamSynchronize(stream));
  const auto occurrence = readDevice(scene.occurrences);
  if (occurrence.volume.type != VolumeType::PRINCIPLED)
    return m_data;
  const auto volume = readDevice(occurrence.volume.principled);
  if (!volume.previewShadowCacheEligible)
    return m_data;
  const auto instance = readDevice(frame.world.lightInstances);
  const auto light = readDevice(frame.registry.lights + instance.lightIndex);
  if (light.type != LightType::DIRECTIONAL)
    return m_data;
  glm::dmat4 inverseWide(1.);
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 4; ++col)
      inverseWide[col][row] = occurrence.worldToObject[row][col];
  const mat4 inverse(inverseWide);
  const mat4 objectToWorld(glm::affineInverse(inverseWide));
  for (int corner = 0; corner < 8; ++corner) {
    vec3 p;
    for (int c = 0; c < 3; ++c)
      p[c] = (corner & (1 << c)) ? volume.bindings.bounds.upper[c]
                                 : volume.bindings.bounds.lower[c];
    p = vec3(objectToWorld * vec4(p, 1.f));
    for (int c = 0; c < 3; ++c)
      if (std::abs(p[c]) > scene.previewOriginLimit)
        return m_data;
  }
  const vec3 direction =
      normalize(vec3(instance.xfm * vec4(-light.distant.direction, 0.f)));
  if (!m_values) {
    const auto error = cudaMalloc(&m_values,
        size_t(m_resolution) * m_resolution * m_resolution * sizeof(float));
    if (error == cudaErrorMemoryAllocation) {
      cudaGetLastError(); // Optional optimization: retain the marcher on OOM.
      return m_data;
    }
    checked(error);
  }
  vec3 cacheLower = volume.bindings.bounds.lower,
       cacheUpper = volume.bindings.bounds.upper;
  mat4 cacheToWorld = objectToWorld, worldToCache = inverse;
  if (m_builder == PreviewShadowBuilder::Columns) {
    const vec3 up = std::abs(direction.z) < .9f ? vec3(0, 0, 1) : vec3(0, 1, 0);
    const vec3 x = normalize(cross(up, direction));
    const vec3 y = cross(direction, x);
    const vec3 center =
        vec3(objectToWorld * vec4((cacheLower + cacheUpper) * .5f, 1.f));
    cacheToWorld = mat4(
        vec4(x, 0.f), vec4(y, 0.f), vec4(direction, 0.f), vec4(center, 1.f));
    worldToCache = mat4(glm::affineInverse(glm::dmat4(cacheToWorld)));
    cacheLower = vec3(INFINITY);
    cacheUpper = vec3(-INFINITY);
    for (int corner = 0; corner < 8; ++corner) {
      vec3 p;
      for (int c = 0; c < 3; ++c)
        p[c] = (corner & (1 << c)) ? volume.bindings.bounds.upper[c]
                                   : volume.bindings.bounds.lower[c];
      p = vec3(worldToCache * objectToWorld * vec4(p, 1.f));
      cacheLower = glm::min(cacheLower, p);
      cacheUpper = glm::max(cacheUpper, p);
    }
  }
  const auto key = contentKey(volume, occurrence, direction, m_rate);
  if (!previousValid || !sameContent(key, m_key)) {
    const size_t count = size_t(m_resolution) * m_resolution * m_resolution;
    if (m_builder == PreviewShadowBuilder::Columns)
      preview_columns::buildPreviewShadowColumns<<<
          (size_t(m_resolution) * m_resolution + 127) / 128,
          128,
          0,
          stream>>>(m_values,
          m_resolution,
          scene,
          cacheToWorld,
          cacheLower,
          cacheUpper,
          float(m_rate));
    else
      buildPreviewShadow<<<(count + 127) / 128, 128, 0, stream>>>(m_values,
          m_resolution,
          scene,
          objectToWorld,
          direction,
          float(m_rate));
    checked(cudaGetLastError());
    ++m_builds;
    m_key = key;
  }
  m_contentValid = true;
  m_data.density = m_values;
  m_data.resolution = m_resolution;
  for (int c = 0; c < 3; ++c) {
    m_data.domainLower[c] = cacheLower[c];
    m_data.domainUpper[c] = cacheUpper[c];
    for (int k = 0; k < 4; ++k)
      m_data.worldToObject[c][k] = worldToCache[k][c];
  }
  for (int c = 0; c < 3; ++c)
    m_data.extinctionPerDensity[c] = float(volume.constantTransmissionLog[c])
        / volume.bindings.transmissionDistance.scale;
  return m_data;
}
} // namespace visrtx
