// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "frame/PreviewShadowCache.h"
#include "gpu/previewShadowCache.h"
#include "spatial_field/StoredFieldPreviewSampler.h"
#include "utility/DeviceBuffer.h"
using namespace visrtx;
static void require(bool value, const char *message)
{
  if (!value)
    throw std::runtime_error(message);
}
static void checked(cudaError_t e)
{
  require(e == cudaSuccess, cudaGetErrorString(e));
}
VISRTX_GLOBAL void sampleCache(
    PreviewShadowCacheGPUData cache, vec3 p, vec3 *out)
{
  if (!lookupPreviewShadow(cache, p, 0, *out))
    *out = vec3(-1.f);
}
VISRTX_GLOBAL void sampleDensity(
    StoredFieldReferenceGPUData field, PreviewDensityIntegral query, vec3 *out)
{
  out->x = preview_math::densityIntegral(field, query);
  out->z = preview_math::textureDensityIntegral(field, query);
  query.allowTexture = false;
  out->y = preview_math::densityIntegral(field, query);
}
static void run(PreviewShadowBuilder builder)
{
  std::vector<float> values(32 * 32 * 32);
  for (int z = 0; z < 32; ++z)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x)
        values[x + 32 * (y + 32 * z)] = 1.f + float(x) / 31.f;
  DeviceBuffer density, fieldBuffer, volumeBuffer, occurrenceBuffer,
      lightBuffer, instanceBuffer, output;
  density.upload(values);
  StoredFieldReferenceGPUData field;
  field.data = density.ptr();
  field.indexMax = ivec3(31);
  field.spacing = vec3(1.f / 31.f);
  PreviewFieldTexture texture;
  require(texture.prepare(field), "density texture preparation");
  field.previewTexture = texture.texture();
  fieldBuffer.upload(&field);
  PrincipledVolumeGPUData volume;
  volume.bindings.bounds = box3(vec3(0), vec3(1));
  volume.bindings.density.field = 0;
  volume.bindings.density.domain = volume.bindings.bounds;
  volume.bindings.density.scale = 2.f;
  volume.fields = fieldBuffer.ptrAs<StoredFieldReferenceGPUData>();
  volume.fieldCount = 1;
  volume.hasConstantTransmissionLog = true;
  volume.previewFloatEligible = true;
  volume.previewShadowCacheEligible = true;
  for (int c = 0; c < 3; ++c)
    volume.constantTransmissionLog[c] = optical_reference::negativeLog(
        volume.bindings.transmissionColor.scale[c]);
  volumeBuffer.upload(&volume);
  VolumeOccurrenceGPUData occurrence;
  for (int c = 0; c < 3; ++c)
    occurrence.worldToObject[c][c] = 1.;
  occurrence.volume.type = VolumeType::PRINCIPLED;
  occurrence.volume.domain = volume.bindings.bounds;
  occurrence.volume.principled = volumeBuffer.ptrAs<PrincipledVolumeGPUData>();
  occurrence.volume.stepSize = .5 / 31.;
  occurrenceBuffer.upload(&occurrence);
  AggregateVolumeSceneGPUData scene{
      occurrenceBuffer.ptrAs<VolumeOccurrenceGPUData>(),
      1,
      true,
      1,
      true,
      1024.f};
  LightGPUData light;
  light.type = LightType::DIRECTIONAL;
  light.distant.direction = vec3(0, 0, -1);
  light.distant.irradiance = 1;
  lightBuffer.upload(&light);
  InstanceLightGPUData instance{};
  instance.lightIndex = 0;
  instance.xfm = mat4(1.f);
  instanceBuffer.upload(&instance);
  FrameGPUData frame{};
  frame.fb.size = uvec2(128);
  frame.referenceShadowRate = 1.;
  frame.world.numLightInstances = 1;
  frame.world.lightInstances = instanceBuffer.ptrAs<InstanceLightGPUData>();
  frame.registry.lights = lightBuffer.ptrAs<LightGPUData>();
  output.reserve(sizeof(vec3));
  for (int segment = 0; segment < 3; ++segment) {
    PreviewDensityIntegral query;
    query.origin[0] = 0.f;
    query.origin[1] = query.origin[2] = .5f;
    query.direction[0] = 1.f;
    query.direction[1] = query.direction[2] = 0.f;
    query.volumeDomain = box3(vec3(0), vec3(1));
    query.fieldDomain = box3(vec3(.25f, 0, 0), vec3(.75f, 1, 1));
    query.lower = segment == 0 ? 0.f : segment == 1 ? .25f : .75f;
    query.upper = segment == 0 ? .25f : segment == 1 ? .75f : 1.f;
    query.anchor = 0.f;
    query.step = 1.f / 16.f;
    query.scale = 1.f;
    query.outside = .4f;
    sampleDensity<<<1, 1>>>(field, query, output.ptrAs<vec3>());
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    vec3 got;
    output.download(&got);
    const float expected = segment == 1 ? .75f : .1f;
    for (int c = 0; c < 3; ++c)
      require(std::abs(got[c] - expected) < 2e-4f,
          "texture/software ROI span integral");
  }
  PreviewShadowCache cache(33, builder);
  auto check = [&](PreviewShadowCacheGPUData data, double expected) {
    checked(cudaDeviceSynchronize());
    sampleCache<<<1, 1>>>(data, vec3(.25f, .5f, .5f), output.ptrAs<vec3>());
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    vec3 got;
    output.download(&got);
    for (int c = 0; c < 3; ++c)
      require(std::abs(double(got[c]) - expected) < 2e-4,
          "cache analytic transmittance");
  };
  check(cache.prepare(scene, frame, 1, 0), std::exp(-1.25));
  require(cache.buildCount() == 1, "first cache build");
  check(cache.prepare(scene, frame, 1, 0), std::exp(-1.25));
  require(cache.buildCount() == 1, "unchanged cache rebuilt");
  light.color = vec3(.5f, 1.f, 2.f);
  light.distant.irradiance = 2.f;
  lightBuffer.upload(&light);
  check(cache.prepare(scene, frame, 10, 0), std::exp(-1.25));
  require(cache.buildCount() == 1, "radiometric light change rebuilt density");
  scene.generation = 10;
  volume.bindings.emission.scale = vec3(5.f);
  volume.bindings.temperature.scale = 2000.f;
  volumeBuffer.upload(&volume);
  check(cache.prepare(scene, frame, 10, 0), std::exp(-1.25));
  require(
      cache.buildCount() == 1, "unrelated scene/source change rebuilt density");
  for (int c = 0; c < 3; ++c)
    volume.constantTransmissionLog[c] *= 2.;
  volumeBuffer.upload(&volume);
  ++scene.generation;
  check(cache.prepare(scene, frame, 10, 0), std::exp(-2.5));
  require(
      cache.buildCount() == 1, "optical coefficient change rebuilt density");
  for (int c = 0; c < 3; ++c)
    volume.constantTransmissionLog[c] *= .5;
  volumeBuffer.upload(&volume);
  ++scene.generation;
  check(cache.prepare(scene, frame, 10, 0), std::exp(-1.25));
  light.distant.direction = vec3(-1, 0, 0);
  lightBuffer.upload(&light);
  check(cache.prepare(scene, frame, 2, 0), std::exp(-2.4375));
  require(cache.buildCount() == 2, "light revision did not rebuild");
  volume.bindings.density.scale = 1.f;
  volumeBuffer.upload(&volume);
  scene.generation = 2;
  check(cache.prepare(scene, frame, 2, 0), std::exp(-1.21875));
  require(cache.buildCount() == 3, "volume revision did not rebuild");
  frame.referenceShadowRate = .5;
  check(cache.prepare(scene, frame, 2, 0), std::exp(-1.21875));
  require(cache.buildCount() == 4, "rate revision did not rebuild");
  occurrence.worldToObject[0][0] = .5; // twice the world-space X extent
  occurrenceBuffer.upload(&occurrence);
  scene.generation = 3;
  check(cache.prepare(scene, frame, 2, 0), std::exp(-2.734375));
  require(cache.buildCount() == 5, "instance revision/world-distance scaling");
  scene.previewFloatEligible = false;
  require(!cache.prepare(scene, frame, 2, 0).density, "wide scene used cache");
  scene.previewFloatEligible = true;
  check(cache.prepare(scene, frame, 2, 0), std::exp(-2.734375));
  require(cache.buildCount() == 6, "eligibility restoration");
  frame.referenceShadowRate = 2.;
  require(!cache.prepare(scene, frame, 2, 0).density,
      "higher shadow rates must bypass the fixed-resolution cache");
  frame.referenceShadowRate = .5;
  frame.world.numLightInstances = 2;
  require(!cache.prepare(scene, frame, 2, 0).density,
      "unsupported light set used cache");
  frame.world.numLightInstances = 1;
  check(cache.prepare(scene, frame, 2, 0), std::exp(-2.734375));
  require(cache.buildCount() == 7, "light-set restoration");
  for (auto &value : values)
    value *= 2.f;
  density.upload(values); // same pointer, different finalized content
  // Real fields replace their immutable snapshot on revision. This fixture
  // retains its allocation, so explicitly remove the old derived texture.
  field.previewTexture = 0;
  fieldBuffer.upload(&field);
  ++volume.densityRevision.revision;
  volumeBuffer.upload(&volume);
  ++scene.generation;
  check(cache.prepare(scene, frame, 2, 0), std::exp(-5.46875));
  require(cache.buildCount() == 8, "density revision was ignored");
  volume.bindings.temperature.field = 0;
  volume.bindings.temperature.domain = box3(vec3(0.f), vec3(.5f));
  volumeBuffer.upload(&volume);
  ++scene.generation;
  check(cache.prepare(scene, frame, 2, 0), std::exp(-5.46875));
  require(cache.buildCount() == 9, "quadrature domain cuts were ignored");
  for (vec3 direction : {vec3(.3f, .7f, .2f), vec3(.1f, .2f, 1.f)}) {
    direction = normalize(direction);
    light.distant.direction = -direction;
    lightBuffer.upload(&light);
    const double length = std::min(
        1.75 / direction.x, std::min(.5 / direction.y, .5 / direction.z));
    const double integral = length * (2.25 + .5 * direction.x * length);
    check(cache.prepare(scene, frame, cache.buildCount() + 10, 0),
        std::exp(-integral));
  }
  std::puts(
      "Preview shadow cache: analytic lookup, precise content keys and fallback pass");
}
int main()
try {
  run(PreviewShadowBuilder::March);
  run(PreviewShadowBuilder::Columns);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
