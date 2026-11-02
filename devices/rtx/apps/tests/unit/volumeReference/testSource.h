// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "gpu/volumeSource.h"
#include "gpu/volumeSourceOwnership.h"
#include "world/VolumeSourcePreparation.h"

namespace source_test {
using namespace visrtx;
constexpr int SAMPLES = 65536;
struct Blackbody
{
  VISRTX_HOST_DEVICE vec3 operator()(float) const;
};
VISRTX_HOST_DEVICE vec3 Blackbody::operator()(float) const
{
  return vec3(0);
}
struct Fixture
{
  PrincipledVolumeGPUData volume;
  VolumeOccurrenceGPUData occurrences[2];
  AggregateVolumeSceneGPUData scene;
  double cdf[9]{};
  VISRTX_HOST_DEVICE explicit Fixture(int id);
};
VISRTX_HOST_DEVICE Fixture::Fixture(int id)
{
  const int variant = id;
  id = variant == 5  ? 0
      : variant == 6 ? 1
      : variant == 7 ? 2
      : variant == 8 ? 4
                     : id;
  for (int i = 0; i <= 8; ++i)
    cdf[i] = variant == 6 ? (i == 0 ? 0. : 1.) : double(i) / 8.;
  volume.bindings.bounds = id == 1
      ? box3(vec3(-.5f), vec3(.5f))
      : box3(vec3(.75f, -.25f, .75f), vec3(1.25f, .25f, 1.25f));
  volume.bindings.transmissionColor.scale = vec3(1.f);
  volume.bindings.emission.scale = vec3(1, 2, 3);
  for (int i = 0; i < 2; ++i) {
    auto &o = occurrences[i];
    o.volume.type = VolumeType::PRINCIPLED;
    o.volume.principled = &volume;
    o.volume.domain = volume.bindings.bounds;
    o.volume.mightEmit = true;
    o.groupSlot = size_t(i);
    for (int c = 0; c < 3; ++c) {
      o.worldToObject[c][c] = 1.;
      o.sourceObjectToWorld[c][c] = 1.;
    }
    o.sourceWorldVolume = id == 1 ? 1. : .125;
    if (variant >= 5)
      o.volume.sourceImportance = {cdf, 2};
    o.sourceCenter = id == 1 ? glm::dvec3(0) : glm::dvec3(1, 0, 1);
    o.sourceRadius = id == 1 ? sqrt(.75) : sqrt(.1875);
    if (id == 3 || id == 4) {
      o.worldToObject[0][0] = .5;
      o.worldToObject[1][1] = id == 3 ? 2. : 1.;
      o.sourceCenter = glm::dvec3(2, 0, 1);
      o.sourceRadius = sqrt(id == 3 ? .328125 : .375);
      o.sourceObjectToWorld[0][0] = 2.;
      o.sourceObjectToWorld[1][1] = id == 3 ? .5 : 1.;
      o.sourceWorldVolume = id == 3 ? .125 : .25;
    }
  }
  scene.occurrences = occurrences;
  scene.count = id == 2 ? 2 : 1;
  scene.complete = true;
  scene.sourceSamplingReady = true;
  scene.sourceCount = scene.count;
}
struct Result
{
  double value[3]{};
  double distance{0.};
};
VISRTX_HOST_DEVICE Result trial(int id, int index)
{
  Fixture f(id);
  PCGState state;
  pcg_init(&state,
      detail::pcg_mix64(uint64_t(index) + 915),
      detail::pcg_mix64(uint64_t(id) + 42));
  Blackbody blackbody;
  const auto sample =
      volume_source::sample(f.scene, glm::dvec3(0), state, blackbody);
  Result result;
  if (sample.valid) {
    const double cosine = sample.ray.direction[2];
    result.distance = sample.ray.upper;
    if (cosine > 0.)
      for (int c = 0; c < 3; ++c)
        result.value[c] =
            .7 * cosine * sample.weight[c] / 3.14159265358979323846;
  }
  return result;
}
inline void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
inline void controls()
{
  using namespace volume_source;
  const auto native = surfacePlan(true, uint32_t(SbtCallableEntryPoints::PBR));
  NextRay next;
  next.event = NextRayEvent::Continuous;
  next.component = NextRayComponent::NativeReflection;
  require(!outgoingOwner(native, next), "native reflected source not replaced");
  next.component = NextRayComponent::NativeTransmission;
  require(outgoingOwner(native, next), "transmitted source suppressed");
  next.event = NextRayEvent::Singular;
  require(outgoingOwner(native, next), "delta source suppressed");
  next.event = NextRayEvent::Continuous;
  next.component = NextRayComponent::MDL;
  require(outgoingOwner(
              surfacePlan(true, uint32_t(SbtCallableEntryPoints::Last)), next),
      "unqualified MDL source suppressed");
  require(surfacePlan(true, uint32_t(SbtCallableEntryPoints::Matte))
          == SurfacePlan::DirectOnly,
      "matte source plan");
  Fixture f(0);
  f.scene.sourceSamplingReady = false;
  PCGState random;
  pcg_init(&random, 1, 3);
  Blackbody blackbody;
  require(!sample(f.scene, glm::dvec3(0), random, blackbody).valid,
      "incomplete proposal used");
  next.component = NextRayComponent::NativeReflection;
  require(outgoingOwner(surfacePlan(available(f.scene),
                            uint32_t(SbtCallableEntryPoints::PBR)),
              next),
      "unavailable proposal lost continuation");
  volume_tracking::PathState path;
  optical_reference::Coefficients coefficient;
  coefficient.emission[0] = 4.;
  path.sourceOwned = false;
  volume_tracking::collide(coefficient, 2., .5, path);
  require(path.radiance[0] == 0. && path.throughput[0] == 1.,
      "null ownership/weight");
  path.sourceOwned = true;
  volume_tracking::collide(coefficient, 2., .5, path);
  require(
      path.radiance[0] == 2., "incoming source not restored at a new vertex");
  glm::dmat4 transform(1.);
  transform[0][0] = 2.;
  transform[1][1] = .5;
  require(prepareVolumeSource(f.occurrences[0], transform),
      "source metadata preparation");
  require(
      glm::length(f.occurrences[0].sourceCenter - glm::dvec3(2, 0, 1)) < 1e-12
          && f.occurrences[0].sourceRadius >= sqrt(.328125),
      "world source bounds");
}
inline void check(int id, const std::vector<Result> &results)
{
  // Far references: independent order16/24 Gauss volume integral agree to
  // <1e-17. Inside: analytic rectangular potential difference, no r clamp.
  const double expected[9]{.009849702122354463,
      .43199212101760975,
      2 * .009849702122354463,
      .002646826250379459,
      .005266233324224048,
      .009849702122354463,
      .43199212101760975,
      2 * .009849702122354463,
      .005266233324224048};
  for (int c = 0; c < 3; ++c) {
    long double sum = 0., squares = 0.;
    for (const auto &r : results) {
      require(std::isfinite(r.value[c]) && r.value[c] >= 0.,
          "invalid source sample");
      if (id == 1 || id == 6)
        require(r.value[c] <= (id == 6 ? 14. : 1.4) * (c + 1) + 1e-12,
            "near-source mixture weight unbounded");
      sum += r.value[c];
      squares += r.value[c] * r.value[c];
    }
    const double mean = double(sum / results.size());
    const double variance = std::max(0.,
        double((squares - sum * sum / results.size()) / (results.size() - 1)));
    const double tolerance = 6. * sqrt(variance / results.size()) + 1e-5;
    if (std::abs(mean - (c + 1) * expected[id]) > tolerance) {
      std::fprintf(stderr,
          "source %d channel %d: %.12g expected %.12g tolerance %.12g\n",
          id,
          c,
          mean,
          (c + 1) * expected[id],
          tolerance);
      throw std::runtime_error("source measure");
    }
  }
}
#ifdef SOURCE_CUDA
inline void checked(cudaError_t e)
{
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
VISRTX_GLOBAL void samples(int id, Result *out)
{
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < SAMPLES)
    out[i] = trial(id, i);
}
#endif
inline int run()
{
  controls();
  std::vector<Result> out(SAMPLES);
#ifdef SOURCE_CUDA
  Result *device = nullptr;
  checked(cudaMalloc(&device, out.size() * sizeof(Result)));
#endif
  for (int id = 0; id < 9; ++id) {
#ifdef SOURCE_CUDA
    samples<<<(SAMPLES + 127) / 128, 128>>>(id, device);
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(out.data(),
        device,
        out.size() * sizeof(Result),
        cudaMemcpyDeviceToHost));
#else
    for (int i = 0; i < SAMPLES; ++i)
      out[i] = trial(id, i);
#endif
    check(id, out);
  }
#ifdef SOURCE_CUDA
  checked(cudaFree(device));
#endif
  std::puts(
      "Volume source: far/inside/duplicate/transformed measures and ownership controls pass");
  return 0;
}
} // namespace source_test
