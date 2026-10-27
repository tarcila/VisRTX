// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cmath>
#include <cstdio>
#include "gpu/renderer/volumePreviewLattice.h"
#ifdef PREVIEW_LATTICE_CUDA
#include <cuda_runtime.h>
#endif

namespace preview_lattice_test {
constexpr int CASES = 18;
struct Result
{
  double cursor, next, expected, upper, source;
  bool partition;
  bool endpointCollapse;
};
VISRTX_HOST_DEVICE Result execute(int id)
{
  using namespace visrtx;
  namespace av = aggregate_volume;
  VolumeOccurrenceGPUData occurrences[2]{};
  for (auto &o : occurrences) {
    for (int c = 0; c < 3; ++c)
      o.worldToObject[c][c] = 1.;
    o.volume.domain = box3(vec3(0), vec3(1));
    o.volume.stepSize = .25;
  }
  AggregateVolumeSceneGPUData scene{occurrences, 1, true, 1};
  av::Ray ray;
  ray.origin[1] = ray.origin[2] = .5;
  ray.direction[0] = 1.;
  ray.upper = 1.;
  av::Interval span{0, 1};
  double cursor = 0., rate = 1., expected = .25;
  switch (id) {
  case 1:
    cursor = .25;
    expected = .5;
    break;
  case 2:
    cursor = nextafter(.25, 0.);
    break;
  case 3:
    ray.lower = span.lower = cursor = .3;
    span.upper = .9;
    expected = .5;
    break;
  case 4:
  case 11:
    scene.count = 2;
    occurrences[1].worldToObject[0][3] = -.125;
    span.lower = .125;
    cursor = .25;
    expected = .375;
    if (id == 11) {
      const auto swap = occurrences[0];
      occurrences[0] = occurrences[1];
      occurrences[1] = swap;
    }
    break;
  case 5:
    occurrences[0].worldToObject[0][0] = .5;
    ray.upper = span.upper = 2.;
    expected = .5;
    break;
  case 6: // A requested cell beyond a clipped span cannot cross its endpoint.
    span.upper = .125;
    expected = .125;
    break;
  case 7:
  case 10:
    ray.origin[0] = 0x1p54;
    occurrences[0].volume.domain.upper.x = 0x1p56f;
    expected = 1.; // Cell index cannot distinguish index+1 or index+2.
    if (id == 10) {
      scene.count = 2;
      occurrences[1].worldToObject[0][3] = -0x1p54;
      expected = .25;
    }
    break;
  case 8:
    ray.origin[0] = -0x1p54;
    occurrences[0].volume.domain.upper.x = 16.f;
    ray.upper = span.upper = 0x1p54 + 16.;
    cursor = span.lower = 0x1p54;
    expected = span.upper;
    break;
  case 9:
    // Exact inverse of a finite float triangular transform with diagonals
    // 2^-110 and upper off-diagonals 2^110. Squaring the inverse ray direction
    // overflows double, although both the span and requested cells are finite.
    occurrences[0].worldToObject[0][0] = 0x1p110;
    occurrences[0].worldToObject[1][1] = 0x1p110;
    occurrences[0].worldToObject[2][2] = 0x1p110;
    occurrences[0].worldToObject[0][1] = -0x1p330;
    occurrences[0].worldToObject[1][2] = -0x1p330;
    occurrences[0].worldToObject[0][2] = 0x1p550;
    occurrences[0].volume.domain = box3(vec3(1, -1, -1), vec3(2, 1, 1));
    ray.origin[0] = ray.origin[1] = ray.origin[2] = 0.;
    ray.direction[0] = 0.;
    ray.direction[2] = 1.;
    av::intersect(occurrences[0], ray, span);
    cursor = span.lower;
    expected = 5. * 0x1p-552;
    break;
  case 12:
    scene.count = 2;
    occurrences[1].worldToObject[0][3] = -.5;
    occurrences[1].volume.stepSize = .01;
    span.upper = .5;
    break;
  case 13:
    rate = 0x1p127;
    expected = 0x1p-129;
    break;
  case 16:
    occurrences[0].worldToObject[0][0] = 3.;
    occurrences[0].worldToObject[1][0] = 4.;
    av::intersect(occurrences[0], ray, span);
    expected = .05; // object speed 5, not the world direction's unit speed
    break;
  case 17:
    ray.origin[0] = 1.;
    ray.direction[0] = -1.;
    break;
  default:
    break;
  }
  Result result{};
  result.cursor = cursor;
  result.upper = span.upper;
  result.expected = expected;
  result.next = volume_preview::nextCell(scene, ray, span, cursor, rate);
  optical_reference::Coefficients coefficients;
  const double source[3]{2, 4, 8};
  av::SegmentIntegral integral;
  av::integrateSegment(coefficients, source, result.next - cursor, integral);
  result.source = integral.radiance[0];
  result.partition = true;
  result.endpointCollapse = true;
  if (id == 15) {
    // Characterize a limit BEFORE nextCell: the exact unit interval at a
    // distance of 2^54 collapses to equal double endpoints. There is no
    // represented span for this iterator to preserve. This is not a claim
    // that the physical medium is empty or that extreme placement is accurate.
    av::Ray far = ray;
    far.origin[0] = -0x1p54;
    far.upper = INFINITY;
    av::Interval interval;
    result.endpointCollapse = !av::intersect(occurrences[0], far, interval);
  }
  if (id == 14) {
    int cells = 0;
    double t = 0., total = 0.;
    // Test-only guard detects lack of progress. The production iterator has
    // no iteration cutoff. Normal dyadic cells must partition the whole span.
    while (t < 1. && cells < 8) {
      const double next = volume_preview::nextCell(scene, ray, span, t, rate);
      total += 2. * (next - t);
      t = next;
      ++cells;
    }
    result.partition = cells == 4 && t == 1. && total == 2.;
  }
  return result;
}
#ifdef PREVIEW_LATTICE_CUDA
VISRTX_GLOBAL void latticeKernel(Result *results)
{
  const int i = threadIdx.x;
  if (i < CASES)
    results[i] = execute(i);
}
#endif
inline int run()
{
  Result host[CASES];
#ifdef PREVIEW_LATTICE_CUDA
  Result *gpu = nullptr;
  auto error = cudaMalloc(&gpu, sizeof(host));
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    return 1;
  }
  latticeKernel<<<1, 32>>>(gpu);
  error = cudaGetLastError();
  if (error == cudaSuccess)
    error = cudaDeviceSynchronize();
  if (error == cudaSuccess)
    error = cudaMemcpy(host, gpu, sizeof(host), cudaMemcpyDeviceToHost);
  cudaFree(gpu);
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    return 1;
  }
#else
  for (int i = 0; i < CASES; ++i)
    host[i] = execute(i);
#endif
  int failures = 0;
  for (int i = 0; i < CASES; ++i) {
    const auto &r = host[i];
    if (!std::isfinite(r.next) || !(r.next > r.cursor) || r.next > r.upper
        || r.next != r.expected || r.source != 2. * (r.next - r.cursor)
        || !r.partition || !r.endpointCollapse) {
      std::fprintf(stderr,
          "lattice case %d: next=%a expected=%a cursor=%a upper=%a\n",
          i,
          r.next,
          r.expected,
          r.cursor,
          r.upper);
      ++failures;
    }
  }
  std::printf("Preview lattice: %d cases, %d failures\n", CASES, failures);
  return failures ? 1 : 0;
}
} // namespace preview_lattice_test
