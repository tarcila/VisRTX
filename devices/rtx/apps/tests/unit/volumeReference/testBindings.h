// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "spatial_field/ReferenceFieldPreparation.h"

#ifdef VOLUME_BINDINGS_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <cfenv>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace reference_binding_test {
namespace ref = visrtx::optical_reference;
using visrtx::box3;
using visrtx::vec3;

// A concrete canonical-corner test provider, not a replacement for the existing
// stored-field finalizers. Its sample and global range read the same corner
// data.
struct Field
{
  ref::FieldSampleDescriptor descriptor;
  float scalar[8]{};
  vec3 rgb[8]{};
  bool rangeAvailable{true};
  uint64_t rangeRevision{1};
};

struct Provider
{
  Field fields[4];
  mutable int rangeCalls[4]{};
  mutable int describeCalls{0};
  bool changeOnRecheck{false};

  bool describe(int32_t index, ref::FieldSampleDescriptor &out) const;
  template <typename T>
  bool range(int32_t index, ref::RawRangeCertificate<T> &out) const;
};

inline bool Provider::describe(
    int32_t index, ref::FieldSampleDescriptor &out) const
{
  ++describeCalls;
  if (index < 0 || index >= 4)
    return false;
  out = fields[index].descriptor;
  if (changeOnRecheck && describeCalls > 5)
    ++out.key.revision;
  return true;
}

template <typename T>
inline bool Provider::range(
    int32_t index, ref::RawRangeCertificate<T> &out) const
{
  ++rangeCalls[index];
  const auto &f = fields[index];
  if (!f.rangeAvailable)
    return false;
  out.key = f.descriptor.key;
  out.key.revision = f.rangeRevision;
  if constexpr (std::is_same_v<T, float>)
    out.range = ref::canonicalRange(f.scalar, 8);
  else
    out.range = ref::canonicalRange(f.rgb, 8);
  return true;
}

struct Sampler
{
  const Field *fields;
  int *calls;
  template <typename T>
  VISRTX_HOST_DEVICE T sample(int32_t index, const vec3 &p) const;
};

template <typename T>
VISRTX_HOST_DEVICE T Sampler::sample(int32_t index, const vec3 &p) const
{
  ++*calls;
  const auto &f = fields[index];
  const T *corners;
  if constexpr (std::is_same_v<T, float>)
    corners = f.scalar;
  else
    corners = f.rgb;
  // All tested field domains are [0,1]^3, with endpoint coordinates p. Uses the
  // production controlled interpolation, not hardware filtering assumptions.
  const T a = ref::canonicalLerp(corners[0], corners[1], p.x);
  const T b = ref::canonicalLerp(corners[2], corners[3], p.x);
  const T c = ref::canonicalLerp(corners[4], corners[5], p.x);
  const T d = ref::canonicalLerp(corners[6], corners[7], p.x);
  return ref::canonicalLerp(
      ref::canonicalLerp(a, b, p.y), ref::canonicalLerp(c, d, p.y), p.z);
}

struct Blackbody
{
  VISRTX_HOST_DEVICE vec3 operator()(float) const;
};
VISRTX_HOST_DEVICE vec3 Blackbody::operator()(float) const
{
  return vec3(0.f);
}

struct Input
{
  Field fields[4];
  visrtx::PrincipledVolumeBindingsGPUData bindings;
  vec3 p;
};
struct Output
{
  ref::Coefficients coefficients;
  float density;
  vec3 transmission;
  float phase;
  int calls;
  int enclosure;
  bool tinyDomain;
};

VISRTX_HOST_DEVICE Output sample(const Input &input)
{
  Output out{};
  out.tinyDomain =
      ref::contains(box3(vec3(0.f), vec3(0x1p-147f)), vec3(0x1p-149f));
  Sampler sampler{input.fields, &out.calls};
  Blackbody blackbody;
  out.coefficients =
      ref::evaluateBindings(input.bindings, input.p, sampler, blackbody);
  out.density = ref::evaluateBinding(input.bindings.density, input.p, sampler);
  out.transmission =
      ref::evaluateBinding(input.bindings.transmissionColor, input.p, sampler);
  out.phase = ref::evaluateBinding(input.bindings.anisotropy, input.p, sampler);
  // The device computes raw hulls too, exercising non-FTZ extrema/reductions.
  out.enclosure = 1;
  for (int field : {0, 1}) {
    if (field == 0) {
      const auto range = ref::canonicalRange(input.fields[0].scalar, 8);
      const auto effective = ref::bindingRange(
          input.bindings.density, input.bindings.bounds, range);
      out.enclosure &=
          ref::promote(out.density) >= ref::promote(effective.lower)
          && ref::promote(out.density) <= ref::promote(effective.upper);
    } else {
      const auto range = ref::canonicalRange(input.fields[1].rgb, 8);
      const auto effective = ref::bindingRange(
          input.bindings.transmissionColor, input.bindings.bounds, range);
      for (int c = 0; c < 3; ++c)
        out.enclosure &= ref::promote(out.transmission[c])
                >= ref::promote(effective.lower[c])
            && ref::promote(out.transmission[c])
                <= ref::promote(effective.upper[c]);
    }
  }
  return out;
}

#ifdef VOLUME_BINDINGS_CUDA
VISRTX_GLOBAL void referenceBindingKernel(
    const Input *in, Output *out, int count)
{
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count)
    out[i] = sample(in[i]);
}

VISRTX_GLOBAL void referenceProductKernel(
    const float *weights, float *out, int count)
{
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    const float maxFloat = std::numeric_limits<float>::max();
    out[3 * i] = ref::canonicalLerp(-maxFloat, maxFloat, weights[i]);
    out[3 * i + 1] =
        ref::bindingProduct(-2.f, -std::numeric_limits<float>::denorm_min());
    out[3 * i + 2] =
        ref::canonicalLerp(std::numeric_limits<float>::denorm_min(),
            std::numeric_limits<float>::denorm_min(),
            weights[i]);
  }
}
#endif

inline void require(bool value, const char *message)
{
  if (!value)
    throw std::runtime_error(message);
}

inline Provider provider()
{
  Provider p;
  for (int i = 0; i < 4; ++i) {
    auto &f = p.fields[i];
    f.descriptor = {{uint64_t(100 + i), 1, 1},
        i == 1 || i == 2 ? visrtx::FieldValueType::RGB
                         : visrtx::FieldValueType::Scalar,
        {vec3(0.f), vec3(1.f)}};
    for (int c = 0; c < 8; ++c) {
      f.scalar[c] = i == 0 ? -float(c + 1) : .25f;
      f.rgb[c] = i == 1 ? vec3(-.25f - .03125f * c, .25f + .03125f * c, .5f)
                        : vec3(2.f, 4.f, 8.f);
    }
  }
  p.fields[2].rangeAvailable = false; // source only: legal without ranges
  p.fields[3].rangeAvailable = false; // phase only: legal without ranges
  return p;
}

inline visrtx::PrincipledVolumeBindingsGPUData bindings()
{
  visrtx::PrincipledVolumeBindingsGPUData b;
  b.bounds = {vec3(-1.f), vec3(2.f)};
  b.density.field = 0;
  b.density.scale = -1.f;
  b.density.outside = -2.f;
  b.transmissionColor.field = 1;
  b.transmissionColor.scale = vec3(-1.f, 1.f, 1.f);
  b.transmissionColor.outside = vec3(-1.f, 1.f, 1.f);
  b.emission.field = 2;
  b.emission.scale = vec3(1.f);
  b.emission.outside = vec3(1.f, 2.f, 3.f);
  b.anisotropy.field = b.secondaryAnisotropy.field = 3;
  b.anisotropy.outside = .5f;
  return b;
}

inline void verify(
    const Input &in, const Output &out, const ref::Bounds &bounds)
{
  require(out.enclosure, "raw/effective range does not enclose sample");
  require(out.tinyDomain, "subnormal domain comparison flushed");
  const bool inside = ref::contains(in.fields[0].descriptor.domain, in.p);
  const bool medium = ref::contains(in.bindings.bounds, in.p);
  // Corner density values are -(1+x+2y+4z), scale=-1.
  const double expected = inside ? 1. + in.p.x + 2. * in.p.y + 4. * in.p.z : 2.;
  require(std::abs(ref::promote(out.density) - expected) < 2e-6,
      "independent trilinear/binding oracle");
  require(out.calls == (inside ? 6 : 0),
      "membership-before-filtering or duplicate sampling");
  for (int c = 0; c < 3; ++c) {
    require(ref::extinction(out.coefficients, c) <= ref::majorant(bounds),
        "field-backed optical underbound");
    require(out.coefficients.emission[c]
            == (medium ? (inside ? (2 << c) : c + 1) : 0),
        "sample-only source/outside/domain");
    const double transmission = inside
        ? (c == 2 ? .5 : .25 + .03125 * (in.p.x + 2. * in.p.y + 4. * in.p.z))
        : 1.;
    require(std::abs(ref::promote(out.transmission[c]) - transmission) < 2e-7,
        "RGB signed multiplier oracle");
    const double e = medium ? expected * -std::log(transmission) : 0.;
    require(std::abs(out.coefficients.scattering[c] - e)
            <= std::abs(e) * 0x1p-36 + 1e-12,
        "optical physical oracle");
  }
}

inline int run()
try {
  require(std::fegetround() == FE_TONEAREST, "RN host required");
  auto p = provider();
  auto b = bindings();
  ref::PreparedVolumeBindings prepared;
  require(ref::prepareVolumeBindings(b, p, prepared).error
          == ref::PreparationError::None,
      "valid preparation");
  require(
      prepared.dependencyCount == 4, "shared phase binding not deduplicated");
  require(p.rangeCalls[0] == 1 && p.rangeCalls[1] == 1 && p.rangeCalls[2] == 0
          && p.rangeCalls[3] == 0,
      "sample-only role incorrectly requires ranges");
  require(
      ref::matchesPreparedFields(prepared, p), "prepared revision mismatch");
  std::vector<Input> inputs;
  for (float x : {-1.f, -.001f, 0.f, .125f, .5f, 1.f, 1.001f, 2.f, 3.f})
    for (float y : {0.f, .25f, 1.f})
      for (float z : {0.f, .5f, 1.f}) {
        Input in{};
        for (int i = 0; i < 4; ++i)
          in.fields[i] = p.fields[i];
        in.bindings = prepared.bindings;
        in.p = {x, y, z};
        inputs.push_back(in);
      }
  std::vector<Output> outputs;
  for (const auto &in : inputs) {
    outputs.push_back(sample(in));
    verify(in, outputs.back(), prepared.bounds);
  }

  // Transactional metadata failures, not per-sample checks.
  auto fails = [&](Provider bad,
                   visrtx::PrincipledVolumeBindingsGPUData badBindings,
                   ref::PreparationError error) {
    auto sentinel = prepared;
    const auto result = ref::prepareVolumeBindings(badBindings, bad, sentinel);
    require(result.error == error, "wrong preparation diagnostic");
    require(sentinel.dependencyCount == prepared.dependencyCount
            && sentinel.bounds.absorption[0] == prepared.bounds.absorption[0]
            && sentinel.dependencies[0].sample.key
                == prepared.dependencies[0].sample.key,
        "failed preparation published partial state");
  };
  auto bad = p;
  bad.fields[0].rangeRevision = 2;
  fails(bad, b, ref::PreparationError::RevisionMismatch);
  bad = p;
  bad.fields[1].descriptor.valueType = visrtx::FieldValueType::Scalar;
  fails(bad, b, ref::PreparationError::TypeMismatch);
  bad = p;
  bad.fields[0].rangeAvailable = false;
  fails(bad, b, ref::PreparationError::RangeUnavailable);
  bad = p;
  bad.describeCalls = 0;
  bad.changeOnRecheck = true;
  fails(bad, b, ref::PreparationError::RevisionMismatch);
  auto invalid = b;
  invalid.transmissionDistance.scale = 0.f;
  fails(p, invalid, ref::PreparationError::OpticalRangeUnavailable);
  invalid = b;
  invalid.bounds.upper = invalid.bounds.lower;
  fails(p, invalid, ref::PreparationError::InvalidMediumDomain);

  bad = p;
  for (float &value : bad.fields[0].scalar)
    value = -std::numeric_limits<float>::max();
  invalid = b;
  invalid.density.scale = -2.f;
  fails(bad, invalid, ref::PreparationError::InvalidRangeMetadata);
  bad = p;
  bad.fields[2].descriptor.valueType = visrtx::FieldValueType::Scalar;
  fails(bad, b, ref::PreparationError::TypeMismatch);

  // Recommits/slot reuse must invalidate a prepared snapshot without samples.
  bad = p;
  ++bad.fields[0].descriptor.key.revision;
  require(!ref::matchesPreparedFields(prepared, bad), "stale sample accepted");
  bad.fields[0].rangeRevision = bad.fields[0].descriptor.key.revision;
  for (float &value : bad.fields[0].scalar)
    value = -16.f;
  const double priorMajorant = ref::majorant(prepared.bounds);
  require(ref::prepareVolumeBindings(b, bad, prepared).error
          == ref::PreparationError::None,
      "same-revision recovery failed");
  require(ref::majorant(prepared.bounds) > priorMajorant,
      "recommit did not rebuild bound from new data");
  auto contractChanged = bad;
  ++contractChanged.fields[0].descriptor.key.contract;
  require(!ref::matchesPreparedFields(prepared, contractChanged),
      "changed sampling contract accepted");
  auto domainChanged = bad;
  domainChanged.fields[0].descriptor.domain.upper = vec3(.5f);
  require(!ref::matchesPreparedFields(prepared, domainChanged),
      "changed domain accepted without revision");
  ++bad.fields[0].descriptor.key.identity;
  require(!ref::matchesPreparedFields(prepared, bad),
      "reused registry index accepted");
  require(ref::prepareVolumeBindings(b, p, prepared).error
          == ref::PreparationError::None,
      "restore fixture");

  // Removing one of two same-field bindings keeps the dependency; removing
  // its final use drops it. A second volume's scales do not modify raw ranges.
  auto removed = b;
  removed.anisotropy.field = -1;
  ref::PreparedVolumeBindings removedPrepared;
  require(ref::prepareVolumeBindings(removed, p, removedPrepared).error
              == ref::PreparationError::None
          && removedPrepared.dependencyCount == 4,
      "remaining shared binding lost dependency");
  removed.secondaryAnisotropy.field = -1;
  require(ref::prepareVolumeBindings(removed, p, removedPrepared).error
              == ref::PreparationError::None
          && removedPrepared.dependencyCount == 3,
      "unused dependency retained");
  auto scaled = b;
  scaled.density.scale = -2.f;
  ref::PreparedVolumeBindings scaledPrepared;
  require(ref::prepareVolumeBindings(scaled, p, scaledPrepared).error
              == ref::PreparationError::None
          && ref::majorant(scaledPrepared.bounds)
              > ref::majorant(prepared.bounds)
          && ref::matchesPreparedFields(prepared, p),
      "volume-specific bound changed shared field state");

  // Full-domain containment excludes unreachable outside=0 transmission.
  auto contained = b;
  contained.bounds = {vec3(0.f), vec3(1.f)};
  contained.transmissionColor.outside = vec3(0.f);
  ref::PreparedVolumeBindings containedPrepared;
  require(ref::prepareVolumeBindings(contained, p, containedPrepared).error
          == ref::PreparationError::None,
      "unreachable outside poisoned bound");
  contained.bounds.upper = vec3(2.f);
  fails(p, contained, ref::PreparationError::OpticalRangeUnavailable);

  // Empty ROI/outside-only need no range, but retain typed sample metadata.
  auto empty = p;
  empty.fields[0].descriptor.domain = {vec3(1.f), vec3(0.f)};
  empty.fields[0].rangeAvailable = false;
  ref::PreparedVolumeBindings emptyPrepared;
  require(ref::prepareVolumeBindings(b, empty, emptyPrepared).error
          == ref::PreparationError::None,
      "empty ROI did not use outside");
  auto touching = b;
  touching.bounds = {vec3(1.f), vec3(2.f)};
  bad = p;
  bad.fields[0].rangeAvailable = false;
  fails(bad, touching, ref::PreparationError::RangeUnavailable);
  // Source-only vacuum can be prepared without any extinction field/range.
  auto vacuum = b;
  vacuum.density.field = -1;
  vacuum.density.scale = 0.f;
  vacuum.transmissionColor.field = -1;
  vacuum.transmissionColor.scale = vec3(.5f);
  ref::PreparedVolumeBindings vacuumPrepared;
  require(ref::prepareVolumeBindings(vacuum, p, vacuumPrepared).error
              == ref::PreparationError::None
          && ref::majorant(vacuumPrepared.bounds) == 0.,
      "vacuum/source-only preparation");

  // Raw range union includes background/endpoints independently of occupancy.
  ref::RawRange<float> raw;
  const float values[] = {-2.f, -1.f};
  raw = ref::canonicalRange(values, 2);
  raw = ref::mergeRanges(raw, ref::RawRange<float>{-9.f, 7.f, false});
  require(raw.lower == -9.f && raw.upper == 7.f, "raw/background range merge");
  require(ref::canonicalRange<float>(nullptr, 0).empty,
      "empty range conflated with zero");

  const float tiny = std::numeric_limits<float>::denorm_min();
  const float maxFloat = std::numeric_limits<float>::max();
  const std::vector<float> weights{
      0.f, tiny, 0x1p-24f, .25f, .5f, .75f, 1.f - 0x1p-24f, 1.f};
  std::vector<float> products;
  for (float w : weights) {
    const float value = ref::canonicalLerp(-maxFloat, maxFloat, w);
    require(std::isfinite(value) && value >= -maxFloat && value <= maxFloat,
        "convex lerp overflow");
    const long double exact = (2.L * static_cast<long double>(w) - 1.L)
        * static_cast<long double>(maxFloat);
    require(std::abs(static_cast<long double>(value) - exact)
            <= static_cast<long double>(maxFloat) * 0x1p-23L,
        "lerp oracle");
    const float product = ref::bindingProduct(-2.f, -tiny);
    require(ref::floatBits(product) == 2, "subnormal product flushed");
    const float same = ref::canonicalLerp(tiny, tiny, w);
    require(ref::floatBits(same) == 1, "equal endpoint lost");
    products.insert(products.end(), {value, product, same});
  }
#ifdef VOLUME_BINDINGS_CUDA
  int count = 0;
  require(
      cudaGetDeviceCount(&count) == cudaSuccess && count > 0, "CUDA required");
  thrust::device_vector<Input> gpuInputs(inputs);
  thrust::device_vector<Output> gpuOutputs(inputs.size());
  referenceBindingKernel<<<(inputs.size() + 63) / 64, 64>>>(
      thrust::raw_pointer_cast(gpuInputs.data()),
      thrust::raw_pointer_cast(gpuOutputs.data()),
      int(inputs.size()));
  require(cudaGetLastError() == cudaSuccess, "binding launch");
  require(cudaDeviceSynchronize() == cudaSuccess, "binding completion");
  const thrust::host_vector<Output> actual = gpuOutputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    verify(inputs[i], actual[i], prepared.bounds);
    require(actual[i].density == outputs[i].density
            && actual[i].phase == outputs[i].phase,
        "CPU/CUDA binding mismatch");
    for (int c = 0; c < 3; ++c)
      require(actual[i].coefficients.scattering[c]
                  == outputs[i].coefficients.scattering[c]
              && actual[i].transmission[c] == outputs[i].transmission[c],
          "CPU/CUDA optics mismatch");
  }
  thrust::device_vector<float> gpuWeights(weights);
  thrust::device_vector<float> gpuProducts(products.size());
  referenceProductKernel<<<1, 32>>>(thrust::raw_pointer_cast(gpuWeights.data()),
      thrust::raw_pointer_cast(gpuProducts.data()),
      int(weights.size()));
  require(cudaGetLastError() == cudaSuccess, "product launch");
  require(cudaDeviceSynchronize() == cudaSuccess, "product completion");
  const thrust::host_vector<float> actualProducts = gpuProducts;
  for (size_t i = 0; i < products.size(); ++i)
    require(ref::floatBits(actualProducts[i]) == ref::floatBits(products[i]),
        "CPU/CUDA endpoint/product mismatch");
#endif
  std::printf(
      "Reference bindings: %zu samples, revision/role/range checks pass\n",
      inputs.size());
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
} // namespace reference_binding_test
