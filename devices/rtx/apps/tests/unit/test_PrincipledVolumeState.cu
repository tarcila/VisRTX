// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "spatial_field/StoredFieldReferenceSampler.h"
#include "volume/PrincipledVolume.h"

#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <atomic>
#include <cfenv>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace visrtx;
namespace ref = optical_reference;

static std::atomic<int> errors{0};
static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "ANARI: %s\n", message);
    ++errors;
  }
}
static void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
static void checked(cudaError_t error)
{
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
struct Objects
{
  ANARIDevice d{};
  std::vector<ANARIObject> handles;
  ~Objects();
  template <typename T>
  T keep(T h);
};
Objects::~Objects()
{
  for (auto it = handles.rbegin(); it != handles.rend(); ++it)
    anariRelease(d, *it);
  anariRelease(d, d);
}
template <typename T>
T Objects::keep(T h)
{
  handles.push_back(reinterpret_cast<ANARIObject>(h));
  return h;
}
static bool valid(Objects &o, ANARIObject object)
{
  bool value = false;
  require(
      anariGetProperty(
          o.d, object, "valid", ANARI_BOOL, &value, sizeof(value), ANARI_WAIT),
      "valid property");
  return value;
}
static ANARIVolume volume(Objects &o, box3 bounds)
{
  auto v = o.keep(anariNewVolume(o.d, "principledVolume"));
  anariSetParameter(o.d, v, "bounds", ANARI_FLOAT32_BOX3, &bounds);
  anariCommitParameters(o.d, v);
  require(valid(o, v), "new volume preparation");
  return v;
}
static const PrincipledVolumeGPUData &state(ANARIVolume v)
{
  const auto *data = reinterpret_cast<PrincipledVolume *>(v)->committedData();
  require(data != nullptr, "missing committed host payload");
  return *data;
}
struct Blackbody
{
  VISRTX_HOST_DEVICE vec3 operator()(float) const;
};
VISRTX_HOST_DEVICE vec3 Blackbody::operator()(float) const
{
  return vec3(
      1.f, 2.f, 3.f); // Interface oracle, NOT a calibrated Planck renderer.
}
struct Result
{
  ref::Coefficients coefficients;
  ref::Coefficients uncached;
  VolumePhaseFunction phase;
};
VISRTX_GLOBAL void samplePrincipledState(
    const PrincipledVolumeGPUData *data, vec3 p, Result *out)
{
  StoredFieldReferenceSampler sampler{data->fields};
  Blackbody blackbody;
  out->coefficients = ref::evaluateBindings(data->bindings,
      p,
      sampler,
      blackbody,
      data->hasConstantTransmissionLog ? data->constantTransmissionLog
                                       : nullptr);
  out->uncached = ref::evaluateBindings(data->bindings, p, sampler, blackbody);
  out->phase = {ref::evaluateBinding(data->bindings.anisotropy, p, sampler),
      ref::evaluateBinding(data->bindings.secondaryAnisotropy, p, sampler),
      ref::evaluateBinding(data->bindings.secondaryWeight, p, sampler)};
}
static Result sample(ANARIVolume v, vec3 p)
{
  const auto *pointer = static_cast<const PrincipledVolumeGPUData *>(
      reinterpret_cast<Object *>(v)->deviceData());
  require(pointer != nullptr, "missing device payload");
  PrincipledVolumeGPUData copied;
  checked(cudaMemcpy(&copied, pointer, sizeof(copied), cudaMemcpyDeviceToHost));
  require(copied.fieldCount == state(v).fieldCount
          && copied.fields == state(v).fields,
      "host/device publication mismatch");
  require(copied.hasConstantTransmissionLog
          == (copied.bindings.transmissionColor.field == -1),
      "constant log eligibility mismatch");
  if (copied.hasConstantTransmissionLog)
    for (int c = 0; c < 3; ++c)
      require(copied.constantTransmissionLog[c]
                  == ref::negativeLog(
                      copied.bindings.transmissionColor.scale[c])
              && copied.constantTransmissionLog[c]
                  == state(v).constantTransmissionLog[c],
          "stale or incorrectly rounded transmission log");
  DeviceBuffer output;
  output.reserve(sizeof(Result));
  samplePrincipledState<<<1, 1>>>(pointer, p, output.ptrAs<Result>());
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
  Result result;
  checked(cudaMemcpy(
      &result, output.ptr(), sizeof(result), cudaMemcpyDeviceToHost));
  require(
      std::memcmp(
          &result.coefficients, &result.uncached, sizeof(result.coefficients))
          == 0,
      "cached state differs from uncached GPU evaluation");
  for (int c = 0; c < 3; ++c)
    require(ref::extinction(result.coefficients, c)
            <= ref::majorant(copied.opticalBounds),
        "committed field sample exceeds volume bound");
  return result;
}
static void near(double a, double b)
{
  require(std::isfinite(a) && std::abs(a - b) <= 1e-10 * (1. + std::abs(b)),
      "numeric payload oracle");
}

static void defaultsAndParameters(Objects &o)
{
  auto v = volume(o, {vec3(0), vec3(1)});
  require(state(v).previewFloatEligible,
      "ordinary constants did not select FP32 preview");
  const auto &b = state(v).bindings;
  require(state(v).fieldCount == 0 && state(v).fields == nullptr,
      "constants require dummy fields");
  require(b.density.scale == 1 && b.density.outside == 0
          && b.temperature.scale == 1 && b.temperature.outside == 0,
      "scalar defaults");
  require(b.scatteringColor.scale == vec3(1)
          && b.transmissionColor.scale == vec3(.36787944117144233f)
          && b.transmissionDistance.scale == 1
          && b.transmissionDistance.outside == 1,
      "optical defaults");
  require(b.emission.scale == vec3(0) && b.emission.outside == vec3(0)
          && b.blackbodyIntensity.scale == 0
          && b.blackbodyIntensity.outside == 1
          && b.blackbodyTint.scale == vec3(1)
          && b.blackbodyTint.outside == vec3(1),
      "source defaults");
  require(b.anisotropy.scale == 0 && b.secondaryAnisotropy.scale == 0
          && b.secondaryWeight.scale == 0 && b.anisotropy.outside == 1
          && b.secondaryAnisotropy.outside == 1
          && b.secondaryWeight.outside == 1,
      "phase defaults");
  auto initial = sample(v, vec3(.5f));
  for (int c = 0; c < 3; ++c) {
    near(initial.coefficients.scattering[c],
        -std::log(double(.36787944117144233f)));
    require(initial.coefficients.absorption[c] == 0
            && initial.coefficients.emission[c] == 0,
        "default medium glows/absorbs");
  }
  const auto outside = sample(v, vec3(2.f));
  require(
      ref::extinction(outside.coefficients, 0) == 0, "medium-domain clipping");
  const float zeroDensity = 0.f;
  const vec3 source(2, 3, 5);
  anariSetParameter(o.d, v, "density", ANARI_FLOAT32, &zeroDensity);
  anariSetParameter(o.d, v, "emission", ANARI_FLOAT32_VEC3, &source);
  anariCommitParameters(o.d, v);
  require(valid(o, v) && ref::majorant(state(v).opticalBounds) == 0.,
      "homogeneous vacuum source rejected");
  const auto sourceOnly = sample(v, vec3(.5f));
  for (int c = 0; c < 3; ++c)
    require(ref::extinction(sourceOnly.coefficients, c) == 0.
            && sourceOnly.coefficients.emission[c] == source[c],
        "homogeneous zero-extinction source lost");

  struct ScalarRole
  {
    const char *name;
    FieldBindingGPUData<float> PrincipledVolumeBindingsGPUData::*member;
    float value;
  };
  const ScalarRole scalars[] = {
      {"density", &PrincipledVolumeBindingsGPUData::density, 2.f},
      {"transmissionDistance",
          &PrincipledVolumeBindingsGPUData::transmissionDistance,
          1.5f},
      {"anisotropy", &PrincipledVolumeBindingsGPUData::anisotropy, .2f},
      {"secondaryAnisotropy",
          &PrincipledVolumeBindingsGPUData::secondaryAnisotropy,
          -.3f},
      {"secondaryWeight",
          &PrincipledVolumeBindingsGPUData::secondaryWeight,
          .4f},
      {"temperature", &PrincipledVolumeBindingsGPUData::temperature, 2500.f},
      {"blackbodyIntensity",
          &PrincipledVolumeBindingsGPUData::blackbodyIntensity,
          2.f}};
  struct RGBRole
  {
    const char *name;
    FieldBindingGPUData<vec3> PrincipledVolumeBindingsGPUData::*member;
    vec3 value;
  };
  const RGBRole colors[] = {
      {"scatteringColor",
          &PrincipledVolumeBindingsGPUData::scatteringColor,
          {.25f, .5f, .75f}},
      {"transmissionColor",
          &PrincipledVolumeBindingsGPUData::transmissionColor,
          {.5f, .25f, .125f}},
      {"emission", &PrincipledVolumeBindingsGPUData::emission, {1, 2, 3}},
      {"blackbodyTint",
          &PrincipledVolumeBindingsGPUData::blackbodyTint,
          {.1f, .2f, .3f}}};
  const float scalarOutside = -2.f;
  const vec3 rgbOutside(2, 3, 4);
  for (const auto &role : scalars) {
    anariSetParameter(o.d, v, role.name, ANARI_FLOAT32, &role.value);
    anariSetParameter(o.d,
        v,
        (std::string(role.name) + ".outside").c_str(),
        ANARI_FLOAT32,
        &scalarOutside);
  }
  for (const auto &role : colors) {
    anariSetParameter(o.d, v, role.name, ANARI_FLOAT32_VEC3, &role.value);
    anariSetParameter(o.d,
        v,
        (std::string(role.name) + ".outside").c_str(),
        ANARI_FLOAT32_VEC3,
        &rgbOutside);
  }
  anariCommitParameters(o.d, v);
  const int savedRound = std::fegetround();
  std::fesetround(FE_DOWNWARD);
  const bool ready = valid(o, v);
  const int after = std::fegetround();
  std::fesetround(savedRound);
  require(ready && after == FE_DOWNWARD, "preparation/rounding restoration");
  for (const auto &role : scalars) {
    const auto &binding = state(v).bindings.*(role.member);
    require(binding.scale == role.value && binding.outside == scalarOutside
            && binding.field == -1,
        "scalar parameter publication");
  }
  for (const auto &role : colors) {
    const auto &binding = state(v).bindings.*(role.member);
    require(binding.scale == role.value && binding.outside == rgbOutside
            && binding.field == -1,
        "RGB parameter publication");
  }
  const auto result = sample(v, vec3(.5f));
  require(result.phase.anisotropy == .2f
          && result.phase.secondaryAnisotropy == -.3f
          && result.phase.secondaryWeight == .4f,
      "phase binding publication");
  for (int c = 0; c < 3; ++c) {
    const double e = 2. * -std::log(double(colors[1].value[c])) / 1.5;
    const double a = e * (1. - double(colors[0].value[c]));
    near(result.coefficients.absorption[c], a);
    near(result.coefficients.emission[c],
        double(c + 1) + a * 2. * double(colors[3].value[c]) * double(c + 1));
  }
}

static void wideThermalCoupling(Objects &o)
{
  auto v = volume(o, {vec3(0), vec3(1)});
  const float density = 1e-37f, intensity = 1e12f, temperature = 100000.f;
  const vec3 tint(1e12f), albedo(.9f);
  anariSetParameter(o.d, v, "density", ANARI_FLOAT32, &density);
  anariSetParameter(o.d, v, "blackbodyIntensity", ANARI_FLOAT32, &intensity);
  anariSetParameter(o.d, v, "temperature", ANARI_FLOAT32, &temperature);
  anariSetParameter(o.d, v, "blackbodyTint", ANARI_FLOAT32_VEC3, &tint);
  anariSetParameter(o.d, v, "scatteringColor", ANARI_FLOAT32_VEC3, &albedo);
  anariCommitParameters(o.d, v);
  require(valid(o, v) && !state(v).previewFloatEligible,
      "large thermal gain with tiny absorption requires wide coupling");
  const auto result = sample(v, vec3(.5f));
  require(result.coefficients.emission[0] > 0.,
      "wide thermal coupling lost source");
}

static void fieldsAndRecovery(Objects &o)
{
  auto scalarArray = o.keep(
      anariNewArray3D(o.d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
  auto fill = [&](float x) {
    auto *p = static_cast<float *>(anariMapArray(o.d, scalarArray));
    std::fill(p, p + 8, x);
    anariUnmapArray(o.d, scalarArray);
  };
  fill(2.f);
  auto rgbArray = o.keep(anariNewArray3D(
      o.d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC3, 2, 2, 2));
  auto *rgb = static_cast<vec3 *>(anariMapArray(o.d, rgbArray));
  std::fill(rgb, rgb + 8, vec3(.5f, .25f, .125f));
  anariUnmapArray(o.d, rgbArray);
  auto scalar = o.keep(anariNewSpatialField(o.d, "structuredRegular"));
  auto color = o.keep(anariNewSpatialField(o.d, "structuredRegular"));
  anariSetParameter(o.d, scalar, "data", ANARI_ARRAY3D, &scalarArray);
  anariSetParameter(o.d, color, "data", ANARI_ARRAY3D, &rgbArray);
  anariCommitParameters(o.d, scalar);
  anariCommitParameters(o.d, color);
  auto v = volume(o, {vec3(-1), vec3(2)});
  const float density = 2.f;
  const vec3 white(1), half(.5f), two(2), outside(3, 4, 5);
  anariSetParameter(o.d, v, "density", ANARI_FLOAT32, &density);
  anariSetParameter(o.d, v, "density.field", ANARI_SPATIAL_FIELD, &scalar);
  anariSetParameter(o.d, v, "temperature.field", ANARI_SPATIAL_FIELD, &scalar);
  anariSetParameter(o.d, v, "transmissionColor", ANARI_FLOAT32_VEC3, &white);
  anariSetParameter(
      o.d, v, "transmissionColor.field", ANARI_SPATIAL_FIELD, &color);
  anariSetParameter(o.d, v, "scatteringColor", ANARI_FLOAT32_VEC3, &half);
  anariSetParameter(o.d, v, "emission", ANARI_FLOAT32_VEC3, &two);
  anariSetParameter(o.d, v, "emission.field", ANARI_SPATIAL_FIELD, &color);
  anariSetParameter(o.d, v, "emission.outside", ANARI_FLOAT32_VEC3, &outside);
  anariCommitParameters(o.d, v);
  require(valid(o, v), "field-backed preparation");
  require(state(v).previewFloatEligible,
      "ordinary fields did not select FP32 preview");
  require(state(v).fieldCount == 2
          && state(v).bindings.density.field
              == state(v).bindings.temperature.field
          && state(v).bindings.transmissionColor.field
              == state(v).bindings.emission.field,
      "compact unique field table");
  auto result = sample(v, vec3(.5f));
  for (int c = 0; c < 3; ++c) {
    near(result.coefficients.scattering[c], 2. * std::log(2.) * (c + 1));
    near(result.coefficients.emission[c], c == 0 ? 1. : c == 1 ? .5 : .25);
  }
  const auto vacuum = sample(v, vec3(1.5f));
  for (int c = 0; c < 3; ++c) {
    require(
        ref::extinction(vacuum.coefficients, c) == 0, "outside optical values");
    near(vacuum.coefficients.emission[c], 2. * outside[c]);
  }
  anariUnsetParameter(o.d, v, "temperature.field");
  anariCommitParameters(o.d, v);
  require(valid(o, v), "remove shared field role");
  const double prior = ref::majorant(state(v).opticalBounds);
  const auto densityRevision = state(v).densityRevision;
  fill(5.f);
  require(valid(o, v), "observer update without volume commit");
  require(!(state(v).densityRevision == densityRevision),
      "density cache identity did not follow field update");
  require(ref::majorant(state(v).opticalBounds) > prior,
      "bound did not follow field update");
  result = sample(v, vec3(.5f));
  for (int c = 0; c < 3; ++c)
    near(result.coefficients.scattering[c], 5. * std::log(2.) * (c + 1));

  anariSetParameter(o.d, color, "data", ANARI_ARRAY3D, &scalarArray);
  anariCommitParameters(o.d, color);
  require(!valid(o, v), "arity update not invalidated");
  require(reinterpret_cast<PrincipledVolume *>(v)->committedData() == nullptr
          && reinterpret_cast<Object *>(v)->deviceData() == nullptr,
      "invalid state retained successful payload");
  anariSetParameter(o.d, color, "data", ANARI_ARRAY3D, &rgbArray);
  anariCommitParameters(o.d, color);
  require(valid(o, v), "observer-only recovery");
  sample(v, vec3(.5f));
  const double sharedBound = ref::majorant(state(v).opticalBounds);
  auto tf = o.keep(anariNewVolume(o.d, "transferFunction1D"));
  anariSetParameter(o.d, tf, "value", ANARI_SPATIAL_FIELD, &scalar);
  const float opacity = .1f;
  anariSetParameter(o.d, tf, "opacity", ANARI_FLOAT32, &opacity);
  anariCommitParameters(o.d, tf);
  require(valid(o, tf) && valid(o, v), "shared TF/principled field");
  require(ref::majorant(state(v).opticalBounds) == sharedBound,
      "TF classification changed principled optical bounds");
  result = sample(v, vec3(.5f));
  for (int c = 0; c < 3; ++c)
    near(result.coefficients.scattering[c], 5. * std::log(2.) * (c + 1));
  anariUnsetParameter(o.d, v, "density.field");
  anariCommitParameters(o.d, v);
  require(valid(o, v) && state(v).fieldCount == 1
          && state(v).bindings.transmissionColor.field == 0
          && state(v).bindings.emission.field == 0,
      "compact table after removing first dependency");
  const auto finalized =
      reinterpret_cast<PrincipledVolume *>(v)->lastFinalized();
  fill(7.f);
  require(valid(o, tf) && valid(o, v),
      "remaining TF observer after principled unbind");
  require(reinterpret_cast<PrincipledVolume *>(v)->lastFinalized() == finalized,
      "unused field still notified principled consumer");
  result = sample(v, vec3(.5f));
  for (int c = 0; c < 3; ++c)
    near(result.coefficients.scattering[c], std::log(2.) * (c + 1));

  // Field -> constant -> field -> constant must never reuse the other path's
  // cache, including observer-only array updates and rejected constants.
  anariUnsetParameter(o.d, v, "transmissionColor.field");
  anariCommitParameters(o.d, v);
  require(valid(o, v), "unbind transmission color");
  result = sample(v, vec3(.5f));
  require(ref::extinction(result.coefficients, 0) == 0.,
      "unbound white transmission retained field log");
  const vec3 constant(.4f, .6f, .8f);
  anariSetParameter(o.d, v, "transmissionColor", ANARI_FLOAT32_VEC3, &constant);
  anariCommitParameters(o.d, v);
  require(valid(o, v), "constant transmission update");
  sample(v, vec3(.5f));
  anariSetParameter(
      o.d, v, "transmissionColor.field", ANARI_SPATIAL_FIELD, &color);
  anariCommitParameters(o.d, v);
  require(valid(o, v), "rebind transmission color");
  sample(v, vec3(.5f));
  rgb = static_cast<vec3 *>(anariMapArray(o.d, rgbArray));
  std::fill(rgb, rgb + 8, vec3(.25f));
  anariUnmapArray(o.d, rgbArray);
  require(valid(o, v), "transmission field observer update");
  sample(v, vec3(.5f));
  anariUnsetParameter(o.d, v, "transmissionColor.field");
  const vec3 invalid(0.f);
  anariSetParameter(o.d, v, "transmissionColor", ANARI_FLOAT32_VEC3, &invalid);
  anariCommitParameters(o.d, v);
  require(!valid(o, v)
          && reinterpret_cast<PrincipledVolume *>(v)->committedData()
              == nullptr,
      "invalid constant retained cache");
  anariSetParameter(o.d, v, "transmissionColor", ANARI_FLOAT32_VEC3, &constant);
  anariCommitParameters(o.d, v);
  require(valid(o, v), "constant transmission recovery");
  sample(v, vec3(.5f));
  const float extremeDensity = 1e30f;
  anariSetParameter(o.d, v, "density", ANARI_FLOAT32, &extremeDensity);
  anariCommitParameters(o.d, v);
  require(valid(o, v) && !state(v).previewFloatEligible,
      "wide optical fallback selection");
  sample(v, vec3(.5f));
  anariSetParameter(o.d, v, "density", ANARI_FLOAT32, &density);
  anariCommitParameters(o.d, v);
  require(valid(o, v) && state(v).previewFloatEligible,
      "FP32 eligibility recovery");
  const vec3 tiny(1e-40f), large(1e36f);
  rgb = static_cast<vec3 *>(anariMapArray(o.d, rgbArray));
  std::fill(rgb, rgb + 8, tiny);
  anariUnmapArray(o.d, rgbArray);
  anariSetParameter(
      o.d, v, "transmissionColor.field", ANARI_SPATIAL_FIELD, &color);
  anariSetParameter(o.d, v, "transmissionColor", ANARI_FLOAT32_VEC3, &large);
  anariSetParameter(
      o.d, v, "transmissionColor.outside", ANARI_FLOAT32_VEC3, &tiny);
  anariCommitParameters(o.d, v);
  require(valid(o, v) && !state(v).previewFloatEligible,
      "subnormal operands must retain wide binding evaluation");
  sample(v, vec3(.5f));
}

int main()
try {
  Objects objects;
  objects.d = makeVisRTXDevice(status);
  anariCommitParameters(objects.d, objects.d);
  defaultsAndParameters(objects);
  wideThermalCoupling(objects);
  fieldsAndRecovery(objects);
  require(errors == 0, "unexpected status errors");
  std::puts(
      "principledVolume: committed GPU constants, fields, bounds and recovery pass");
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
