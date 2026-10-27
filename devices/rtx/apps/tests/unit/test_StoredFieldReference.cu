// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "spatial_field/SpatialField.h"
#include "spatial_field/StoredFieldReferenceSampler.h"
#include "utility/DeviceBuffer.h"

#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <anari/anari_cpp.hpp>

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace visrtx;
namespace ref = visrtx::optical_reference;

static int errors = 0;
static int samples = 0;
static void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
static void checkedCuda(cudaError_t error)
{
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "ANARI error: %s\n", message);
    ++errors;
  }
}

struct Objects
{
  ANARIDevice device{};
  std::vector<ANARIObject> handles;
  ~Objects();
  template <typename T>
  T keep(T object);
};
Objects::~Objects()
{
  for (auto it = handles.rbegin(); it != handles.rend(); ++it)
    anariRelease(device, *it);
  anariRelease(device, device);
}
template <typename T>
T Objects::keep(T object)
{
  handles.push_back(reinterpret_cast<ANARIObject>(object));
  return object;
}

static const StoredFieldReference &snapshot(
    ANARIDevice d, ANARISpatialField field)
{
  bool valid = false;
  require(anariGetProperty(
              d, field, "valid", ANARI_BOOL, &valid, sizeof(valid), ANARI_WAIT)
          && valid,
      "field finalization failed");
  // White-box integration seam after PUBLIC creation/commit/update. No public
  // sampling channel or reference-range property is introduced for this test.
  auto *object = reinterpret_cast<SpatialField *>(field);
  const int previousRound = std::fegetround();
  require(std::fesetround(FE_DOWNWARD) == 0, "cannot set test rounding mode");
  const auto *reference = object->referenceField();
  const int retainedRound = std::fegetround();
  std::fesetround(previousRound);
  require(
      retainedRound == FE_DOWNWARD, "snapshot changed caller rounding mode");
  require(reference != nullptr, "reference backend unavailable");
  require(
      object->referenceField() == reference, "reference snapshot not cached");
  return *reference;
}

VISRTX_GLOBAL void referenceFetch(
    StoredFieldReferenceGPUData field, vec3 p, bool rgb, vec3 *out)
{
  StoredFieldReferenceSampler sampler{&field};
  if (rgb)
    *out = sampler.sample<vec3>(0, p);
  else
    *out = vec3(sampler.sample<float>(0, p));
}

struct NoThermal
{
  VISRTX_HOST_DEVICE vec3 operator()(float) const;
};
VISRTX_HOST_DEVICE vec3 NoThermal::operator()(float) const
{
  return vec3(0.f);
}

VISRTX_GLOBAL void referenceOpticsFetch(StoredFieldReferenceGPUData field,
    PrincipledVolumeBindingsGPUData bindings,
    vec3 p,
    ref::Coefficients *out)
{
  StoredFieldReferenceSampler sampler{&field};
  NoThermal blackbody;
  *out = ref::evaluateBindings(bindings, p, sampler, blackbody);
}

static vec3 sample(const StoredFieldReference &reference, vec3 p)
{
  DeviceBuffer result;
  result.reserve(sizeof(vec3));
  const bool rgb = reference.descriptor().valueType == FieldValueType::RGB;
  referenceFetch<<<1, 1>>>(reference.gpuData(), p, rgb, result.ptrAs<vec3>());
  checkedCuda(cudaGetLastError());
  checkedCuda(cudaDeviceSynchronize());
  vec3 value;
  checkedCuda(
      cudaMemcpy(&value, result.ptr(), sizeof(value), cudaMemcpyDeviceToHost));
  if (rgb) {
    ref::RawRangeCertificate<vec3> range;
    require(reference.range(range), "RGB range unavailable");
    require(range.key == reference.descriptor().key,
        "range/sample generation mismatch");
    for (int c = 0; c < 3; ++c)
      require(ref::promote(value[c]) >= ref::promote(range.range.lower[c])
              && ref::promote(value[c]) <= ref::promote(range.range.upper[c]),
          "RGB sample outside hull");
  } else {
    ref::RawRangeCertificate<float> range;
    require(reference.range(range), "scalar range unavailable");
    require(range.key == reference.descriptor().key,
        "range/sample generation mismatch");
    require(ref::promote(value.x) >= ref::promote(range.range.lower)
            && ref::promote(value.x) <= ref::promote(range.range.upper),
        "scalar sample outside hull");
  }
  ++samples;
  return value;
}

static void near(vec3 value, vec3 expected, float tolerance = 2e-3f)
{
  for (int c = 0; c < 3; ++c)
    require(std::isfinite(value[c])
            && std::abs(double(value[c]) - expected[c]) <= tolerance,
        "independent sample oracle");
}

static void axes(Objects &objects,
    ANARISpatialField field,
    bool cell,
    const float *nonuniform = nullptr)
{
  static const float uniform[] = {0.f, 1.f, 2.f};
  const float *values = nonuniform ? nonuniform : uniform;
  for (const char *name : {"coordsX", "coordsY", "coordsZ"}) {
    auto a = objects.keep(anariNewArray1D(
        objects.device, values, nullptr, nullptr, ANARI_FLOAT32, cell ? 3 : 2));
    anariSetParameter(objects.device, field, name, ANARI_ARRAY1D, &a);
  }
}

template <typename T>
static void denseFormat(Objects &objects,
    ANARIDataType type,
    const std::array<T, 8> &values,
    vec3 low,
    vec3 high,
    vec3 midpoint)
{
  const auto d = objects.device;
  // Arrays need owned storage after this call's input leaves scope; copy via
  // map.
  auto array = objects.keep(
      anariNewArray3D(d, nullptr, nullptr, nullptr, type, 2, 2, 2));
  auto *mapped = anariMapArray(d, array);
  std::memcpy(mapped, values.data(), sizeof(values));
  anariUnmapArray(d, array);
  for (bool rect : {false, true})
    for (bool cell : {false, true})
      for (bool linear : {false, true}) {
        auto field = objects.keep(anariNewSpatialField(
            d, rect ? "structuredRectilinear" : "structuredRegular"));
        anariSetParameter(d, field, "data", ANARI_ARRAY3D, &array);
        const char *center = cell ? "cell" : "node";
        const char *filter = linear ? "linear" : "nearest";
        anariSetParameter(d, field, "dataCentering", ANARI_STRING, center);
        anariSetParameter(d, field, "filter", ANARI_STRING, filter);
        if (rect)
          axes(objects, field, cell);
        anariCommitParameters(d, field);
        const auto &reference = snapshot(d, field);
        near(sample(reference, vec3(0.f)), low);
        near(sample(reference, vec3(cell ? 2.f : 1.f)), high);
        if (linear)
          near(sample(reference, vec3(cell ? 1.f : .5f)), midpoint);
      }
}

template <typename BuildT>
static void sparseFormat(Objects &objects)
{
  using Value = std::conditional_t<std::is_same_v<BuildT, nanovdb::Vec3f>,
      nanovdb::Vec3f,
      float>;
  auto value = [](float f) {
    if constexpr (std::is_same_v<Value, float>)
      return f;
    else
      return Value(f, 2.f * f, 3.f * f);
  };
  nanovdb::tools::build::Grid<Value> source(value(10.f));
  source.setTransform(2., nanovdb::Vec3d(-1.));
  auto a = source.getAccessor();
  // Offset bbox tests min-relative mapping. One inactive stored corner remains
  // background10 and must be included in both sampling and global range.
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        if (x || y || z)
          a.setValue(nanovdb::Coord(x + 4, y + 4, z + 4),
              value(float(1 + x + 2 * y + 4 * z)));
  // Inactive tiles at all three levels must enter a global certificate even
  // when they are outside the active index bbox. Active-only stats miss these.
  source.tree().root().template addTile<1>(
      nanovdb::Coord(16, 0, 0), value(-20.f), false);
  source.tree().root().template addTile<2>(
      nanovdb::Coord(128, 0, 0), value(-30.f), false);
  source.tree().root().template addTile<3>(
      nanovdb::Coord(4096, 0, 0), value(40.f), false);
  auto grid = nanovdb::tools::createNanoGrid<decltype(source), BuildT>(source);
  // Independent endpoint oracle uses NanoVDB's own decoder, not the reference
  // decoder. Its float multiply/add can differ by an ULP from controlled
  // decode.
  const auto *host = grid.template grid<BuildT>();
  auto endpoint = [&](int x, int y, int z) {
    const auto v = host->tree().getValue(nanovdb::Coord(x, y, z));
    if constexpr (std::is_same_v<Value, float>)
      return vec3(v);
    else
      return vec3(v[0], v[1], v[2]);
  };
  const vec3 low = endpoint(4, 4, 4), high = endpoint(5, 5, 5);
  vec3 midpoint(0.f);
  for (int z = 4; z < 6; ++z)
    for (int y = 4; y < 6; ++y)
      for (int x = 4; x < 6; ++x)
        midpoint += endpoint(x, y, z) * .125f;
  auto bytes = objects.keep(anariNewArray1D(objects.device,
      nullptr,
      nullptr,
      nullptr,
      ANARI_UINT8,
      grid.bufferSize()));
  std::memcpy(
      anariMapArray(objects.device, bytes), grid.data(), grid.bufferSize());
  anariUnmapArray(objects.device, bytes);
  for (bool rect : {false, true})
    for (bool cell : {false, true})
      for (bool linear : {false, true}) {
        auto field = objects.keep(anariNewSpatialField(
            objects.device, rect ? "nanovdbRectilinear" : "nanovdb"));
        anariSetParameter(objects.device, field, "data", ANARI_ARRAY1D, &bytes);
        const char *center = cell ? "cell" : "node";
        const char *filter = linear ? "linear" : "nearest";
        anariSetParameter(
            objects.device, field, "dataCentering", ANARI_STRING, center);
        anariSetParameter(
            objects.device, field, "filter", ANARI_STRING, filter);
        if (rect)
          axes(objects, field, cell);
        anariCommitParameters(objects.device, field);
        const auto &reference = snapshot(objects.device, field);
        near(sample(reference, vec3(7.f)), low, 1e-5f);
        // Preserve the existing Nano rect-cell LUT/scale convention: its high
        // endpoint reaches the final half-cell for this two-voxel fixture.
        near(sample(reference, vec3(11.f)),
            rect && cell && linear ? midpoint : high,
            1e-5f);
        if (linear)
          near(sample(reference, vec3(9.f)), midpoint, 1e-5f);
        if constexpr (std::is_same_v<BuildT, nanovdb::Vec3f>) {
          ref::RawRangeCertificate<vec3> range;
          require(reference.range(range) && range.range.upper.x >= 40.f
                  && range.range.lower.x <= -30.f,
              "sparse background missing");
        } else {
          ref::RawRangeCertificate<float> range;
          require(reference.range(range) && range.range.upper >= 40.f
                  && range.range.lower <= -30.f,
              "sparse background missing");
        }
      }
}

static void fpnWidths(Objects &objects)
{
  nanovdb::tools::build::Grid<float> source(0.f);
  auto access = source.getAccessor();
  for (int z = 0; z < 8; ++z)
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        access.setValue(
            nanovdb::Coord(x, y, z), float(x + 8 * y + 64 * z) / 511.f);
  for (int width : {1, 2, 4, 8, 16}) {
    // This ramp exercises quantization errors near half a bin. The preceding
    // permitted width has strictly larger error than this fixed oracle budget.
    const float tolerance = .51f / float((1u << width) - 1u);
    auto grid =
        nanovdb::tools::createNanoGrid<decltype(source), nanovdb::FpN>(source,
            nanovdb::tools::StatsMode::Default,
            nanovdb::CheckMode::Default,
            false,
            0,
            nanovdb::tools::AbsDiff(tolerance));
    const auto *host = grid.grid<nanovdb::FpN>();
    require(host->tree().root().probeLeaf(nanovdb::Coord(0))->data()->bitWidth()
            == width,
        "FpN fixture did not exercise intended bit width");
    auto data = objects.keep(anariNewArray1D(objects.device,
        nullptr,
        nullptr,
        nullptr,
        ANARI_UINT8,
        grid.bufferSize()));
    std::memcpy(
        anariMapArray(objects.device, data), grid.data(), grid.bufferSize());
    anariUnmapArray(objects.device, data);
    auto field = objects.keep(anariNewSpatialField(objects.device, "nanovdb"));
    anariSetParameter(objects.device, field, "data", ANARI_ARRAY1D, &data);
    anariSetParameter(objects.device, field, "filter", ANARI_STRING, "nearest");
    anariCommitParameters(objects.device, field);
    const auto &reference = snapshot(objects.device, field);
    for (int i : {0, 3, 7})
      near(sample(reference, vec3(float(i) + .5f)),
          vec3(host->tree().getValue(nanovdb::Coord(i))),
          2e-6f);
  }
}

static void nonuniformAxes(Objects &objects)
{
  const auto d = objects.device;
  auto data = objects.keep(
      anariNewArray3D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
  auto *v = static_cast<float *>(anariMapArray(d, data));
  for (int i = 0; i < 8; ++i)
    v[i] = (i & 1) ? 2.f : 0.f;
  anariUnmapArray(d, data);
  auto field = objects.keep(anariNewSpatialField(d, "structuredRectilinear"));
  anariSetParameter(d, field, "data", ANARI_ARRAY3D, &data);
  anariSetParameter(d, field, "dataCentering", ANARI_STRING, "cell");
  ANARIArray1D xAxis{};
  for (const char *name : {"coordsX", "coordsY", "coordsZ"}) {
    auto axis = objects.keep(
        anariNewArray1D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 3));
    auto *p = static_cast<float *>(anariMapArray(d, axis));
    p[0] = 0.f;
    p[1] = .25f;
    p[2] = 2.f;
    anariUnmapArray(d, axis);
    anariSetParameter(d, field, name, ANARI_ARRAY1D, &axis);
    if (name[6] == 'X')
      xAxis = axis;
  }
  anariCommitParameters(d, field);
  const auto &first = snapshot(d, field);
  const auto key = first.descriptor().key;
  near(sample(first, vec3(.25f, .5f, .5f)), vec3(1.f), 2e-3f);
  near(sample(first, vec3(1.125f, .5f, .5f)), vec3(2.f), 2e-3f);
  auto *p = static_cast<float *>(anariMapArray(d, xAxis));
  p[0] = 0.f;
  p[1] = 1.f;
  p[2] = 2.f;
  anariUnmapArray(d, xAxis);
  const auto &second = snapshot(d, field);
  require(second.descriptor().key.revision != key.revision,
      "axis update did not invalidate reference");
  near(sample(second, vec3(.25f, .5f, .5f)), vec3(0.f));
}

static void doubleRounding(Objects &objects)
{
  const auto d = objects.device;
  const std::array<double, 8> values{1. + 0x1p-24,
      1. + 3. * 0x1p-24,
      0x1p-150,
      3. * 0x1p-150,
      double(std::numeric_limits<float>::max()),
      -double(std::numeric_limits<float>::max()),
      -0.,
      0x1p-1074};
  const uint32_t expected[8] = {0x3f800000u,
      0x3f800002u,
      0u,
      2u,
      0x7f7fffffu,
      0xff7fffffu,
      0x80000000u,
      0u};
  auto data = objects.keep(
      anariNewArray3D(d, nullptr, nullptr, nullptr, ANARI_FLOAT64, 2, 2, 2));
  std::memcpy(anariMapArray(d, data), values.data(), sizeof(values));
  anariUnmapArray(d, data);
  auto field = objects.keep(anariNewSpatialField(d, "structuredRegular"));
  anariSetParameter(d, field, "data", ANARI_ARRAY3D, &data);
  anariSetParameter(d, field, "filter", ANARI_STRING, "nearest");
  anariCommitParameters(d, field);
  const auto &reference = snapshot(d, field);
  for (int i = 0; i < 8; ++i) {
    const vec3 p(float(i & 1), float((i >> 1) & 1), float((i >> 2) & 1));
    require(ref::floatBits(sample(reference, p).x) == expected[i],
        "FLOAT64 canonical RN/underflow/zero decoding");
  }
}

static void deviceSourceAndSubnormal(ANARIDevice d)
{
  DeviceBuffer memory;
  const std::vector<float> values(8, std::numeric_limits<float>::denorm_min());
  memory.upload(values);
  Objects scoped;
  anariRetain(d, d);
  scoped.device = d;
  auto data = scoped.keep(anariNewArray3D(
      d, memory.ptr(), nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
  auto field = scoped.keep(anariNewSpatialField(d, "structuredRegular"));
  anariSetParameter(d, field, "data", ANARI_ARRAY3D, &data);
  anariCommitParameters(d, field);
  const auto &reference = snapshot(d, field);
  require(ref::floatBits(sample(reference, vec3(.5f)).x) == 1,
      "device input/subnormal decode or interpolation lost bits");
  // scoped releases array ownership while its externally supplied memory is
  // still alive; no unsupported writes to unmapped CUDA memory occur.
}

static void revisions(Objects &objects)
{
  const auto d = objects.device;
  auto array = objects.keep(
      anariNewArray3D(d, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2));
  auto fill = [&](float value) {
    auto *p = static_cast<float *>(anariMapArray(d, array));
    std::fill(p, p + 8, value);
    anariUnmapArray(d, array);
  };
  fill(1.f);
  auto field = objects.keep(anariNewSpatialField(d, "structuredRegular"));
  anariSetParameter(d, field, "data", ANARI_ARRAY3D, &array);
  anariCommitParameters(d, field);
  const auto key1 = snapshot(d, field).descriptor().key;
  near(sample(snapshot(d, field), vec3(.5f)), vec3(1.f));
  fill(3.f); // observer-driven finalization, no field recommit
  const auto &changed = snapshot(d, field);
  const auto key2 = changed.descriptor().key;
  require(key2.identity == key1.identity && key2.revision != key1.revision,
      "array update failed to revise snapshot");
  near(sample(changed, vec3(.5f)), vec3(3.f));
  const box3 roi(vec3(.25f), vec3(.75f));
  anariSetParameter(d, field, "roi", ANARI_FLOAT32_BOX3, &roi);
  anariCommitParameters(d, field);
  const auto &clipped = snapshot(d, field);
  require(clipped.descriptor().key.revision != key2.revision
          && clipped.descriptor().domain.lower == roi.lower,
      "ROI revision not published");
  const StoredFieldReference *table[] = {&clipped};
  StoredFieldReferenceProvider provider{table, 1};
  PrincipledVolumeBindingsGPUData b;
  b.bounds = {vec3(0.f), vec3(1.f)};
  b.density.field = 0;
  ref::PreparedVolumeBindings prepared;
  require(ref::prepareVolumeBindings(b, provider, prepared).error
          == ref::PreparationError::None,
      "real provider bridge failed");
  require(ref::matchesPreparedFields(prepared, provider),
      "real snapshot key mismatch");
  DeviceBuffer opticalOutput;
  opticalOutput.reserve(sizeof(ref::Coefficients));
  for (float coordinate : {0.f, .5f, 1.f}) {
    referenceOpticsFetch<<<1, 1>>>(clipped.gpuData(),
        prepared.bindings,
        vec3(coordinate),
        opticalOutput.ptrAs<ref::Coefficients>());
    checkedCuda(cudaGetLastError());
    checkedCuda(cudaDeviceSynchronize());
    ref::Coefficients coefficients;
    checkedCuda(cudaMemcpy(&coefficients,
        opticalOutput.ptr(),
        sizeof(coefficients),
        cudaMemcpyDeviceToHost));
    const double expected = coordinate == .5f
        ? -3. * std::log(double(b.transmissionColor.scale.x))
        : 0.;
    for (int c = 0; c < 3; ++c) {
      require(
          ref::extinction(coefficients, c) <= ref::majorant(prepared.bounds),
          "real stored-binding optics exceed prepared bound");
      require(std::abs(coefficients.scattering[c] - expected) < 1e-11,
          "real stored-binding optics/outside oracle");
    }
  }
  const box3 outsideROI(vec3(2.f), vec3(3.f));
  anariSetParameter(d, field, "roi", ANARI_FLOAT32_BOX3, &outsideROI);
  anariCommitParameters(d, field);
  const auto &empty = snapshot(d, field);
  require(
      !ref::nonemptyDomain(empty.descriptor().domain), "empty ROI descriptor");
  const StoredFieldReference *emptyTable[] = {&empty};
  StoredFieldReferenceProvider emptyProvider{emptyTable, 1};
  ref::PreparedVolumeBindings outsidePrepared;
  require(ref::prepareVolumeBindings(b, emptyProvider, outsidePrepared).error
              == ref::PreparationError::None
          && ref::majorant(outsidePrepared.bounds) == 0.,
      "real empty ROI must use outside everywhere");
  anariUnsetParameter(d, field, "data");
  anariCommitParameters(d, field);
  bool valid = true;
  anariGetProperty(
      d, field, "valid", ANARI_BOOL, &valid, sizeof(valid), ANARI_WAIT);
  require(!valid
          && reinterpret_cast<SpatialField *>(field)->referenceField()
              == nullptr,
      "invalid field retained stale reference");
  anariSetParameter(d, field, "data", ANARI_ARRAY3D, &array);
  anariSetParameter(d, field, "roi", ANARI_FLOAT32_BOX3, &roi);
  anariCommitParameters(d, field);
  const auto &recovered = snapshot(d, field);
  require(recovered.descriptor().key.identity == key1.identity
          && recovered.descriptor().key.revision != key2.revision,
      "invalid-to-valid recovery");
  near(sample(recovered, vec3(.5f)), vec3(3.f));
  const auto recoveredKey = recovered.descriptor().key;
  auto *ramp = static_cast<float *>(anariMapArray(d, array));
  for (int i = 0; i < 8; ++i)
    ramp[i] = float(i + 1);
  anariUnmapArray(d, array);
  const vec3 origin(10.f), spacing(2.f);
  anariUnsetParameter(d, field, "roi");
  anariSetParameter(d, field, "origin", ANARI_FLOAT32_VEC3, &origin);
  anariSetParameter(d, field, "spacing", ANARI_FLOAT32_VEC3, &spacing);
  anariCommitParameters(d, field);
  const auto &mapped = snapshot(d, field);
  require(mapped.descriptor().key.revision != recoveredKey.revision
          && mapped.descriptor().domain.lower == origin
          && mapped.descriptor().domain.upper == vec3(12.f),
      "mapping revision/domain");
  near(sample(mapped, vec3(10.f)), vec3(1.f));
  near(sample(mapped, vec3(11.f)), vec3(4.5f));
  near(sample(mapped, vec3(12.f)), vec3(8.f));
}

int main()
try {
  Objects objects;
  objects.device = makeVisRTXDevice(status);
  anariCommitParameters(objects.device, objects.device);
  denseFormat<float>(objects,
      ANARI_FLOAT32,
      {1, 2, 3, 4, 5, 6, 7, 8},
      vec3(1),
      vec3(8),
      vec3(4.5));
  denseFormat<double>(objects,
      ANARI_FLOAT64,
      {1, 2, 3, 4, 5, 6, 7, 8},
      vec3(1),
      vec3(8),
      vec3(4.5));
  denseFormat<int8_t>(objects,
      ANARI_FIXED8,
      {-128, -127, 0, 0, 0, 0, 127, 127},
      vec3(-1),
      vec3(1),
      vec3(0));
  denseFormat<uint8_t>(objects,
      ANARI_UFIXED8,
      {0, 0, 0, 0, 255, 255, 255, 255},
      vec3(0),
      vec3(1),
      vec3(.5));
  denseFormat<int16_t>(objects,
      ANARI_FIXED16,
      {-32768, -32767, 0, 0, 0, 0, 32767, 32767},
      vec3(-1),
      vec3(1),
      vec3(0));
  denseFormat<uint16_t>(objects,
      ANARI_UFIXED16,
      {0, 0, 0, 0, 65535, 65535, 65535, 65535},
      vec3(0),
      vec3(1),
      vec3(.5));
  std::array<vec3, 8> rgb;
  for (int i = 0; i < 8; ++i)
    rgb[i] = vec3(i + 1, 2 * (i + 1), 3 * (i + 1));
  denseFormat<vec3>(objects,
      ANARI_FLOAT32_VEC3,
      rgb,
      vec3(1, 2, 3),
      vec3(8, 16, 24),
      vec3(4.5, 9, 13.5));
  sparseFormat<float>(objects);
  sparseFormat<nanovdb::Fp4>(objects);
  sparseFormat<nanovdb::Fp8>(objects);
  sparseFormat<nanovdb::Fp16>(objects);
  sparseFormat<nanovdb::FpN>(objects);
  sparseFormat<nanovdb::Vec3f>(objects);
  fpnWidths(objects);
  revisions(objects);
  nonuniformAxes(objects);
  deviceSourceAndSubnormal(objects.device);
  doubleRounding(objects);
  require(errors == 0, "unexpected ANARI errors");
  std::printf(
      "Stored reference: %d samples across four backends, all stored formats and revisions pass\n",
      samples);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
