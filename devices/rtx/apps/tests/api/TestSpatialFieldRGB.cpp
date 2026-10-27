/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <anari/anari_cpp.hpp>

#include <array>
#include <cstdio>
#include <cstring>

static int failures = 0;
static void check(bool ok, const char *message)
{
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR
      || severity == ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "Unexpected error: %s\n", message);
    ++failures;
  }
}

static bool valid(ANARIDevice d, ANARIObject object)
{
  bool result = false;
  check(
      anariGetProperty(
          d, object, "valid", ANARI_BOOL, &result, sizeof(result), ANARI_WAIT),
      "valid property available");
  return result;
}

static ANARIDataType valueType(ANARIDevice d, ANARISpatialField field)
{
  ANARIDataType result = ANARI_UNKNOWN;
  check(anariGetProperty(d,
            field,
            "valueType",
            ANARI_DATA_TYPE,
            &result,
            sizeof(result),
            ANARI_WAIT),
      "valueType property available");
  return result;
}

static void axes(ANARIDevice d, ANARISpatialField field, bool cell)
{
  static const float axis[] = {0.f, 1.f, 2.f};
  for (const char *name : {"coordsX", "coordsY", "coordsZ"}) {
    auto a =
        anariNewArray1D(d, axis, nullptr, nullptr, ANARI_FLOAT32, cell ? 3 : 2);
    anariSetParameter(d, field, name, ANARI_ARRAY1D, &a);
    anariRelease(d, a);
  }
}

static void rejectScalarConsumers(ANARIDevice d, ANARISpatialField field)
{
  auto volume = anariNewVolume(d, "transferFunction1D");
  anariSetParameter(d, volume, "value", ANARI_SPATIAL_FIELD, &field);
  anariCommitParameters(d, volume);
  auto iso = anariNewGeometry(d, "isosurface");
  const float level = 1.f;
  anariSetParameter(d, iso, "field", ANARI_SPATIAL_FIELD, &field);
  anariSetParameter(d, iso, "isovalue", ANARI_FLOAT32, &level);
  anariCommitParameters(d, iso);
  valueType(d, field); // Flush deferred consumer commits too.
  check(!valid(d, volume), "TF1D rejects RGB field");
  check(!valid(d, iso), "isosurface rejects RGB field");
  anariRelease(d, iso);
  anariRelease(d, volume);
}

int main()
{
  auto d = makeVisRTXDevice(status);
  anariCommitParameters(d, d);
  std::array<std::array<float, 3>, 8> rgb;
  for (auto &v : rgb)
    v = {1.f, 2.f, 3.f};
  std::array<float, 8> scalar{};
  for (const char *subtype : {"structuredRegular", "structuredRectilinear"}) {
    auto field = anariNewSpatialField(d, subtype);
    auto array = anariNewArray3D(
        d, rgb.data(), nullptr, nullptr, ANARI_FLOAT32_VEC3, 2, 2, 2);
    anariSetParameter(d, field, "data", ANARI_ARRAY3D, &array);
    anariRelease(d, array);
    if (std::strcmp(subtype, "structuredRectilinear") == 0)
      axes(d, field, false);
    anariCommitParameters(d, field);
    check(valueType(d, field) == ANARI_FLOAT32_VEC3, "dense RGB value type");
    check(valid(d, field), "dense RGB field valid");
    rejectScalarConsumers(d, field);

    // Changing arity and recovering after failed finalization must not leave
    // stale GPU resources or a stale valid bit behind.
    std::array<std::array<float, 4>, 8> rgba{};
    array = anariNewArray3D(
        d, rgba.data(), nullptr, nullptr, ANARI_FLOAT32_VEC4, 2, 2, 2);
    anariSetParameter(d, field, "data", ANARI_ARRAY3D, &array);
    anariRelease(d, array);
    anariCommitParameters(d, field);
    check(valueType(d, field) == ANARI_UNKNOWN && !valid(d, field),
        "RGBA rejected without channel truncation");
    array = anariNewArray3D(
        d, scalar.data(), nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2);
    anariSetParameter(d, field, "data", ANARI_ARRAY3D, &array);
    anariRelease(d, array);
    anariCommitParameters(d, field);
    check(valueType(d, field) == ANARI_FLOAT32 && valid(d, field),
        "scalar recovery");
    auto volume = anariNewVolume(d, "transferFunction1D");
    anariSetParameter(d, volume, "value", ANARI_SPATIAL_FIELD, &field);
    anariCommitParameters(d, volume);
    valueType(d, field);
    check(valid(d, volume), "scalar TF1D initially valid");
    array = anariNewArray3D(
        d, rgb.data(), nullptr, nullptr, ANARI_FLOAT32_VEC3, 2, 2, 2);
    anariSetParameter(d, field, "data", ANARI_ARRAY3D, &array);
    anariRelease(d, array);
    anariCommitParameters(d, field);
    valueType(d, field);
    check(!valid(d, volume),
        "existing TF1D invalidated by scalar-to-RGB recommit");
    array = anariNewArray3D(
        d, scalar.data(), nullptr, nullptr, ANARI_FLOAT32, 2, 2, 2);
    anariSetParameter(d, field, "data", ANARI_ARRAY3D, &array);
    anariRelease(d, array);
    anariCommitParameters(d, field);
    valueType(d, field);
    check(valid(d, volume), "TF1D recovers after RGB-to-scalar recommit");
    if (std::strcmp(subtype, "structuredRectilinear") == 0) {
      const float shortAxis[] = {0.f};
      auto axis =
          anariNewArray1D(d, shortAxis, nullptr, nullptr, ANARI_FLOAT32, 1);
      anariSetParameter(d, field, "coordsX", ANARI_ARRAY1D, &axis);
      anariRelease(d, axis);
      anariCommitParameters(d, field);
      check(valueType(d, field) == ANARI_UNKNOWN && !valid(d, field),
          "axis shape rejected");
      axes(d, field, false);
      anariCommitParameters(d, field);
      check(valueType(d, field) == ANARI_FLOAT32 && valid(d, field),
          "axis shape recovery");
    }
    anariRelease(d, volume);
    anariRelease(d, field);
  }

  nanovdb::tools::build::Grid<nanovdb::Vec3f> grid(nanovdb::Vec3f(0.f));
  auto acc = grid.getAccessor();
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        acc.setValue(nanovdb::Coord(x, y, z), nanovdb::Vec3f(1.f, 2.f, 3.f));
  auto handle = nanovdb::tools::createNanoGrid(grid);
  for (const char *subtype : {"nanovdb", "nanovdbRectilinear"}) {
    auto field = anariNewSpatialField(d, subtype);
    auto array = anariNewArray1D(
        d, handle.data(), nullptr, nullptr, ANARI_UINT8, handle.bufferSize());
    anariSetParameter(d, field, "data", ANARI_ARRAY1D, &array);
    anariRelease(d, array);
    if (std::strcmp(subtype, "nanovdbRectilinear") == 0)
      axes(d, field, true);
    anariCommitParameters(d, field);
    check(valueType(d, field) == ANARI_FLOAT32_VEC3 && valid(d, field),
        "NanoVDB Vec3f valid");
    rejectScalarConsumers(d, field);
    uint8_t shortData[8]{};
    array = anariNewArray1D(d, shortData, nullptr, nullptr, ANARI_UINT8, 8);
    anariSetParameter(d, field, "data", ANARI_ARRAY1D, &array);
    anariRelease(d, array);
    anariCommitParameters(d, field);
    check(valueType(d, field) == ANARI_UNKNOWN && !valid(d, field),
        "short NanoVDB header rejected");
    array = anariNewArray1D(
        d, handle.data(), nullptr, nullptr, ANARI_UINT8, handle.bufferSize());
    anariSetParameter(d, field, "data", ANARI_ARRAY1D, &array);
    anariRelease(d, array);
    anariCommitParameters(d, field);
    check(valueType(d, field) == ANARI_FLOAT32_VEC3 && valid(d, field),
        "NanoVDB recovery");
    anariRelease(d, field);
  }
  anariRelease(d, d);
  std::printf("Spatial field RGB API: %d failures\n", failures);
  return failures ? 1 : 0;
}
