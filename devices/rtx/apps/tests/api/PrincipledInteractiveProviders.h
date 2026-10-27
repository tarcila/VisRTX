// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Included after the public-API Scene fixture and its analytic expect() helper.
#include <nanovdb/tools/CreateNanoGrid.h>
#include <type_traits>

namespace provider_controls {
static int renders = 0;

static void axes(Scene &s, ANARISpatialField field, int cells, bool nonuniform)
{
  for (const char *name : {"coordsX", "coordsY", "coordsZ"}) {
    auto array = s.keep(anariNewArray1D(
        s.d, nullptr, nullptr, nullptr, ANARI_FLOAT32, cells + 1));
    auto *p = static_cast<float *>(anariMapArray(s.d, array));
    for (int i = 0; i <= cells; ++i)
      p[i] = nonuniform ? (i == 0 ? 0.f : (i == 1 ? .25f : 1.f)) : float(i);
    anariUnmapArray(s.d, array);
    s.set(field, name, ANARI_ARRAY1D, array);
  }
}

static void check(Scene &s,
    ANARISpatialField field,
    bool rgb,
    RGB value,
    bool interior = false)
{
  auto v = s.volume({.5f, .5f, .5f},
      {0, 0, 0},
      rgb ? RGB{1, 1, 1} : RGB{2, 4, 8},
      rgb ? 0.f : 1.f);
  s.set(
      v, rgb ? "emission.field" : "density.field", ANARI_SPATIAL_FIELD, field);
  if (interior) {
    s.set(v,
        "bounds",
        ANARI_FLOAT32_BOX3,
        Box{5.25f, 5.25f, 5.25f, 5.75f, 5.75f, 5.75f});
    s.set(s.camera, "position", ANARI_FLOAT32_VEC3, RGB{5.5f, 5.5f, 7.f});
    s.set(s.camera, "height", ANARI_FLOAT32, .1f);
    s.commit(s.camera);
  }
  s.commit(v);
  s.volumes({v});
  const double length = interior ? .5 : 1.;
  if (rgb)
    expect(s.render(),
        {value[0] * length, value[1] * length, value[2] * length},
        0);
  else {
    const double k = double(value[0]) * std::log(2.);
    const double mass = -std::expm1(-k * length),
                 f = k == 0. ? length : mass / k;
    expect(s.render(), {2 * f, 4 * f, 8 * f}, mass);
  }
  ++renders;
}

template <typename T>
static void dense(Scene &s, ANARIDataType type, T value, RGB expected)
{
  auto data =
      s.keep(anariNewArray3D(s.d, nullptr, nullptr, nullptr, type, 2, 2, 2));
  auto *p = static_cast<T *>(anariMapArray(s.d, data));
  std::fill(p, p + 8, value);
  anariUnmapArray(s.d, data);
  for (bool rect : {false, true}) {
    auto field = s.keep(anariNewSpatialField(
        s.d, rect ? "structuredRectilinear" : "structuredRegular"));
    s.set(field, "data", ANARI_ARRAY3D, data);
    s.set(field, "spacing", ANARI_FLOAT32_VEC3, RGB{.5f, .5f, .5f});
    anariSetParameter(s.d, field, "dataCentering", ANARI_STRING, "cell");
    if (rect)
      axes(s, field, 2, true);
    for (const char *filter : {"nearest", "linear"}) {
      anariSetParameter(s.d, field, "filter", ANARI_STRING, filter);
      s.commit(field);
      check(s, field, type == ANARI_FLOAT32_VEC3, expected);
    }
  }
}

template <typename BuildT, typename Handle>
static void sparse(Scene &s, const Handle &grid, bool interior, int width = 0)
{
  constexpr bool rgb = std::is_same_v<BuildT, nanovdb::Vec3f>;
  const auto *host = grid.template grid<BuildT>();
  require(host != nullptr, "NanoVDB fixture type");
  const auto coord = nanovdb::Coord(interior ? 5 : 0);
  if constexpr (std::is_same_v<BuildT, nanovdb::FpN>)
    if (width)
      require(host->tree().root().probeLeaf(coord)->data()->bitWidth() == width,
          "FpN fixture bit width");
  const auto value = host->tree().getValue(coord);
  RGB expected;
  if constexpr (rgb)
    expected = {value[0], value[1], value[2]};
  else {
    expected = {value, value, value};
    if (interior)
      require(
          value > 0.f, "packed ramp must exercise nonzero decoded extinction");
  }
  auto bytes = s.keep(anariNewArray1D(
      s.d, nullptr, nullptr, nullptr, ANARI_UINT8, grid.bufferSize()));
  std::memcpy(anariMapArray(s.d, bytes), grid.data(), grid.bufferSize());
  anariUnmapArray(s.d, bytes);
  for (bool rect : {false, true}) {
    auto field = s.keep(
        anariNewSpatialField(s.d, rect ? "nanovdbRectilinear" : "nanovdb"));
    s.set(field, "data", ANARI_ARRAY1D, bytes);
    anariSetParameter(s.d, field, "dataCentering", ANARI_STRING, "cell");
    if (rect)
      axes(s, field, 8, false);
    for (const char *filter : {"nearest", "linear"}) {
      // Quantized ramp: nearest sampling stays entirely within voxel (5,5,5),
      // whose decoded value is nonzero even for the one-bit FpN case.
      // Constant grids additionally exercise the linear interpolation path.
      if (interior && std::strcmp(filter, "linear") == 0)
        continue;
      anariSetParameter(s.d, field, "filter", ANARI_STRING, filter);
      s.commit(field);
      check(s, field, rgb, expected, interior);
    }
    if (!interior) {
      // Keep the last volume bound to this same field while its snapshot and
      // logical domain change; do not replace the observing volume.
      s.set(field, "roi", ANARI_FLOAT32_BOX3, Box{0, 0, 0, 1, 1, .5f});
      s.commit(field);
      if constexpr (rgb)
        expect(s.render(),
            {expected[0] * .5, expected[1] * .5, expected[2] * .5},
            0);
      else {
        const double k = double(expected[0]) * std::log(2.);
        const double mass = -std::expm1(-.5 * k), f = .5 + mass / k;
        expect(s.render(), {2 * f, 4 * f, 8 * f}, mass);
      }
      ++renders;
      anariUnsetParameter(s.d, field, "roi");
      s.commit(field);
      if constexpr (rgb)
        expect(s.render(), {expected[0], expected[1], expected[2]}, 0);
      else {
        const double k = double(expected[0]) * std::log(2.);
        const double mass = -std::expm1(-k), f = mass / k;
        expect(s.render(), {2 * f, 4 * f, 8 * f}, mass);
      }
      ++renders;
    }
  }
}

template <typename BuildT>
static void constantSparse()
{
  using Value = std::conditional_t<std::is_same_v<BuildT, nanovdb::Vec3f>,
      nanovdb::Vec3f,
      float>;
  Value value;
  if constexpr (std::is_same_v<Value, float>)
    value = .5f;
  else
    value = Value(2.f, 4.f, 8.f);
  nanovdb::tools::build::Grid<Value> source(value);
  auto access = source.getAccessor();
  for (int z = 0; z < 8; ++z)
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        access.setValue(nanovdb::Coord(x, y, z), value);
  auto grid = nanovdb::tools::createNanoGrid<decltype(source), BuildT>(source);
  Scene s;
  sparse<BuildT>(s, grid, false);
}

template <typename BuildT>
static void packedRamp()
{
  nanovdb::tools::build::Grid<float> source(0.f);
  auto access = source.getAccessor();
  for (int z = 0; z < 8; ++z)
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x)
        access.setValue(
            nanovdb::Coord(x, y, z), float(x + 8 * y + 64 * z) / 511.f);
  Scene s;
  if constexpr (std::is_same_v<BuildT, nanovdb::FpN>) {
    for (int width : {1, 2, 4, 8, 16}) {
      auto grid =
          nanovdb::tools::createNanoGrid<decltype(source), BuildT>(source,
              nanovdb::tools::StatsMode::Default,
              nanovdb::CheckMode::Default,
              false,
              0,
              nanovdb::tools::AbsDiff(.51f / float((1u << width) - 1u)));
      sparse<BuildT>(s, grid, true, width);
    }
  } else {
    auto grid =
        nanovdb::tools::createNanoGrid<decltype(source), BuildT>(source);
    sparse<BuildT>(s, grid, true);
  }
}

static void run()
{
  {
    Scene s;
    dense<float>(s, ANARI_FLOAT32, .5f, {.5f, .5f, .5f});
    dense<double>(s, ANARI_FLOAT64, .5, {.5f, .5f, .5f});
    dense<int8_t>(
        s, ANARI_FIXED8, 64, RGB{64.f / 127.f, 64.f / 127.f, 64.f / 127.f});
    dense<uint8_t>(s,
        ANARI_UFIXED8,
        128,
        RGB{128.f / 255.f, 128.f / 255.f, 128.f / 255.f});
    dense<int16_t>(s,
        ANARI_FIXED16,
        16384,
        RGB{16384.f / 32767.f, 16384.f / 32767.f, 16384.f / 32767.f});
    dense<uint16_t>(s,
        ANARI_UFIXED16,
        32768,
        RGB{32768.f / 65535.f, 32768.f / 65535.f, 32768.f / 65535.f});
    dense<RGB>(s, ANARI_FLOAT32_VEC3, RGB{2, 4, 8}, RGB{2, 4, 8});
  }
  constantSparse<float>();
  constantSparse<nanovdb::Vec3f>();
  constantSparse<nanovdb::Fp4>();
  constantSparse<nanovdb::Fp8>();
  constantSparse<nanovdb::Fp16>();
  constantSparse<nanovdb::FpN>();
  packedRamp<nanovdb::Fp4>();
  packedRamp<nanovdb::Fp8>();
  packedRamp<nanovdb::Fp16>();
  packedRamp<nanovdb::FpN>();
  std::printf(
      "Stored-provider OptiX controls: %d analytic renders pass\n", renders);
}
} // namespace provider_controls
