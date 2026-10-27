#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#ifdef EXPORT_OPENVDB
#include <nanovdb/tools/NanoToOpenVDB.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Interpolation.h>
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef EXPORT_OPENVDB
// NanoVDB inputs use VisRTX's boundary-origin cell convention. Ordinary
// OpenVDB integer indices are sample centers. A copied halo reproduces edge
// clamping over the outer half-cells; the reference shader clips the domain.
template <typename T>
openvdb::GridBase::Ptr referenceGrid(
    const nanovdb::NanoGrid<T> &grid, bool cellCentered)
{
  auto base = nanovdb::tools::nanoToOpenVDB(grid);
  if (!cellCentered)
    return base;
  using Grid = std::conditional_t<std::is_same_v<T, float>,
      openvdb::FloatGrid,
      openvdb::Vec3fGrid>;
  auto result = openvdb::gridPtrCast<Grid>(base);
  const auto shift = grid.indexToWorld(nanovdb::Vec3d(.5))
      - grid.indexToWorld(nanovdb::Vec3d(0));
  result->transform().postTranslate(
      openvdb::Vec3d(shift[0], shift[1], shift[2]));
  const auto box = grid.indexBBox();
  const auto lo = box.min(), hi = box.max();
  auto src = grid.getAccessor();
  auto dst = result->getAccessor();
  auto pad = [&](int x, int y, int z) {
    const T value = src.getValue(nanovdb::Coord(std::clamp(x, lo[0], hi[0]),
        std::clamp(y, lo[1], hi[1]),
        std::clamp(z, lo[2], hi[2])));
    if constexpr (std::is_same_v<T, float>)
      dst.setValue(openvdb::Coord(x, y, z), value);
    else
      dst.setValue(openvdb::Coord(x, y, z),
          openvdb::Vec3f(value[0], value[1], value[2]));
  };
  // Six disjoint faces: O(surface area), including edges and corners.
  for (int x : {lo[0] - 1, hi[0] + 1})
    for (int y = lo[1] - 1; y <= hi[1] + 1; ++y)
      for (int z = lo[2] - 1; z <= hi[2] + 1; ++z)
        pad(x, y, z);
  for (int y : {lo[1] - 1, hi[1] + 1})
    for (int x = lo[0]; x <= hi[0]; ++x)
      for (int z = lo[2] - 1; z <= hi[2] + 1; ++z)
        pad(x, y, z);
  for (int z : {lo[2] - 1, hi[2] + 1})
    for (int x = lo[0]; x <= hi[0]; ++x)
      for (int y = lo[1]; y <= hi[1]; ++y)
        pad(x, y, z);
  return result;
}

template <typename Grid>
void sampleReferenceGrid(const Grid &grid, const openvdb::Vec3d &p)
{
  auto accessor = grid.getConstAccessor();
  openvdb::tools::GridSampler<decltype(accessor), openvdb::tools::BoxSampler>
      sampler(accessor, grid.transform());
  const auto value = sampler.wsSample(p);
  if constexpr (std::is_same_v<typename Grid::ValueType, float>)
    std::cout << '[' << value << "]\n";
  else
    std::cout << '[' << value[0] << ',' << value[1] << ',' << value[2] << "]\n";
}
#endif

// Private interchange: native float32, C-order [x,y,z,channel].
template <typename T>
void convert(const std::string &input,
    const std::string &output,
    const std::string &name,
    int nx,
    int ny,
    int nz,
    double spacing,
    const nanovdb::Vec3d &origin,
    bool cellCentered)
{
  constexpr int channels = std::is_same<T, float>::value ? 1 : 3;
  const size_t count = size_t(nx) * ny * nz;
  std::vector<float> data(count * channels);
  std::ifstream stream(input, std::ios::binary);
  if (!stream.read(
          reinterpret_cast<char *>(data.data()), data.size() * sizeof(float))
      || stream.peek() != EOF)
    throw std::runtime_error("Invalid raw input size");
  for (float v : data)
    if (!std::isfinite(v) || v < 0)
      throw std::runtime_error("Nonfinite or negative field value");

  nanovdb::tools::build::Grid<T> grid(T(0), name);
  grid.setTransform(spacing, origin);
  auto accessor = grid.getAccessor();
  auto value = [&](size_t i) {
    if constexpr (channels == 1)
      return data[i];
    else
      return T(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
  };
  for (int x = 0; x < nx; ++x)
    for (int y = 0; y < ny; ++y)
      for (int z = 0; z < nz; ++z) {
        const T v = value((size_t(x) * ny + y) * nz + z);
        // Active zero corners pin the declared sampling domain, independent
        // of sparse support. Interior zero values remain sparse background.
        const bool corner = (x == 0 || x == nx - 1) && (y == 0 || y == ny - 1)
            && (z == 0 || z == nz - 1);
        if (v != T(0) || corner)
          accessor.setValue(nanovdb::Coord(x, y, z), v);
      }
  auto handle = nanovdb::tools::createNanoGrid(grid);
  nanovdb::io::writeGrid(output, handle);
  auto loaded = nanovdb::io::readGrid<>(output);
  auto *result = loaded.template grid<T>();
  if (!result || result->gridName() != name)
    throw std::runtime_error("Grid type/name round-trip failure");
  const auto worldOrigin = result->indexToWorld(nanovdb::Vec3d(0));
  for (int axis = 0; axis < 3; ++axis)
    if (std::abs(worldOrigin[axis] - origin[axis]) > 1e-12
        || std::abs(result->voxelSize()[axis] - spacing) > 1e-12)
      throw std::runtime_error("Transform round-trip failure");
  if (result->indexBBox().min() != nanovdb::Coord(0)
      || result->indexBBox().max() != nanovdb::Coord(nx - 1, ny - 1, nz - 1))
    throw std::runtime_error("Domain round-trip failure");
  const int dims[3] = {nx, ny, nz};
  for (int axis = 0; axis < 3; ++axis)
    if (std::abs(result->worldBBox().min()[axis] - origin[axis]) > 1e-12
        || std::abs(result->worldBBox().max()[axis]
               - (origin[axis] + dims[axis] * spacing))
            > 1e-12)
      throw std::runtime_error("Full-cell world bounds round-trip failure");
  auto check = result->getAccessor();
  for (int x = 0; x < nx; ++x)
    for (int y = 0; y < ny; ++y)
      for (int z = 0; z < nz; ++z)
        if (check.getValue(nanovdb::Coord(x, y, z))
            != value((size_t(x) * ny + y) * nz + z))
          throw std::runtime_error("Value round-trip failure");
#ifdef EXPORT_OPENVDB
  auto vdb = referenceGrid(*result, cellCentered);
  openvdb::io::File file(output + ".vdb");
  file.write(openvdb::GridPtrVec{vdb});
  file.close();
#endif
  std::cout << name << ": verified " << count << " samples, "
            << handle.bufferSize() << " bytes\n";
}

template <typename T>
void dumpGrid(const nanovdb::GridHandle<> &handle, const std::string &path)
{
  const auto *grid = handle.grid<T>();
  if (!grid)
    throw std::runtime_error("Unexpected grid type");
  const auto box = grid->indexBBox();
  auto acc = grid->getAccessor();
  std::ofstream out(path, std::ios::binary);
  for (int x = box.min()[0]; x <= box.max()[0]; ++x)
    for (int y = box.min()[1]; y <= box.max()[1]; ++y)
      for (int z = box.min()[2]; z <= box.max()[2]; ++z) {
        const T value = acc.getValue(nanovdb::Coord(x, y, z));
        if constexpr (std::is_same<T, float>::value)
          out.write(reinterpret_cast<const char *>(&value), sizeof(float));
        else
          for (int c = 0; c < 3; ++c) {
            const float component = value[c];
            out.write(
                reinterpret_cast<const char *>(&component), sizeof(float));
          }
      }
  if (!out)
    throw std::runtime_error("Failed to write raw grid");
}

template <typename T>
void gridInfo(const nanovdb::NanoGrid<T> &grid)
{
  auto vec = [](const auto &v) {
    std::cout << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
  };
  std::cout << "{\"origin\":";
  vec(grid.indexToWorld(nanovdb::Vec3d(0)));
  std::cout << ",\"voxel_size\":";
  vec(grid.voxelSize());
  std::cout << ",\"index_bounds\":[";
  vec(grid.indexBBox().min());
  std::cout << ',';
  vec(grid.indexBBox().max());
  std::cout << "],\"world_bounds\":[";
  vec(grid.worldBBox().min());
  std::cout << ',';
  vec(grid.worldBBox().max());
  std::cout << "]}\n";
}

int main(int argc, char **argv)
{
  try {
    std::cout << std::setprecision(17);
#ifdef EXPORT_OPENVDB
    openvdb::initialize();
#endif
    if (argc == 3 && std::string(argv[1]) == "--info") {
      auto handle = nanovdb::io::readGrid<>(argv[2]);
      if (auto *grid = handle.grid<float>())
        gridInfo(*grid);
      else if (auto *grid = handle.grid<nanovdb::Vec3f>())
        gridInfo(*grid);
      else
        throw std::runtime_error("Info supports Float and Vec3f only");
      return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--dump") {
      auto handle = nanovdb::io::readGrid<>(argv[2]);
      if (handle.grid<float>())
        dumpGrid<float>(handle, argv[3]);
      else if (handle.grid<nanovdb::Vec3f>())
        dumpGrid<nanovdb::Vec3f>(handle, argv[3]);
      else
        throw std::runtime_error("Dump supports Float and Vec3f only");
      return 0;
    }
#ifdef EXPORT_OPENVDB
    if (argc == 7 && std::string(argv[1]) == "--sample-vdb") {
      openvdb::io::File file(argv[2]);
      file.open();
      auto base = file.readGrid(argv[3]);
      const openvdb::Vec3d p(
          std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]));
      if (base->isType<openvdb::FloatGrid>())
        sampleReferenceGrid(*openvdb::gridPtrCast<openvdb::FloatGrid>(base), p);
      else if (base->isType<openvdb::Vec3fGrid>())
        sampleReferenceGrid(*openvdb::gridPtrCast<openvdb::Vec3fGrid>(base), p);
      else
        throw std::runtime_error("Sample supports Float and Vec3f only");
      return 0;
    }
    if (argc >= 4
        && (std::string(argv[1]) == "--merge-vdb"
            || std::string(argv[1]) == "--merge-vdb-cell")) {
      const bool cellCentered = std::string(argv[1]) == "--merge-vdb-cell";
      openvdb::GridPtrVec grids;
      for (int i = 3; i < argc; ++i) {
        auto handle = nanovdb::io::readGrid<>(argv[i]);
        if (auto *grid = handle.grid<float>())
          grids.push_back(referenceGrid(*grid, cellCentered));
        else if (auto *grid = handle.grid<nanovdb::Vec3f>())
          grids.push_back(referenceGrid(*grid, cellCentered));
        else
          throw std::runtime_error("Export supports Float and Vec3f only");
      }
      openvdb::io::File file(argv[2]);
      file.write(grids);
      file.close();
      return 0;
    }
#endif
    if (argc != 12 && argc != 13)
      throw std::runtime_error(
          "usage: write_nvdb raw output name channels nx ny nz spacing ox oy oz [node|cell]");
    const std::string centering = argc == 13 ? argv[12] : "node";
    if (centering != "node" && centering != "cell")
      throw std::runtime_error("Expected node or cell centering");
    const int channels = std::stoi(argv[4]);
    const int nx = std::stoi(argv[5]), ny = std::stoi(argv[6]),
              nz = std::stoi(argv[7]);
    const double spacing = std::stod(argv[8]);
    const nanovdb::Vec3d origin(
        std::stod(argv[9]), std::stod(argv[10]), std::stod(argv[11]));
    if (nx < 2 || ny < 2 || nz < 2 || !std::isfinite(spacing) || spacing <= 0)
      throw std::runtime_error("Invalid dimensions/spacing");
    for (int axis = 0; axis < 3; ++axis)
      if (!std::isfinite(origin[axis]))
        throw std::runtime_error("Invalid origin");
    if (channels == 1)
      convert<float>(argv[1],
          argv[2],
          argv[3],
          nx,
          ny,
          nz,
          spacing,
          origin,
          centering == "cell");
    else if (channels == 3)
      convert<nanovdb::Vec3f>(argv[1],
          argv[2],
          argv[3],
          nx,
          ny,
          nz,
          spacing,
          origin,
          centering == "cell");
    else
      throw std::runtime_error("Expected 1 or 3 channels");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
