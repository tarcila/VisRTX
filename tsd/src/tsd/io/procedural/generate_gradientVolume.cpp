// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/io/procedural.hpp"

namespace tsd::io {

VolumeRef generate_gradientVolume(Scene &scene,
    LayerNodeRef location,
    ArrayRef colorArray,
    ArrayRef opacityArray)
{
  if (!location)
    location = scene.defaultLayer()->root();

  // Generate spatial field //
  auto field = scene.createObject<SpatialField>(
      tokens::spatial_field::structuredRegular);
  field->setName("gradient_field");

  // Generate gradient data
  auto voxelArray = scene.createArray(ANARI_UFIXED8, 64, 64, 64);
  auto *voxels = (uint8_t *)voxelArray->map();
  for (int z = 0; z < 64; z++) {
    for (int y = 0; y < 64; y++) {
      for (int x = 0; x < 64; x++) {
        size_t idx = z * 64 * 64 + y * 64 + x;
        // Create a gradient from 0 to 1 along one axis (e.g., Z)
        voxels[idx] = 255 * float(z) / float(64 - 1);
      }
    }
  }

  voxelArray->unmap();

  field->setParameter("origin"_t, float3(-1, -1, -1));
  field->setParameterObject("data"_t, *voxelArray);

  // Setup volume //

  auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
      location, tokens::volume::transferFunction1D);
  volume->setName("gradient_volume");

  if (!colorArray) {
    colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
    colorArray->setData(makeDefaultColorMap(colorArray->size()).data());
  }

  volume->setParameterObject("color"_t, *colorArray);
  volume->setParameterObject("value"_t, *field);

  return volume;
}

} // namespace tsd::io
