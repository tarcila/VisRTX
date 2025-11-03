// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"
#include "tsd/core/ColorMapUtil.hpp"
// std
#include <cmath>

namespace tsd::io {

void generate_volumeSliceTest(Scene &scene, LayerNodeRef location)
{
  if (!location)
    location = scene.defaultLayer()->root();

  location = scene.insertChildTransformNode(location);

  auto root = scene.insertChildNode(location);
  (*root)->name() = "Volume Slice Test";

  

  // Volume dimensions
  constexpr int volumeSize = 512;
  constexpr float voxelSpacing = 1.0f / volumeSize;
  const float3 volumeOrigin(-0.5f, -0.5f, -0.5f);

  // Generate 3D volume data with spheres //
  auto voxelArray = scene.createArray(ANARI_UFIXED8, volumeSize, volumeSize, volumeSize);
  auto *voxels = voxelArray->mapAs<uint8_t>();

  // Define some spheres in normalized volume space [0,1]
  struct Sphere {
    float3 center;
    float radius;
  };

  const Sphere spheres[] = {
    {{0.3f, 0.3f, 0.3f}, 0.15f},
    {{0.7f, 0.7f, 0.3f}, 0.12f},
    {{0.5f, 0.5f, 0.7f}, 0.18f},
    {{0.2f, 0.7f, 0.6f}, 0.10f},
    {{0.8f, 0.3f, 0.7f}, 0.14f},
  };

  // Fill volume with sphere data
  for (int z = 0; z < volumeSize; z++) {
    for (int y = 0; y < volumeSize; y++) {
      for (int x = 0; x < volumeSize; x++) {
        float3 pos(
          x / float(volumeSize),
          y / float(volumeSize),
          z / float(volumeSize)
        );

        float minDist = 1.0f;
        for (const auto &sphere : spheres) {
          float dist = linalg::length(pos - sphere.center) / sphere.radius;
          minDist = std::min(minDist, dist);
        }

        // Inside spheres = 255, outside = 0, with smooth falloff
        float value = 0.0f;
        if (minDist < 1.0f) {
          value = 1.0f - minDist;
        } else if (minDist < 1.2f) {
          // Smooth falloff
          value = (1.2f - minDist) / 0.2f * 0.5f;
        }

        voxels[z * volumeSize * volumeSize + y * volumeSize + x] = 
          static_cast<uint8_t>(value * 255.0f);
      }
    }
  }

  voxelArray->unmap();

  // Create spatial field //
  auto field = scene.createObject<SpatialField>(
    tokens::spatial_field::structuredRegular);
  field->setName("slice_test_field");
  field->setParameter("origin"_t, volumeOrigin);
  field->setParameter("spacing"_t, float3(voxelSpacing, voxelSpacing, voxelSpacing));
  field->setParameterObject("data"_t, *voxelArray);

  // Create TransferFunction1D volume //
  auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
  colorArray->setData(makeDefaultColorMap(256).data());

  auto [volumeNode, volume] = scene.insertNewChildObjectNode<Volume>(
    root, tokens::volume::transferFunction1D);
  volume->setName("slice_test_volume");
  volume->setParameterObject("color"_t, *colorArray);
  volume->setParameterObject("value"_t, *field);

  // Create 3D texture sampler for MDL material //
  auto sampler3D = scene.createObject<Sampler>(tokens::sampler::image3D);
  sampler3D->setName("slice_test_texture3d");
  sampler3D->setParameterObject("image"_t, *voxelArray);
  sampler3D->setParameter("filter"_t, "linear");
  sampler3D->setParameter("wrapMode1"_t, "clampToBorder");
  sampler3D->setParameter("wrapMode2"_t, "clampToBorder");
  sampler3D->setParameter("wrapMode3"_t, "clampToBorder");
  sampler3D->setParameter("borderColor"_t, float4(0.25f, 0.5f, 1.0f, 0.0f));
  
  // Set up transform from world space [-0.5, 0.5] to texture space [0, 1]
  auto inTransform = math::translation_matrix(float3(0.5f, 0.5f, 0.5f));
  sampler3D->setParameter("inTransform"_t, inTransform);

  // Create 1D transfer function sampler (red-green-blue gradient) //
  auto transferFunction1D = scene.createObject<Sampler>(tokens::sampler::image1D);
  transferFunction1D->setName("slice_test_transfer_function");
  
  auto tfColorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
  auto *tfColors = tfColorArray->mapAs<float4>();
  
  // Create red -> green -> blue gradient
  for (int i = 0; i < 256; i++) {
    float t = i / 255.0f;
    if (t < 0.5f) {
      // Red to Green (first half)
      float s = t * 2.0f;
      tfColors[i] = float4(1.0f - s, s, 0.0f, t);
    } else {
      // Green to Blue (second half)
      float s = (t - 0.5f) * 2.0f;
      tfColors[i] = float4(0.0f, 1.0f - s, s, t);
    }
  }
  
  tfColorArray->unmap();
  
  transferFunction1D->setParameterObject("image"_t, *tfColorArray);
  transferFunction1D->setParameter("filter"_t, "linear");
  transferFunction1D->setParameter("wrapMode1"_t, "clampToEdge");

  // Create triangle geometry for the slice //
  auto sliceGeom = scene.createObject<Geometry>(tokens::geometry::triangle);
  sliceGeom->setName("slice");

  // Create a quad in the middle of the volume, perpendicular to Z axis
  auto slicePositions = scene.createArray(ANARI_FLOAT32_VEC3, 6);
  auto *positions = slicePositions->mapAs<float3>();
  
  // Triangle vertices in world space (centered, 1x1 size to match volume)
  positions[0] = float3(-0.5f, -0.5f, 0.0f); // bottom-left
  positions[1] = float3( 0.5f, -0.5f, 0.0f); // bottom-right
  positions[2] = float3( 0.5f,  0.5f, 0.0f); // top-right

  positions[3] = float3(-0.5f,  0.5f, 0.0f); // top-left
  positions[4] = float3(-0.5f,  -0.5f, 0.0f); // bottom-left
  positions[5] = float3( 0.5f,  0.5f, 0.0f); // top-right
  slicePositions->unmap();

  sliceGeom->setParameterObject("vertex.position"_t, *slicePositions);

  // Create MDL material with 3D texture input //
  auto material = scene.createObject<Material>(tokens::material::mdl);
  material->setName("slice_test_mdl_material");
  
  // Use a simple test material that displays the texture
  // This assumes you have a custom MDL material that accepts a texture3D parameter
  // For now, we'll set it up generically - you may need to adjust based on your MDL module
  material->setParameter("source"_t, "::visrtx::slice::slice");
  material->setParameterObject("volume_texture"_t, *sampler3D);
  material->setParameterObject("transfer_function"_t, *transferFunction1D);
  material->setParameter("data_range"_t, float2(0.0f, 1.0f));
  material->setParameter("contrast"_t, 1.0f);

  // Create surface and add to scene //
  auto surface = scene.createSurface("slice_surface", sliceGeom, material);
  scene.insertChildObjectNode(
    scene.insertChildTransformNode(root), surface);

  // Add a simple directional light for better visibility //
  auto [lightNode, light] = scene.insertNewChildObjectNode<Light>(
    root, tokens::light::directional, "slice_test_light");
  light->setParameter("direction"_t, float2(45.0f, 45.0f));
  light->setParameter("irradiance"_t, 1.0f);
}

} // namespace tsd::io
