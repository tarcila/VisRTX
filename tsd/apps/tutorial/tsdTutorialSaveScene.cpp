// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd_core
#include <tsd/core/scene/Scene.hpp>
// tsd_io
#include <tsd/io/procedural.hpp>
#include <tsd/io/serialization.hpp>

int main()
{
  tsd::core::Scene scene;
  tsd::io::generate_material_orb(scene, scene.defaultLayer()->root());
  tsd::io::save_Scene(scene, "saved_context.tsd");
  return 0;
}
