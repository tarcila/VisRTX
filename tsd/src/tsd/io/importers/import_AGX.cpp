// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/algorithms/computeScalarRange.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <algorithm>
#include <vector>
// agx
#define AGX_READ_IMPL
#include <agx/agx_read.h>

namespace tsd::io {

// Importer Animated Geometry eXchange (AGX) format
//
// See https://github.com/jeffamstutz/agx
//
void import_AGX(Scene &scene, const char *filepath, LayerNodeRef location)
{
  std::string file = fileOf(filepath);
  if (file.empty()) {
    logError("[import_AGX] no file specified from path '%s'", filepath);
    return;
  }

  AGXReader r = agxNewReader(filepath);

  //// Parse header ////

  AGXHeader hdr{};
  if (agxReaderGetHeader(r, &hdr) != 0) {
    logError("[import_AGX] failed to read header");
    agxReleaseReader(r);
    return;
  }

  //// Validate usable file contents ////

  if (hdr.objectType != ANARI_GEOMETRY) {
    logError("[import_AGX] unsupported object type '%s' in file '%s'",
        anari::toString(hdr.objectType),
        filepath);
    agxReleaseReader(r);
    return;
  }

  const char *subtype = agxReaderGetSubtype(r);
  if (!subtype || subtype[0] == '\0') {
    logError("[import_AGX] no subtype specified in file '%s'", filepath);
    agxReleaseReader(r);
    return;
  }

  //// Create TSD objects ////

  auto agx_root = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());
  (*agx_root)->name() = "agx_transform_" + file;

  // geometry

  auto geom = scene.createObject<tsd::core::Geometry>(subtype);

  geom->setName(("agx_geometry_" + file).c_str());

  // geometry constants

  agxReaderResetConstants(r);
  AGXParamView pv{};
  while (true) {
    int rc = agxReaderNextConstant(r, &pv);
    if (rc < 0) {
      logWarning(
          "[import_AGX] error reading constant from file '%s'", filepath);
      break;
    }
    if (rc == 0)
      break;

    if (pv.isArray) {
      auto arr = scene.createArray(pv.elementType, pv.elementCount);
      arr->setData(pv.data);
      geom->setParameterObject(pv.name, *arr);
    } else {
      geom->setParameter(pv.name, anari::DataType(pv.type), pv.data);
    }
  }

  // geometry time steps

  std::vector<Token> timeStepNames;
  std::vector<TimeStepArrays> timeSteps;

  agxReaderResetTimeSteps(r);
  uint32_t stepIndex = 0, paramCount = 0;
  bool firstStep = true;
  while (agxReaderBeginNextTimeStep(r, &stepIndex, &paramCount) == 1) {
    logInfo("[import_AGX] time step %u: %u params", stepIndex, paramCount);
    int paramIdx = 0;
    while (true) {
      int rc = agxReaderNextTimeStepParam(r, &pv);
      if (rc < 0) {
        logError("[import_AGX] error reading step params");
        break;
      }
      if (rc == 0)
        break;

      // TODO: support non-array animations
      if (firstStep && pv.isArray) {
        timeStepNames.push_back(Token(pv.name));
        timeSteps.emplace_back();
      }

      if (pv.isArray) {
        auto arr = scene.createArray(pv.elementType, pv.elementCount);
        arr->setData(pv.data);
        timeSteps[paramIdx++].push_back(arr);
        if (firstStep)
          geom->setParameterObject(pv.name, *arr);
      } else {
        logWarning(
            "[import_AGX] ignoring non-array parameter '%s' in time step",
            pv.name);
      }
    }
    firstStep = false;
  }

  // material

  auto mat = scene.createObject<tsd::core::Material>(
      tsd::core::tokens::material::matte);
  mat->setName("agx_material");
  mat->setParameter("color", tsd::math::float3(0.8f, 0.8f, 0.8f));

  // surface

  auto surface = scene.createSurface("agx_surface", geom, mat);
  scene.insertChildObjectNode(agx_root, surface);

  // animation

  if (!timeSteps.empty()) {
    auto *anim = scene.addAnimation(file.c_str());
    anim->setAsTimeSteps(*geom, timeStepNames, timeSteps);
  }

  //// Cleanup ////

  agxReleaseReader(r);
}

} // namespace tsd::io
