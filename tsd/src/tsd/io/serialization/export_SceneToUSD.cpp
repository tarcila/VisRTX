// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_USD
#define TSD_USE_USD 1
#endif

// tsd_io
#include "tsd/io/serialization.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/scene/objects/Surface.hpp"
#if TSD_USE_USD
// usd
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/scope.h>
#endif
// std
#include <algorithm>
#include <cstdio>
#include <numeric>

namespace tsd::io {

#if TSD_USE_USD
static const auto allObjectsPath = pxr::SdfPath("/AllObjects");
static const auto allLightsPath =
    allObjectsPath.AppendChild(pxr::TfToken("Lights"));
static const auto allSurfacesPath =
    allObjectsPath.AppendChild(pxr::TfToken("Surfaces"));
static const auto allMaterialsPath =
    allObjectsPath.AppendChild(pxr::TfToken("Materials"));
static const auto allLayersPath = pxr::SdfPath("/AllLayers");

static std::unordered_map<const Object *, pxr::SdfPath> objectToUsdPathMap;

static pxr::SdfPath tsdSurfaceToUSD(
    pxr::UsdStageRefPtr &stage, const Object *object)
{
  if (auto it = objectToUsdPathMap.find(object);
      it != objectToUsdPathMap.end()) {
    return it->second;
  }

  auto *anariGeom =
      object->parameterValueAsObject<Geometry>(tokens::surface::geometry);
  auto *anariMaterial =
      object->parameterValueAsObject<Material>(tokens::surface::material);

  pxr::SdfPath objectPath =
      allSurfacesPath.AppendChild(pxr::TfToken(object->name().c_str()));

  if (!anariGeom) { // && !anariMaterial) {
    return pxr::SdfPath::EmptyPath();
  }

  if (anariGeom->subtype() == tokens::geometry::triangle) {
    auto usdGeomMesh = pxr::UsdGeomMesh::Define(stage, objectPath);

    auto triangles =
        anariGeom->parameterValueAsObject<Array>("vertex.position");
    auto points = triangles->dataAs<pxr::GfVec3f>();
    auto pointsCount = triangles->size();
    usdGeomMesh.CreatePointsAttr().Set(
        pxr::VtArray<pxr::GfVec3f>(points, points + pointsCount));

    std::vector<int> faceVertexCounts;
    std::vector<int> faceVertexIndices;

    if (auto triangleIds =
            anariGeom->parameterValueAsObject<Array>("primitive.index")) {
      auto triangles = triangleIds->dataAs<pxr::GfVec3i>();
      auto triangleCount = triangleIds->size();

      faceVertexCounts.resize(triangleCount, 3);
      faceVertexIndices.resize(triangleCount);
      std::copy_n(
          (int *)triangles, triangleCount * 3, faceVertexIndices.begin());
    } else {
      faceVertexCounts.resize(pointsCount / 3, 3);
      faceVertexIndices.resize(pointsCount);
      std::iota(begin(faceVertexIndices), end(faceVertexIndices), 0);
    }

    usdGeomMesh.CreateFaceVertexCountsAttr().Set(
        pxr::VtArray<int>(cbegin(faceVertexCounts), cend(faceVertexCounts)));
    usdGeomMesh.CreateFaceVertexIndicesAttr().Set(
        pxr::VtArray<int>(cbegin(faceVertexIndices), cend(faceVertexIndices)));

  } else {
    objectPath = pxr::SdfPath::EmptyPath();
  }

  if (!objectPath.IsEmpty()) {
    objectToUsdPathMap.insert({object, objectPath});
  }
  return objectPath;
}

void export_SceneToUSD(Scene &scene, const char *filename)
{
  objectToUsdPathMap.clear();

  tsd::core::logStatus("Exporting scene to USD file: %s", filename);

  pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateNew(filename);
  if (!stage) {
    tsd::core::logError("Failed to create USD stage for file: %s", filename);
    return;
  }

  pxr::SdfPath currentPath = allLayersPath;

  std::unordered_set<pxr::SdfPath, pxr::SdfPath::Hash> existingNames;

  for (auto l : scene.layers()) {
    if (auto layer = l.second.ptr) {
      int currentLevel = -1;

      layer->traverse(layer->root(), [&](const LayerNode &node, int level) {
        if (level == 0) {
          // special case for root -- output UsdGeomScope
          currentPath = currentPath.AppendChild(pxr::TfToken(l.first.c_str()));
          pxr::UsdGeomScope::Define(stage, currentPath);
        } else {
          if (node->isEmpty())
            return true;
          if (node->isObject()) {
            auto object = node->getObject();
            auto name = node->name();
            if (name.empty())
              name = object->name();
            if (name.empty()) {
              fprintf(stderr, "Don't know what to do yet");
              std::exit(1);
            }

            pxr::SdfPath objectPath =
                currentPath.AppendChild(pxr::TfToken(name.c_str()));

            if (auto insertPair = existingNames.insert(objectPath);
                insertPair.second) {
              switch (object->type()) {
              case ANARI_SURFACE: {
                if (auto meshPath = tsdSurfaceToUSD(stage, object);
                    meshPath != pxr::SdfPath::EmptyPath()) {
                  auto mesh = pxr::UsdGeomMesh::Define(stage, objectPath);
                  mesh.GetPrim().GetReferences().AddInternalReference(meshPath);
                }
                break;
              }
              default:
                break;
              }
            } else {
              fprintf(stderr, "Collision with %s\n", objectPath.GetText());
            }
          }
        }
        return true;
      });
    }
  }

  stage->GetPrimAtPath(allObjectsPath)
      .GetPrimStack()[0]
      ->SetSpecifier(pxr::SdfSpecifierOver);

  stage->Save();

  tsd::core::logStatus("...done exporting USD scene to file: %s", filename);
}
#else
void export_SceneToUSD(Scene &, const char *)
{
  tsd::core::logError("[export_USD] USD not enabled in TSD build.");
}
#endif

} // namespace tsd::io
