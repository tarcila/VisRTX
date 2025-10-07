// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#ifndef TSD_USE_USD
#define TSD_USE_USD 1
#endif

// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/io/serialization.hpp"
// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/frontend/anari_enums.h>
#include <anari/frontend/type_utility.h>

#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/Material.hpp"
#include "tsd/core/scene/objects/Sampler.hpp"
#include "tsd/core/scene/objects/Surface.hpp"

// anari
#include <anari/frontend/type_utility.h>
// usd
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformOp.h>
// std
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

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

pxr::SdfPath allocateUniquePath(
    pxr::UsdStagePtr stage, const pxr::SdfPath &path)
{
  const auto basePath = path.GetParentPath();
  auto baseName = path.GetName();

  int uniqueIndex = 0;
  auto newPath = pxr::SdfPath::EmptyPath();
  do {
    auto uniqueName = baseName + "_" + std::to_string(uniqueIndex++);
    newPath = basePath.AppendChild(pxr::TfToken(uniqueName.c_str()));
  } while (stage->GetObjectAtPath(newPath));

  return newPath;
}

static pxr::SdfPath tsdSurfaceToUSD(
    pxr::UsdStageRefPtr &stage, const Surface *surface)
{
  if (auto it = objectToUsdPathMap.find(surface);
      it != objectToUsdPathMap.end()) {
    return it->second;
  }

  auto *anariGeom =
      surface->parameterValueAsObject<Geometry>(tokens::surface::geometry);
  auto *anariMaterial =
      surface->parameterValueAsObject<Material>(tokens::surface::material);

  pxr::SdfPath surfacePath =
      allSurfacesPath.AppendChild(pxr::TfToken(surface->name().c_str()));
  surfacePath = allocateUniquePath(stage, surfacePath);

  if (!anariGeom) { // && !anariMaterial) {
    return pxr::SdfPath::EmptyPath();
  }

  if (anariGeom->subtype() == tokens::geometry::triangle) {
    auto usdGeomMesh = pxr::UsdGeomMesh::Define(stage, surfacePath);

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
      faceVertexIndices.resize(triangleCount * 3);
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

  objectToUsdPathMap.insert({surface, surfacePath});

  return surfacePath;
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
      std::stack<math::mat4> transformStack;

      layer->traverse(layer->root(), [&](const LayerNode &node, int level) {
        // Handle depth traversal
        if (level == 0) {
          // special case for root -- output UsdGeomScope
          currentPath = currentPath.AppendChild(pxr::TfToken(l.first.c_str()));
          pxr::UsdGeomScope::Define(stage, currentPath);
          transformStack.push(math::mat4(math::identity));

          return true;
        }

        if (level > currentLevel) {
          for (int i = currentLevel; i < level; i++)
            transformStack.push(transformStack.top());

        } else if (level < currentLevel) {
          for (int i = level; i < currentLevel; i++)
            transformStack.pop();
        }

        currentLevel = level;

        // Handle this node
        if (node->isEmpty())
          return true;
        if (node->isTransform()) {
          auto transform = node->getTransform();
          transformStack.pop();
          transformStack.push(math::mul(transformStack.top(), transform));
        }
        if (node->isObject()) {
          auto object = node->getObject();
          auto name = node->name();
          if (name.empty())
            name = object->name();
          if (name.empty()) {
            fprintf(stderr, "Don't know what to do yet");
            std::exit(1);
          }

          auto objectPath = currentPath.AppendChild(pxr::TfToken(name.c_str()));
          objectPath = allocateUniquePath(
              stage, currentPath.AppendChild(pxr::TfToken(name.c_str())));

          switch (object->type()) {
          case ANARI_SURFACE: {
            if (auto meshPath = tsdSurfaceToUSD(
                    stage, static_cast<const Surface *>(object));
                meshPath != pxr::SdfPath::EmptyPath()) {
              auto mesh = pxr::UsdGeomMesh::Define(stage, objectPath);
              mesh.GetPrim().GetReferences().AddInternalReference(meshPath);
              auto xfm = transformStack.top();
              mesh.AddXformOp(pxr::UsdGeomXformOp::TypeTransform)
                  .Set(pxr::GfMatrix4d(xfm[0][0],
                      xfm[0][1],
                      xfm[0][2],
                      xfm[0][3],
                      xfm[1][0],
                      xfm[1][1],
                      xfm[1][2],
                      xfm[1][3],
                      xfm[2][0],
                      xfm[2][1],
                      xfm[2][2],
                      xfm[2][3],
                      xfm[3][0],
                      xfm[3][1],
                      xfm[3][2],
                      xfm[3][3]));
            }
            break;
          }
          default:
            break;
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
