#include "tsd/core/scene/objects/Surface.hpp"
#include "tsd/io/serialization.hpp"

#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/Logging.hpp"

#include <anari/frontend/anari_enums.h>

#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/scope.h>


#include <algorithm>
#include <cstdio>
#include <numeric>

namespace tsd::io {

const auto allObjectsPath = pxr::SdfPath("/AllObjects");
const auto allLightsPath = allObjectsPath.AppendChild(pxr::TfToken("Lights"));
const auto allSurfacesPath = allObjectsPath.AppendChild(pxr::TfToken("Surfaces"));
const auto allMaterialsPath = allObjectsPath.AppendChild(pxr::TfToken("Materials"));
const auto allLayersPath = pxr::SdfPath("/AllLayers");

std::unordered_map<const Object*, pxr::SdfPath> objectToUsdPathMap;

pxr::SdfPath exportAnariSurface(pxr::UsdStageRefPtr& stage,  const Object* object) {
    if (auto it = objectToUsdPathMap.find(object); it != objectToUsdPathMap.end()) {
        return it->second;
    }

    auto* anariGeom = object->parameterValueAsObject<Geometry>(tokens::surface::geometry);
    auto* anariMaterial = object->parameterValueAsObject<Material>(tokens::surface::material);

    pxr::SdfPath objectPath = allSurfacesPath.AppendChild(pxr::TfToken(object->name().c_str()));

    if (!anariGeom) { // && !anariMaterial) {
        return pxr::SdfPath::EmptyPath();
    }

    if (anariGeom->subtype() == "triangle") {
        auto usdGeomMesh = pxr::UsdGeomMesh::Define(stage, objectPath);

        auto triangles = anariGeom->parameterValueAsObject<Array>("vertex.position");
        auto points = triangles->dataAs<pxr::GfVec3f>();
        auto pointsCount = triangles->size();
        usdGeomMesh.CreatePointsAttr().Set(pxr::VtArray<pxr::GfVec3f>(points, points + pointsCount));

        std::vector<int> faceVertexCounts;
        std::vector<int> faceVertexIndices;

        if (auto triangleIds = anariGeom->parameterValueAsObject<Array>("primitive.index")) {
            auto triangles = triangleIds->dataAs<pxr::GfVec3i>();
            auto triangleCount = triangleIds->size();

            faceVertexCounts.resize(triangleCount, 3);
            faceVertexIndices.resize(triangleCount);
            std::copy_n((int*)triangles, triangleCount * 3, faceVertexIndices.begin());
        } else {
            // std::generate(begin(faceVertexIndices), end(faceVertexIndices), [n = 0]() mutable { return n++; });
            faceVertexCounts.resize(pointsCount / 3, 3);
            faceVertexIndices.resize(pointsCount);
            std::iota(begin(faceVertexIndices), end(faceVertexIndices), 0);
        }

        usdGeomMesh.CreateFaceVertexCountsAttr().Set(pxr::VtArray<int>(cbegin(faceVertexCounts), cend(faceVertexCounts)));
        usdGeomMesh.CreateFaceVertexIndicesAttr().Set(pxr::VtArray<int>(cbegin(faceVertexIndices), cend(faceVertexIndices)));

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
// #if defined(USD_ENABLED)

  objectToUsdPathMap.clear();

  tsd::core::logStatus("Exporting scene to USD file: %s", filename);

  pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateNew(filename);
  if (!stage) {
    tsd::core::logError("Failed to create USD stage for file: %s", filename);
    return;
  }

  // pxr::UsdPrim allObjects = stage->DefinePrim(pxr::SdfPath("/AllObjects"));
  pxr::UsdPrim allObjects = stage->DefinePrim(pxr::SdfPath("/AllObjects"));
  pxr::UsdPrim allLights = stage->DefinePrim(pxr::SdfPath("/AllObjects/Lights"));
  pxr::UsdPrim allSurfaces = stage->DefinePrim(pxr::SdfPath("/AllObjects/Surfaces"));
  pxr::UsdPrim allMaterials = stage->DefinePrim(pxr::SdfPath("/AllObjects/Materials"));

  pxr::UsdPrim allLayers = stage->DefinePrim(pxr::SdfPath("/AllLayers"));
  pxr::SdfPath currentPath = allLayers.GetPath();

  std::unordered_set<pxr::SdfPath, pxr::SdfPath::Hash> existingNames;

  for (auto l : scene.layers()) {
    if (auto layer = l.second.ptr) {
        int currentLevel = -1;
        

        layer->traverse(layer->root(), [&](const LayerNode& node, int level) {
            fprintf(stderr, "Visiting node %s at level %d\n", node->name().c_str(), level);
            if (level == 0) {
                // special case for root. put scope
                currentPath = currentPath.AppendChild(pxr::TfToken(l.first.c_str()));
                pxr::UsdGeomScope::Define(stage, currentPath);
            } else {
                if (node->isEmpty())
                    return true;
                if (node->isObject()) {
                    auto object = node->getObject();
                    auto name = node->name();
                    if (name.empty())
                    {
                        name = object->name();
                    }
                    if (name.empty()) {
                        fprintf(stderr, "Don't know what to do yet");
                        std::exit(1);
                    }
                    pxr::SdfPath objectPath = currentPath.AppendChild(pxr::TfToken(name.c_str()));
                    if (auto insertPair = existingNames.insert(objectPath); insertPair.second) {
                        switch (object->type()) {
                        case ANARI_SURFACE: {
                            auto meshPath = exportAnariSurface(stage, object);
                            fprintf(stderr, "Exported surface %s to %s\n", name.c_str(), objectPath.GetText());
                            if (meshPath != pxr::SdfPath::EmptyPath()) {
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

  stage->GetPrimAtPath(allObjectsPath).GetPrimStack()[0]->SetSpecifier(pxr::SdfSpecifierOver);

  stage->Save();
  tsd::core::logStatus("...done exporting USD scene to file: %s", filename);

// #else
//   tsd::core::logError("USD export is not enabled.");
// #endif
}

} // namespace tsd::io
