// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "OmniPbrMaterial.h"

// tsd_core
#include <tsd/core/Logging.hpp>

// pxr
#include <pxr/usd/usdShade/output.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/assetPath.h>

namespace tsd::io::materials {

MaterialRef importOmniPBRMaterial(Scene &scene,
    const pxr::UsdShadeMaterial &usdMaterial,
    const pxr::UsdShadeShader &usdShader,
    const std::string &basePath,
    TextureCache &textureCache)
{ 
  // Create physicallyBased material
  auto mat = scene.createObject<Material>(tokens::material::physicallyBased);
  
  // Set material name
  std::string matName = usdMaterial.GetPrim().GetName().GetString();
  if (matName.empty())
    matName = "OmniPBR_Material";
  mat->setName(matName.c_str());
  
  // Read OmniPBR parameters
  
  // Base color (diffuse) - try texture first, then constant
  std::string diffuseTexPath;
  if (getShaderTextureInput(usdShader, "diffuse_texture", diffuseTexPath)) {
    // Resolve relative path
    std::string resolvedPath = diffuseTexPath;
    if (!resolvedPath.empty() && resolvedPath[0] != '/') {
      resolvedPath = basePath + diffuseTexPath;
    }
    
    auto sampler = importTexture(scene, resolvedPath, textureCache, false);
    if (sampler) {
      mat->setParameterObject("baseColor"_t, *sampler);
    } else {
      logWarning("[import_USD2] Failed to load diffuse texture: %s\n", resolvedPath.c_str());
    }
  } else {
    // Use constant color
    pxr::GfVec3f diffuseColor;
    if (getShaderColorInput(usdShader, "diffuse_color_constant", diffuseColor)) {
      mat->setParameter("baseColor"_t, 
                       tsd::math::float3(diffuseColor[0], diffuseColor[1], diffuseColor[2]));
    }
  }
  
  // Handle emissive with intensity
  pxr::GfVec3f emissiveColor(0, 0, 0);
  float emissiveIntensity = 1.0f;
  bool enableEmission = false;
  
  getShaderColorInput(usdShader, "emissive_color", emissiveColor);
  getShaderFloatInput(usdShader, "emissive_intensity", emissiveIntensity);
  getShaderBoolInput(usdShader, "enable_emission", enableEmission);
  
  if (enableEmission) {
    // Scale emissive color by intensity
    tsd::math::float3 finalEmissive(
        emissiveColor[0] * emissiveIntensity,
        emissiveColor[1] * emissiveIntensity,
        emissiveColor[2] * emissiveIntensity
    );
    mat->setParameter("emissive"_t, finalEmissive);
  }
  
  // Metallic - try texture first, then constant
  std::string metallicTexPath;
  if (getShaderTextureInput(usdShader, "metallic_texture", metallicTexPath)) {
    std::string resolvedPath = metallicTexPath;
    if (!resolvedPath.empty() && resolvedPath[0] != '/') {
      resolvedPath = basePath + metallicTexPath;
    }
    
    auto sampler = importTexture(scene, resolvedPath, textureCache, true);
    if (sampler) {
      mat->setParameterObject("metallic"_t, *sampler);
    } else {
      logWarning("[import_USD2] Failed to load metallic texture: %s\n", resolvedPath.c_str());
    }
  } else {
    float metallic = 0.0f;
    if (getShaderFloatInput(usdShader, "metallic_constant", metallic)) {
      mat->setParameter("metallic"_t, metallic);
    } else {
      mat->setParameter("metallic"_t, 0.0f);
    }
  }
  
  // Roughness - try texture first, then constant
  std::string roughnessTexPath;
  if (getShaderTextureInput(usdShader, "reflectionroughness_texture", roughnessTexPath)) {
    std::string resolvedPath = roughnessTexPath;
    if (!resolvedPath.empty() && resolvedPath[0] != '/') {
      resolvedPath = basePath + roughnessTexPath;
    }
    
    auto sampler = importTexture(scene, resolvedPath, textureCache, true);
    if (sampler) {
      mat->setParameterObject("roughness"_t, *sampler);
    } else {
      logWarning("[import_USD2] Failed to load roughness texture: %s\n", resolvedPath.c_str());
    }
  } else {
    float roughness = 0.5f;  // Default to mid-range roughness
    if (getShaderFloatInput(usdShader, "reflection_roughness_constant", roughness)) {
      mat->setParameter("roughness"_t, roughness);
    } else {
      mat->setParameter("roughness"_t, 0.5f);
    }
  }
  
  // Normal map
  std::string normalTexPath;
  if (getShaderTextureInput(usdShader, "normalmap_texture", normalTexPath)) {
    std::string resolvedPath = normalTexPath;
    if (!resolvedPath.empty() && resolvedPath[0] != '/') {
      resolvedPath = basePath + normalTexPath;
    }
    
    auto sampler = importTexture(scene, resolvedPath, textureCache, true);
    if (sampler) {
      mat->setParameterObject("normal"_t, *sampler);
    } else {
      logWarning("[import_USD2] Failed to load normal texture: %s\n", resolvedPath.c_str());
    }
  }
  
  // Ambient Occlusion map
  std::string aoTexPath;
  if (getShaderTextureInput(usdShader, "ao_texture", aoTexPath)) {
    std::string resolvedPath = aoTexPath;
    if (!resolvedPath.empty() && resolvedPath[0] != '/') {
      resolvedPath = basePath + aoTexPath;
    }
    
    auto sampler = importTexture(scene, resolvedPath, textureCache, true);
    if (sampler) {
      mat->setParameterObject("occlusion"_t, *sampler);
    } else {
      logWarning("[import_USD2] Failed to load occlusion texture: %s\n", resolvedPath.c_str());
    }
  }
  
  // Opacity - try texture first, then constant
  std::string opacityTexPath;
  bool enableOpacity = false;
  getShaderBoolInput(usdShader, "enable_opacity", enableOpacity);
  
  if (enableOpacity) {
    if (getShaderTextureInput(usdShader, "opacity_texture", opacityTexPath)) {
      std::string resolvedPath = opacityTexPath;
      if (!resolvedPath.empty() && resolvedPath[0] != '/') {
        resolvedPath = basePath + opacityTexPath;
      }
      
      auto sampler = importTexture(scene, resolvedPath, textureCache, true);
      if (sampler) {
        mat->setParameterObject("opacity"_t, *sampler);
      } else {
        logWarning("[import_USD2] Failed to load opacity texture: %s\n", resolvedPath.c_str());
      }
    } else {
      float opacityConstant = 1.0f;
      if (getShaderFloatInput(usdShader, "opacity_constant", opacityConstant)) {
        mat->setParameter("opacity"_t, opacityConstant);
      }
    }
    
    // Set alpha mode based on opacity threshold
    float opacityThreshold = 0.0f;
    if (getShaderFloatInput(usdShader, "opacity_threshold", opacityThreshold)) {
      if (opacityThreshold > 0.0f) {
        mat->setParameter("alphaMode"_t, "mask");
        mat->setParameter("alphaCutoff"_t, opacityThreshold);
      } else {
        mat->setParameter("alphaMode"_t, "blend");
      }
    } else {
      // Default to blend mode when opacity is enabled
      mat->setParameter("alphaMode"_t, "blend");
    }
  } else {
    // Fully opaque
    mat->setParameter("alphaMode"_t, "opaque");
  }
  
  // IOR (index of refraction)
  float ior = 1.5f;
  if (getShaderFloatInput(usdShader, "ior_constant", ior)) {
    mat->setParameter("ior"_t, ior);
  }
  
  // Specular level
  float specularLevel = 0.5f;
  if (getShaderFloatInput(usdShader, "specular_level", specularLevel)) {
    mat->setParameter("specular"_t, specularLevel);
  }
  
  logStatus("[import_USD2] Created OmniPBR material: '%s'\n", matName.c_str());
  return mat;
}

} // namespace tsd::io::materials
