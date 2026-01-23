// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ParameterHelpers.hpp"
#include "tsd/core/Parameter.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/core/scene/Object.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <fmt/format.h>

namespace tsd::scripting {

void setParameterFromLua(
    core::Object *obj, const std::string &name, sol::object value)
{
  core::Token token(name);

  if (value.is<bool>()) {
    obj->setParameter(token, value.as<bool>());
  } else if (value.is<int>()) {
    obj->setParameter(token, value.as<int>());
  } else if (value.is<double>()) {
    obj->setParameter(token, static_cast<float>(value.as<double>()));
  } else if (value.is<std::string>()) {
    obj->setParameter(token, value.as<std::string>().c_str());
  } else if (value.is<math::float2>()) {
    obj->setParameter(token, value.as<math::float2>());
  } else if (value.is<math::float3>()) {
    obj->setParameter(token, value.as<math::float3>());
  } else if (value.is<math::float4>()) {
    obj->setParameter(token, value.as<math::float4>());
  } else if (value.is<math::mat4>()) {
    obj->setParameter(token, value.as<math::mat4>());
  } else if (value.is<core::ArrayRef>()) {
    auto ref = value.as<core::ArrayRef>();
    if (ref.valid()) {
      obj->setParameterObject(token, *ref.data());
    }
  } else if (value.is<core::SamplerRef>()) {
    auto ref = value.as<core::SamplerRef>();
    if (ref.valid()) {
      obj->setParameterObject(token, *ref.data());
    }
  } else if (value.is<core::GeometryRef>()) {
    auto ref = value.as<core::GeometryRef>();
    if (ref.valid())
      obj->setParameterObject(token, *ref.data());
  } else if (value.is<core::MaterialRef>()) {
    auto ref = value.as<core::MaterialRef>();
    if (ref.valid())
      obj->setParameterObject(token, *ref.data());
  } else if (value.is<core::SpatialFieldRef>()) {
    auto ref = value.as<core::SpatialFieldRef>();
    if (ref.valid())
      obj->setParameterObject(token, *ref.data());
  } else if (value.is<core::VolumeRef>()) {
    auto ref = value.as<core::VolumeRef>();
    if (ref.valid())
      obj->setParameterObject(token, *ref.data());
  } else if (value.is<core::LightRef>()) {
    auto ref = value.as<core::LightRef>();
    if (ref.valid())
      obj->setParameterObject(token, *ref.data());
  } else if (value.is<core::CameraRef>()) {
    auto ref = value.as<core::CameraRef>();
    if (ref.valid())
      obj->setParameterObject(token, *ref.data());
  } else if (value.is<core::SurfaceRef>()) {
    auto ref = value.as<core::SurfaceRef>();
    if (ref.valid())
      obj->setParameterObject(token, *ref.data());
  } else if (value.is<sol::table>()) {
    sol::table t = value.as<sol::table>();
    size_t len = t.size();
    try {
      if (len == 2) {
        obj->setParameter(
            token, math::float2(t[1].get<float>(), t[2].get<float>()));
      } else if (len == 3) {
        obj->setParameter(token,
            math::float3(
                t[1].get<float>(), t[2].get<float>(), t[3].get<float>()));
      } else if (len == 4) {
        obj->setParameter(token,
            math::float4(t[1].get<float>(),
                t[2].get<float>(),
                t[3].get<float>(),
                t[4].get<float>()));
      } else {
        throw std::runtime_error(
            fmt::format("Cannot convert table of size {} for parameter '{}'"
                        ": expected 2, 3, or 4 elements",
                len,
                name));
      }
    } catch (const sol::error &) {
      throw std::runtime_error(
          fmt::format("Failed to convert table to vector for parameter '{}'"
                      ": table elements must be numbers",
              name));
    }
  } else {
    std::string typeName = sol::type_name(value.lua_state(), value.get_type());
    throw std::runtime_error(fmt::format(
        "Unsupported type '{}' for parameter '{}'"
        ": expected bool, number, string, float2/3/4, mat4, ArrayRef, "
        "SamplerRef, GeometryRef, MaterialRef, SpatialFieldRef, "
        "VolumeRef, LightRef, CameraRef, SurfaceRef, "
        "or table of 2-4 numbers",
        typeName,
        name));
  }
}

void applyParameterTable(core::Object *obj, const sol::table &params)
{
  for (auto &[key, value] : params) {
    if (key.is<std::string>())
      setParameterFromLua(obj, key.as<std::string>(), value);
  }
}

sol::object getParameterAsLua(
    sol::state_view lua, const core::Object *obj, const std::string &name)
{
  core::Token token(name);
  const core::Parameter *p = obj->parameter(token);
  if (!p)
    return sol::nil;

  const core::Any &val = p->value();

  if (val.is<bool>())
    return sol::make_object(lua, val.get<bool>());
  if (val.is<int>())
    return sol::make_object(lua, val.get<int>());
  if (val.is<uint32_t>())
    return sol::make_object(lua, val.get<uint32_t>());
  if (val.is<float>())
    return sol::make_object(lua, val.get<float>());
  if (val.is<std::string>())
    return sol::make_object(lua, val.get<std::string>());
  if (val.is<math::float2>())
    return sol::make_object(lua, val.get<math::float2>());
  if (val.is<math::float3>())
    return sol::make_object(lua, val.get<math::float3>());
  if (val.is<math::float4>())
    return sol::make_object(lua, val.get<math::float4>());
  if (val.is<math::mat4>())
    return sol::make_object(lua, val.get<math::mat4>());

  // Handle object references
  if (val.is<core::ArrayRef>())
    return sol::make_object(lua, val.get<core::ArrayRef>());
  if (val.is<core::SamplerRef>())
    return sol::make_object(lua, val.get<core::SamplerRef>());
  if (val.is<core::GeometryRef>())
    return sol::make_object(lua, val.get<core::GeometryRef>());
  if (val.is<core::MaterialRef>())
    return sol::make_object(lua, val.get<core::MaterialRef>());
  if (val.is<core::SpatialFieldRef>())
    return sol::make_object(lua, val.get<core::SpatialFieldRef>());
  if (val.is<core::VolumeRef>())
    return sol::make_object(lua, val.get<core::VolumeRef>());
  if (val.is<core::LightRef>())
    return sol::make_object(lua, val.get<core::LightRef>());
  if (val.is<core::CameraRef>())
    return sol::make_object(lua, val.get<core::CameraRef>());
  if (val.is<core::SurfaceRef>())
    return sol::make_object(lua, val.get<core::SurfaceRef>());

  return sol::nil;
}

void setMetadataFromLua(
    core::Object *obj, const std::string &key, sol::object value)
{
  if (value.is<bool>()) {
    obj->setMetadataValue(key, core::Any(value.as<bool>()));
  } else if (value.is<int>()) {
    obj->setMetadataValue(
        key, core::Any(static_cast<int32_t>(value.as<int>())));
  } else if (value.is<double>()) {
    obj->setMetadataValue(
        key, core::Any(static_cast<float>(value.as<double>())));
  } else if (value.is<std::string>()) {
    auto str = value.as<std::string>();
    obj->setMetadataValue(key, core::Any(ANARI_STRING, str.c_str()));
  } else {
    std::string typeName = sol::type_name(value.lua_state(), value.get_type());
    throw std::runtime_error(
        fmt::format("Unsupported type '{}' for metadata key '{}'"
                    ": expected bool, number, or string",
            typeName,
            key));
  }
}

sol::object getMetadataAsLua(
    sol::state_view lua, const core::Object *obj, const std::string &key)
{
  auto val = obj->getMetadataValue(key);
  if (val.is<bool>())
    return sol::make_object(lua, val.get<bool>());
  if (val.is<int32_t>())
    return sol::make_object(lua, val.get<int32_t>());
  if (val.is<float>())
    return sol::make_object(lua, val.get<float>());
  if (val.type() == ANARI_STRING)
    return sol::make_object(lua, val.getString());
  return sol::nil;
}

} // namespace tsd::scripting
