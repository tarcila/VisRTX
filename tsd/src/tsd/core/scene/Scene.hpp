// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Animation.hpp"
#include "tsd/core/scene/Layer.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/Geometry.hpp"
#include "tsd/core/scene/objects/Light.hpp"
#include "tsd/core/scene/objects/Material.hpp"
#include "tsd/core/scene/objects/Sampler.hpp"
#include "tsd/core/scene/objects/SpatialField.hpp"
#include "tsd/core/scene/objects/Surface.hpp"
#include "tsd/core/scene/objects/Volume.hpp"
// std
#include <memory>
#include <type_traits>
#include <utility>

namespace tsd::core {
struct Scene;
} // namespace tsd::core

namespace tsd::io {
void save_Scene(core::Scene &scene, core::DataNode &root);
void load_Scene(core::Scene &scene, core::DataNode &root);
} // namespace tsd::io

namespace tsd::core {

struct BaseUpdateDelegate;

struct ObjectDatabase
{
  IndexedVector<Array> array;
  IndexedVector<Surface> surface;
  IndexedVector<Geometry> geometry;
  IndexedVector<Material> material;
  IndexedVector<Sampler> sampler;
  IndexedVector<Volume> volume;
  IndexedVector<SpatialField> field;
  IndexedVector<Light> light;

  // Not copyable or moveable //
  ObjectDatabase() = default;
  ObjectDatabase(const ObjectDatabase &) = delete;
  ObjectDatabase(ObjectDatabase &&) = delete;
  ObjectDatabase &operator=(const ObjectDatabase &) = delete;
  ObjectDatabase &operator=(ObjectDatabase &&) = delete;
  //////////////////////////////
};

std::string objectDBInfo(const ObjectDatabase &db);

///////////////////////////////////////////////////////////////////////////////
// Main TSD Scene /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// clang-format off
using LayerPtr = std::shared_ptr<Layer>;
struct LayerState { LayerPtr ptr; bool active{true}; };
using LayerMap = FlatMap<Token, LayerState>;
// clang-format on

struct Scene
{
  Scene();
  ~Scene();

  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  Scene(Scene &&) = delete;
  Scene &operator=(Scene &&) = delete;

  MaterialRef defaultMaterial() const;
  Layer *defaultLayer() const;

  /////////////////////////////
  // Flat object collections //
  /////////////////////////////

  template <typename T>
  IndexedVectorRef<T> createObject();
  template <typename T>
  IndexedVectorRef<T> createObject(Token subtype);
  ArrayRef createArray(anari::DataType type,
      size_t items0,
      size_t items1 = 0,
      size_t items2 = 0);
  ArrayRef createArrayCUDA(anari::DataType type,
      size_t items0,
      size_t items1 = 0,
      size_t items2 = 0);
  SurfaceRef createSurface(const char *name, GeometryRef g, MaterialRef m = {});

  template <typename T>
  IndexedVectorRef<T> getObject(size_t i) const;
  Object *getObject(const Any &a) const;
  Object *getObject(anari::DataType type, size_t i) const;
  size_t numberOfObjects(anari::DataType type) const;

  void removeObject(const Object *o);
  void removeObject(const Any &o);
  void removeAllObjects();

  BaseUpdateDelegate *updateDelegate() const;
  void setUpdateDelegate(BaseUpdateDelegate *ud);

  const ObjectDatabase &objectDB() const;

  ///////////////////////////////////////////////////////
  // Instanced objects (surfaces, volumes, and lights) //
  ///////////////////////////////////////////////////////

  // Layers //

  const LayerMap &layers() const;
  size_t numberOfLayers() const;
  Layer *layer(Token name) const;
  Layer *layer(size_t i) const;

  Layer *addLayer(Token name);

  bool layerIsActive(Token name) const;
  void setLayerActive(const Layer *layer, bool active);
  void setLayerActive(Token name, bool active);
  void setOnlyLayerActive(Token name);
  void setAllLayersActive();
  size_t numberOfActiveLayers() const;
  std::vector<const Layer *> getActiveLayers() const;

  void removeLayer(Token name);
  void removeLayer(const Layer *layer);

  // Insert nodes //

  LayerNodeRef insertChildNode(LayerNodeRef parent, const char *name = "");
  LayerNodeRef insertChildTransformNode(LayerNodeRef parent,
      mat4 xfm = mat4(tsd::math::identity),
      const char *name = "");
  template <typename T>
  LayerNodeRef insertChildObjectNode(
      LayerNodeRef parent, IndexedVectorRef<T> obj, const char *name = "");
  LayerNodeRef insertChildObjectNode(LayerNodeRef parent,
      anari::DataType type,
      size_t idx,
      const char *name = "");

  // NOTE: convenience to create an object _and_ insert it into the tree
  template <typename T>
  using AddedObject = std::pair<LayerNodeRef, IndexedVectorRef<T>>;
  template <typename T>
  AddedObject<T> insertNewChildObjectNode(
      LayerNodeRef parent, Token subtype, const char *name = "");

  // Remove nodes //

  void removeInstancedObject(
      LayerNodeRef obj, bool deleteReferencedObjects = false);

  // Indicate changes occurred //

  void signalLayerChange(const Layer *l);
  void signalActiveLayersChanged();
  void signalObjectParameterUseCountZero(const Object *obj);
  void signalObjectLayerUseCountZero(const Object *obj);

  ////////////////
  // Animations //
  ////////////////

  template <typename T>
  T *addAnimation(const char *name = "");
  size_t numberOfAnimations() const;
  Animation *animation(size_t i) const;
  void removeAnimation(Animation *a);
  void removeAllAnimations();

  void setAnimationTime(float time /* 0.f - 1.f */);
  float getAnimationTime() const;

  void setAnimationIncrement(float increment);
  float getAnimationIncrement() const;
  void incrementAnimationTime();

  ////////////////////////
  // Cleanup operations //
  ////////////////////////

  void removeUnusedObjects();
  void defragmentObjectStorage();
  void cleanupScene(); // remove unused + defragment

 private:
  void removeAllSecondaryLayers();

  friend void ::tsd::io::save_Scene(Scene &scene, core::DataNode &root);
  friend void ::tsd::io::load_Scene(Scene &scene, core::DataNode &root);

  template <typename OBJ_T>
  IndexedVectorRef<OBJ_T> createObjectImpl(
      IndexedVector<OBJ_T> &iv, Token subtype);
  template <typename OBJ_T>
  IndexedVectorRef<OBJ_T> createObjectImpl(IndexedVector<OBJ_T> &iv);

  ArrayRef createArrayImpl(anari::DataType type,
      size_t items0,
      size_t items1,
      size_t items2,
      Array::MemoryKind kind);

  ObjectDatabase m_db;
  BaseUpdateDelegate *m_updateDelegate{nullptr};
  LayerMap m_layers;
  size_t m_numActiveLayers{0};
  struct AnimationData
  {
    float incrementSize{0.01f};
    float time{0.f};
    std::vector<std::unique_ptr<Animation>> objects;
  } m_animations;
};

using GeometryTimeSeries = AnimatedTimeSeries<Geometry>;
using SpatialFieldTimeSeries = AnimatedTimeSeries<SpatialField>;

///////////////////////////////////////////////////////////////////////////////
// Inlined definitions ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Scene //

template <typename T>
inline IndexedVectorRef<T> Scene::createObject()
{
  static_assert(std::is_base_of<Object, T>::value,
      "Scene::createObject<> can only create tsd::Object subclasses");
  static_assert(!std::is_same<T, Array>::value,
      "Use Scene::createArray() to create tsd::Array objects");
  return {};
}

template <>
inline SurfaceRef Scene::createObject()
{
  return createObjectImpl(m_db.surface);
}

template <typename T>
inline IndexedVectorRef<T> Scene::createObject(Token subtype)
{
  static_assert(std::is_base_of<Object, T>::value,
      "Scene::createObject<> can only create tsd::Object subclasses");
  static_assert(!std::is_same<T, Array>::value,
      "Use Scene::createArray() to create tsd::Array objects");
  return {};
}

template <>
inline GeometryRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.geometry, subtype);
}

template <>
inline MaterialRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.material, subtype);
}

template <>
inline SamplerRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.sampler, subtype);
}

template <>
inline VolumeRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.volume, subtype);
}

template <>
inline SpatialFieldRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.field, subtype);
}

template <>
inline LightRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.light, subtype);
}

template <typename T>
inline IndexedVectorRef<T> Scene::getObject(size_t i) const
{
  static_assert(std::is_base_of<Object, T>::value,
      "Scene::getObject<> can only get tsd::Object subclasses");
  return {};
}

template <>
inline ArrayRef Scene::getObject(size_t i) const
{
  return m_db.array.at(i);
}

template <>
inline GeometryRef Scene::getObject(size_t i) const
{
  return m_db.geometry.at(i);
}

template <>
inline MaterialRef Scene::getObject(size_t i) const
{
  return m_db.material.at(i);
}

template <>
inline SamplerRef Scene::getObject(size_t i) const
{
  return m_db.sampler.at(i);
}

template <>
inline VolumeRef Scene::getObject(size_t i) const
{
  return m_db.volume.at(i);
}

template <>
inline SpatialFieldRef Scene::getObject(size_t i) const
{
  return m_db.field.at(i);
}

template <>
inline LightRef Scene::getObject(size_t i) const
{
  return m_db.light.at(i);
}

template <typename T>
T *Scene::addAnimation(const char *name)
{
  static_assert(std::is_base_of<Animation, T>::value,
      "Scene::addAnimation<> can only create tsd::Animation subclasses");
  auto anim = std::make_unique<T>();
  anim->name() = name;
  auto *retval = anim.get();
  m_animations.objects.push_back(std::move(anim));
  return retval;
}

template <typename OBJ_T>
inline IndexedVectorRef<OBJ_T> Scene::createObjectImpl(
    IndexedVector<OBJ_T> &iv, Token subtype)
{
  auto retval = iv.emplace(subtype);
  retval->m_scene = this;
  retval->m_index = retval.index();
  if (m_updateDelegate) {
    retval->setUpdateDelegate(m_updateDelegate);
    m_updateDelegate->signalObjectAdded(retval.data());
  }
  return retval;
}

template <typename OBJ_T>
inline IndexedVectorRef<OBJ_T> Scene::createObjectImpl(IndexedVector<OBJ_T> &iv)
{
  auto retval = iv.emplace();
  retval->m_scene = this;
  retval->m_index = retval.index();
  if (m_updateDelegate) {
    retval->setUpdateDelegate(m_updateDelegate);
    m_updateDelegate->signalObjectAdded(retval.data());
  }
  return retval;
}

template <typename T>
inline LayerNodeRef Scene::insertChildObjectNode(
    LayerNodeRef parent, IndexedVectorRef<T> obj, const char *name)
{
  return insertChildObjectNode(parent, obj->type(), obj->index(), name);
}

template <typename T>
inline Scene::AddedObject<T> Scene::insertNewChildObjectNode(
    LayerNodeRef parent, Token subtype, const char *name)
{
  auto obj = createObject<T>(subtype);
  auto inst = insertChildObjectNode(parent, obj, name);
  return std::make_pair(inst, obj);
}

// Object definitions /////////////////////////////////////////////////////////

template <typename T>
inline T *Object::parameterValueAsObject(Token name) const
{
  static_assert(isObject<T>(),
      "Object::parameterValueAsObject() can only retrieve object values");

  auto *p = parameter(name);
  auto *s = scene();
  if (!p || !s || !p->value().holdsObject())
    return nullptr;
  return (T *)s->getObject(p->value());
}

} // namespace tsd::core
