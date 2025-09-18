// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/DataTree.hpp"
#include "tsd/core/FlatMap.hpp"
#include "tsd/core/IndexedVector.hpp"
#include "tsd/core/Parameter.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/core/scene/UpdateDelegate.hpp"
// std
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>

namespace tsd::core {

using namespace literals;
struct Scene;
struct AnariObjectCache;

// Token declarations /////////////////////////////////////////////////////////

namespace tokens {

extern Token none;
extern Token unknown;

} // namespace tokens

// Helper macros //////////////////////////////////////////////////////////////

#define DECLARE_OBJECT_DEFAULT_LIFETIME(TYPE_NAME)                             \
  TYPE_NAME(const TYPE_NAME &) = delete;                                       \
  TYPE_NAME &operator=(const TYPE_NAME &) = delete;                            \
  TYPE_NAME(TYPE_NAME &&) = default;                                           \
  TYPE_NAME &operator=(TYPE_NAME &&) = default;

// Type declarations //////////////////////////////////////////////////////////

struct Object : public ParameterObserver
{
  using ParameterMap = FlatMap<Token, Parameter>;

  Object(anari::DataType type = ANARI_UNKNOWN, Token subtype = tokens::none);
  virtual ~Object();

  // Movable, not copyable
  Object(const Object &) = delete;
  Object &operator=(const Object &) = delete;
  Object(Object &&);
  Object &operator=(Object &&);

  virtual anari::DataType type() const;
  Token subtype() const;
  size_t index() const;
  Scene *scene() const;

  //// Use count tracking (Scene garbage collection) ////

  size_t useCount() const;
  void incUseCount();
  void decUseCount();

  //// Metadata ////

  const std::string &name() const;
  void setName(const char *n);

  Any getMetadataValue(const std::string &name) const;
  void getMetadataArray(const std::string &name,
      anari::DataType *type,
      const void **ptr,
      size_t *size) const;

  void setMetadataValue(const std::string &name, Any v);
  void setMetadataArray(const std::string &name,
      anari::DataType type,
      const void *v,
      size_t numElements);
  void removeMetadata(const std::string &name);

  size_t numMetadata() const;
  const char *getMetadataName(size_t i) const;

  //// Parameters ////

  // Token-based access
  Parameter &addParameter(Token name);
  template <typename T>
  Parameter *setParameter(Token name, T value);
  Parameter *setParameter(Token name, ANARIDataType type, const void *v);
  Parameter *setParameterObject(Token name, const Object &obj);

  const Parameter *parameter(Token name) const;
  Parameter *parameter(Token name);
  template <typename T>
  std::optional<T> parameterValueAs(Token name);
  template <typename T = Object>
  T *parameterValueAsObject(Token name) const;

  void removeParameter(Token name);
  void removeAllParameters();

  // Index-based access
  size_t numParameters() const;
  const Parameter &parameterAt(size_t i) const;
  Parameter &parameterAt(size_t i);
  const char *parameterNameAt(size_t i) const;

  //// ANARI Objects /////

  virtual anari::Object makeANARIObject(anari::Device d) const;

  void updateANARIParameter(anari::Device d,
      anari::Object o,
      const Parameter &p,
      const char *n,
      AnariObjectCache *cache = nullptr) const;
  void updateAllANARIParameters(anari::Device d,
      anari::Object o,
      AnariObjectCache *cache = nullptr) const;

  //// Updates ////

  void setUpdateDelegate(BaseUpdateDelegate *ud);

 protected:
  virtual void parameterChanged(const Parameter *p, const Any &oldVal) override;
  virtual void removeParameter(const Parameter *p) override;
  BaseUpdateDelegate *updateDelegate() const;

 private:
  friend struct Scene;

  void incObjectUseCountParameter(const Parameter *p);
  void decObjectUseCountParameter(const Parameter *p);

  void initMetadata() const;

  Scene *m_scene{nullptr};
  ParameterMap m_parameters;
  anari::DataType m_type{ANARI_UNKNOWN};
  Token m_subtype;
  std::string m_name;
  std::string m_description;
  size_t m_index{0};
  BaseUpdateDelegate *m_updateDelegate{nullptr};
  mutable std::unique_ptr<core::DataTree> m_metadata;
  size_t m_useCount{0};
};

void print(const Object &obj, std::ostream &out = std::cout);

// Type trait-like helper functions //

template <typename T>
constexpr bool isObject()
{
  return std::is_same<Object, T>::value || std::is_base_of<Object, T>::value;
}

// Object use-count pointer type //////////////////////////////////////////////

template <typename T>
struct ObjectUsePtr
{
  static_assert(isObject<T>(),
      "ObjectUsePtr can only be instantiated with tsd::core::Object types");

  ObjectUsePtr() = default;
  ObjectUsePtr(T *o);
  ObjectUsePtr(IndexedVectorRef<T> o);
  ObjectUsePtr(const ObjectUsePtr<T> &o);
  ObjectUsePtr(ObjectUsePtr<T> &&o);
  ~ObjectUsePtr();

  ObjectUsePtr &operator=(const ObjectUsePtr<T> &o);
  ObjectUsePtr &operator=(ObjectUsePtr<T> &&o);

  ObjectUsePtr &operator=(T *o);
  ObjectUsePtr &operator=(IndexedVectorRef<T> o);

  void reset();

  const T *get() const;
  const T *operator->() const;
  const T &operator*() const;
  T *get();
  T *operator->();
  T &operator*();

  operator bool() const;

 private:
  IndexedVectorRef<T> m_object;
};

// ANARI object info parsing //////////////////////////////////////////////////

std::vector<std::string> getANARIObjectSubtypes(
    anari::Device d, anari::DataType type);

Object parseANARIObjectInfo(
    anari::Device d, ANARIDataType type, const char *subtype);

// Inlined definitions ////////////////////////////////////////////////////////

// Object //

template <typename T>
inline Parameter *Object::setParameter(Token name, T value)
{
  auto *p = m_parameters.at(name);
  if (p)
    p->setValue(value);
  else
    p = &(addParameter(name).setValue(value));
  return p;
}

template <typename T>
inline std::optional<T> Object::parameterValueAs(Token name)
{
  static_assert(!isObject<T>(),
      "Object::parameterValueAs() does not work on parameters holding objects");

  auto *p = parameter(name);
  if (!p || !p->value().is<T>())
    return {};
  return p->value().get<T>();
}

// ObjectUsePtr //

template <typename T>
inline ObjectUsePtr<T>::ObjectUsePtr(T *o)
    : m_object(o ? o->self() : IndexedVectorRef<T>{})
{
  if (m_object)
    m_object->incUseCount();
}

template <typename T>
inline ObjectUsePtr<T>::ObjectUsePtr(IndexedVectorRef<T> o)
    : m_object(o)
{
  if (m_object)
    m_object->incUseCount();
}

template <typename T>
inline ObjectUsePtr<T>::ObjectUsePtr(const ObjectUsePtr<T> &o)
    : m_object(o.m_object)
{
  if (m_object)
    m_object->incUseCount();
}

template <typename T>
inline ObjectUsePtr<T>::ObjectUsePtr(ObjectUsePtr<T> &&o) : m_object(o.m_object)
{
  o.m_object = {};
}

template <typename T>
inline ObjectUsePtr<T>::~ObjectUsePtr()
{
  reset();
}

template <typename T>
inline ObjectUsePtr<T> &ObjectUsePtr<T>::operator=(const ObjectUsePtr &o)
{
  if (this != &o) {
    reset();
    m_object = o.m_object;
    if (m_object)
      m_object->incUseCount();
  }
  return *this;
}

template <typename T>
inline ObjectUsePtr<T> &ObjectUsePtr<T>::operator=(ObjectUsePtr<T> &&o)
{
  if (this != &o) {
    reset();
    m_object = o.m_object;
    o.m_object = {};
    if (m_object)
      m_object->incUseCount();
  }
  return *this;
}

template <typename T>
inline ObjectUsePtr<T> &ObjectUsePtr<T>::operator=(T *o)
{
  reset();
  if (o) {
    m_object = o->self();
    o->incUseCount();
  }
  return *this;
}

template <typename T>
inline ObjectUsePtr<T> &ObjectUsePtr<T>::operator=(IndexedVectorRef<T> o)
{
  static_assert(isObject<T>(),
      "ObjectUsePtr can only be assigned IndexedVectorRef<T> when T is a"
      " tsd::core::Object type");
  reset();
  if (o) {
    m_object = o;
    o->incUseCount();
  }
  return *this;
}

template <typename T>
void ObjectUsePtr<T>::reset()
{
  if (m_object)
    m_object->decUseCount();
  m_object = {};
}

template <typename T>
inline const T *ObjectUsePtr<T>::get() const
{
  return m_object.data();
}

template <typename T>
inline const T *ObjectUsePtr<T>::operator->() const
{
  return m_object.data();
}

template <typename T>
inline const T &ObjectUsePtr<T>::operator*() const
{
  return *get();
}

template <typename T>
inline T *ObjectUsePtr<T>::get()
{
  return m_object.data();
}

template <typename T>
inline T *ObjectUsePtr<T>::operator->()
{
  return m_object.data();
}

template <typename T>
inline T &ObjectUsePtr<T>::operator*()
{
  return *get();
}

template <typename T>
inline ObjectUsePtr<T>::operator bool() const
{
  return m_object;
}

} // namespace tsd::core
