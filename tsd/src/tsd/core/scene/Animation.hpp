// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/Object.hpp"
#include "tsd/core/scene/ObjectUsePtr.hpp"
#include "tsd/core/scene/objects/Array.hpp"
// std
#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

namespace tsd::core {

struct Animation
{
  Animation() = default;
  virtual ~Animation() = default;

  virtual void update(float time) = 0;

  std::string &name();
  const std::string &name() const;

  const std::string &info() const;

 protected:
  std::string m_info;

 private:
  std::string m_name;
};

// Concrete animation types ///////////////////////////////////////////////////

template <typename T>
struct AnimatedTimeSeries : public Animation
{
  static_assert(std::is_base_of<Object, T>::value,
      "AnimatedTimeSeries can only animate tsd::Object subclasses");

  AnimatedTimeSeries();
  ~AnimatedTimeSeries() override = default;

  void setTargetObject(T &obj);
  void setTargetParameterName(Token name);
  void setSteps(const std::vector<ObjectUsePtr<Array>> &steps);

  void update(float time) override;

 private:
  ObjectUsePtr<T> m_object;
  Token m_parameterName;
  std::vector<ObjectUsePtr<Array>> m_steps;
  size_t m_currentStep{TSD_INVALID_INDEX};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline AnimatedTimeSeries<T>::AnimatedTimeSeries()
{
  m_info = "<incomplete animation>";
}

template <typename T>
inline void AnimatedTimeSeries<T>::setTargetObject(T &obj)
{
  m_object = &obj;
}

template <typename T>
inline void AnimatedTimeSeries<T>::setTargetParameterName(Token name)
{
  m_parameterName = name;
}

template <typename T>
inline void AnimatedTimeSeries<T>::setSteps(
    const std::vector<ObjectUsePtr<Array>> &steps)
{
  m_steps = steps;
  m_info = "current timestep: 0/" + std::to_string(steps.size() - 1);
}

template <typename T>
inline void AnimatedTimeSeries<T>::update(float time)
{
  if (m_steps.empty() || !m_object || !m_parameterName) {
    logWarning(
        "[AnimatedTimeSeries::update()] incomplete animation object '%s'",
        name().c_str());
    return;
  }

  const float scaledTime = std::clamp(time, 0.f, 1.f) * (m_steps.size() - 1);
  const size_t idx = static_cast<size_t>(std::ceil(scaledTime));
  if (idx != m_currentStep) {
    m_object->setParameterObject(m_parameterName, *m_steps[idx]);
    m_info = "current timestep: " + std::to_string(idx) + "/"
        + std::to_string(m_steps.size() - 1);
  }
}

} // namespace tsd::core