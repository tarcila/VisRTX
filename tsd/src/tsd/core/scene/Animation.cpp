// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Animation.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <algorithm>

namespace tsd::core {

std::string &Animation::name()
{
  return m_name;
}

const std::string &Animation::name() const
{
  return m_name;
}

const std::string &Animation::info() const
{
  return m_info;
}

void Animation::setAsTimeSteps(
    Object &obj, Token parameter, const std::vector<ObjectUsePtr<Array>> &steps)
{
  m_timesteps.object = obj;
  m_timesteps.parameterName = parameter;
  m_timesteps.steps = steps;
  m_info = "current timestep: 0/" + std::to_string(steps.size() - 1);
}

void Animation::update(float time)
{
  auto &ts = m_timesteps;

  if (ts.steps.empty() || !ts.object || !ts.parameterName) {
    logWarning(
        "[AnimatedTimeSeries::update()] incomplete animation object '%s'",
        name().c_str());
    return;
  }

  const float scaledTime = std::clamp(time, 0.f, 1.f) * (ts.steps.size() - 1);
  const size_t idx = static_cast<size_t>(std::ceil(scaledTime));
  if (idx != ts.currentStep) {
    ts.object->setParameterObject(ts.parameterName, *ts.steps[idx]);
    m_info = "current timestep: " + std::to_string(idx) + "/"
        + std::to_string(ts.steps.size() - 1);
  }
}

} // namespace tsd::core
