// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Animation.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/Scene.hpp"
// std
#include <algorithm>

namespace tsd::core {

Animation::Animation(Scene *s, const char *name) : m_scene(s), m_name(name) {}

Scene *Animation::scene() const
{
  return m_scene;
}

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

void Animation::serialize(DataNode &node) const
{
  node["name"] = name();

  // Serialize as timestep animation //

  auto &ts = m_timesteps;

  // Collect timestep array indices

  std::vector<size_t> arrayIndices;
  arrayIndices.reserve(ts.steps.size());
  for (auto &s : ts.steps)
    arrayIndices.push_back(s->index());

  // Write node values

  auto &timeseries = node["timeseries"];
  timeseries["object"] = ts.object
      ? tsd::core::Any(ts.object->type(), ts.object->index())
      : tsd::core::Any();
  timeseries["parameterName"] = ts.parameterName.str();
  timeseries["currentStep"] = ts.currentStep;
  timeseries["steps"].setValueAsArray(arrayIndices);
}

void Animation::deserialize(DataNode &node)
{
  name() = node["name"].getValueAs<std::string>();

  if (auto *c = node.child("timeseries"); c != nullptr) {
    auto &ts = m_timesteps;
    auto &tsNode = *c;

    ts.object = *m_scene->getObject(tsNode["object"].getValue());
    ts.parameterName = tsNode["parameterName"].getValueAs<std::string>();
    ts.currentStep = tsNode["currentStep"].getValueAs<size_t>();

    size_t *indices = nullptr;
    size_t numSteps = 0;
    tsNode["steps"].getValueAsArray<size_t>(&indices, &numSteps);
    for (size_t i = 0; i < numSteps; i++)
      ts.steps.emplace_back(m_scene->getObject<Array>(indices[i]));
  }
}

} // namespace tsd::core
