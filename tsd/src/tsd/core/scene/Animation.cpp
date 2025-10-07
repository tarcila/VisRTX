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
    Object &obj, Token parameter, const TimeStepArrays &steps)
{
  m_timesteps.object = obj;
  m_timesteps.parameterName = {parameter};
  m_timesteps.steps = {steps};
  m_info = "current timestep: 0/" + std::to_string(steps.size() - 1);
}

void Animation::setAsTimeSteps(Object &obj,
    const std::vector<Token> &parameters,
    const std::vector<TimeStepArrays> &steps)
{
  if (parameters.size() != steps.size()) {
    logError(
        "[AnimatedTimeSeries::setAsTimeSteps()] parameter/steps size mismatch");
    return;
  }

  m_timesteps.object = obj;
  m_timesteps.parameterName = parameters;
  m_timesteps.steps = steps;
  m_info = "current timestep: 0/"
      + std::to_string(steps.empty() ? 0 : steps[0].size() - 1);
}

void Animation::update(float time)
{
  auto &ts = m_timesteps;

  if (ts.steps.empty() || !ts.object || ts.parameterName.empty()) {
    logWarning(
        "[AnimatedTimeSeries::update()] incomplete animation object '%s'",
        name().c_str());
    return;
  }

  for (size_t i = 0; i < ts.steps.size(); i++) {
    auto &steps = ts.steps[i];
    auto &parameterName = ts.parameterName[i];

    const float scaledTime = std::clamp(time, 0.f, 1.f) * (steps.size() - 1);
    const size_t idx = static_cast<size_t>(std::ceil(scaledTime));
    if (idx != ts.currentStep) {
      ts.object->setParameterObject(parameterName, *steps[idx]);
      m_info = "current timestep: " + std::to_string(idx) + "/"
          + std::to_string(ts.steps.size() - 1);
    }
  }
}

void Animation::serialize(DataNode &node) const
{
  node["name"] = name();

  // Serialize as timestep animation //

  auto &ts = m_timesteps;

  // Write node values

  auto &timeseries = node["timeseries"];
  timeseries["object"] = ts.object
      ? tsd::core::Any(ts.object->type(), ts.object->index())
      : tsd::core::Any();
  timeseries["currentStep"] = ts.currentStep;

  // Write animation sets

  auto &animationSets = timeseries["animationSets"];
  for (size_t i = 0; i < ts.steps.size(); i++) {
    auto &setNode = animationSets.append();
    setNode["parameterName"] = ts.parameterName[i].str();

    std::vector<size_t> setArrayIndices;
    setArrayIndices.reserve(ts.steps[i].size());
    for (auto &s : ts.steps[i])
      setArrayIndices.push_back(s->index());

    setNode["steps"].setValueAsArray(setArrayIndices);
  }
}

void Animation::deserialize(DataNode &node)
{
  auto getTimeStepArrays = [this](DataNode &setNode) {
    size_t *indices = nullptr;
    size_t numSteps = 0;
    setNode["steps"].getValueAsArray<size_t>(&indices, &numSteps);
    TimeStepArrays steps;
    for (size_t i = 0; i < numSteps; i++)
      steps.emplace_back(m_scene->getObject<Array>(indices[i]));
    return steps;
  };

  //////////////////

  name() = node["name"].getValueAs<std::string>();

  if (auto *c = node.child("timeseries"); c != nullptr) {
    auto &ts = m_timesteps;
    auto &tsNode = *c;

    auto object = m_scene->getObject(tsNode["object"].getValue());

    if (auto *sets = tsNode.child("animationSets"); sets != nullptr) {
      std::vector<Token> parameterNames;
      std::vector<TimeStepArrays> allSteps;
      sets->foreach_child([&](DataNode &setNode) {
        auto parameterName =
            Token(setNode["parameterName"].getValueAs<std::string>().c_str());
        auto steps = getTimeStepArrays(setNode);
        parameterNames.push_back(parameterName);
        allSteps.push_back(steps);
      });
      setAsTimeSteps(*object, parameterNames, allSteps);
    } else {
      auto parameterName =
          Token(tsNode["parameterName"].getValueAs<std::string>().c_str());
      auto steps = getTimeStepArrays(tsNode);
      setAsTimeSteps(*object, Token(parameterName.c_str()), steps);
    }

    ts.currentStep = tsNode["currentStep"].getValueAs<size_t>();
  }
}

} // namespace tsd::core
