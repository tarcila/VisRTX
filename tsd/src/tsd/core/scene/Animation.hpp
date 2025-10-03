// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/AnyObjectUsePtr.hpp"
#include "tsd/core/scene/ObjectUsePtr.hpp"
#include "tsd/core/scene/objects/Array.hpp"
// std
#include <string>
#include <vector>

namespace tsd::core {

struct Animation
{
  Animation() = default;
  ~Animation() = default;

  std::string &name();
  const std::string &name() const;
  const std::string &info() const;

  void setAsTimeSteps(Object &obj,
      Token parameter,
      const std::vector<ObjectUsePtr<Array>> &steps);

  void update(float time);

 private:
  std::string m_name;
  std::string m_info{"<incomplete animation>"};

  struct TimeStepData
  {
    AnyObjectUsePtr object;
    Token parameterName;
    std::vector<ObjectUsePtr<Array>> steps;
    size_t currentStep{TSD_INVALID_INDEX};
  } m_timesteps;
};

} // namespace tsd::core