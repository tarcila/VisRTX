// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"

namespace tsd::rendering {

struct ClearBuffersPass : public RenderPass
{
  ClearBuffersPass();
  ~ClearBuffersPass() override;

 private:
  void render(RenderBuffers &b, int stageId) override;
};

} // namespace tsd::rendering
