// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "RenderPass.h"
#include "tsd/core/TSDMath.hpp"

namespace tsd::rendering {

struct SSAOPass : public RenderPass
{
  SSAOPass();
  ~SSAOPass() override;

  // Configuration parameters
  void setRadius(float radius);
  void setBias(float bias);
  void setIntensity(float intensity);
  void setSampleCount(int samples);
  void setKernelSize(int size);
  void setNoiseSize(int size);

  float getRadius() const;
  float getBias() const;
  float getIntensity() const;
  int getSampleCount() const;

 protected:
  void render(RenderBuffers &b, int stageId) override;
  void updateSize() override;

 private:
  void initializeKernel();
  void initializeNoise();
  void generateAOBuffer(RenderBuffers &b);
  void blurAOBuffer();
  void applyAOToColor(RenderBuffers &b);

  // Configuration
  float m_radius{0.5f};
  float m_bias{0.025f};
  float m_intensity{1.0f};
  int m_sampleCount{64};
  int m_kernelSize{64};
  int m_noiseSize{4};

  // Internal buffers
  float *m_aoBuffer{nullptr};
  float *m_blurBuffer{nullptr};
  
  // Kernel and noise data
  float *m_kernel{nullptr};  // Store as flat array of vec3
  float *m_noise{nullptr};   // Store as flat array of vec3
  bool m_initialized{false};
};

} // namespace tsd::rendering