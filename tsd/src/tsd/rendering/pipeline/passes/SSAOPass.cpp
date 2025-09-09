// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "SSAOPass.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace tsd::rendering {

SSAOPass::SSAOPass() = default;

SSAOPass::~SSAOPass()
{
  if (m_aoBuffer) {
    detail::free(m_aoBuffer);
    m_aoBuffer = nullptr;
  }
  if (m_blurBuffer) {
    detail::free(m_blurBuffer);
    m_blurBuffer = nullptr;
  }
  if (m_kernel) {
    detail::free(m_kernel);
    m_kernel = nullptr;
  }
  if (m_noise) {
    detail::free(m_noise);
    m_noise = nullptr;
  }
}

void SSAOPass::setRadius(float radius)
{
  m_radius = std::max(0.01f, radius);
}

void SSAOPass::setBias(float bias)
{
  m_bias = std::max(0.0f, bias);
}

void SSAOPass::setIntensity(float intensity)
{
  m_intensity = std::max(0.0f, intensity);
}

void SSAOPass::setSampleCount(int samples)
{
  m_sampleCount = std::clamp(samples, 1, 256);
  m_initialized = false; // Need to regenerate kernel
}

void SSAOPass::setKernelSize(int size)
{
  m_kernelSize = std::clamp(size, 1, 256);
  m_initialized = false; // Need to regenerate kernel
}

void SSAOPass::setNoiseSize(int size)
{
  m_noiseSize = std::clamp(size, 1, 16);
  m_initialized = false; // Need to regenerate noise
}

float SSAOPass::getRadius() const
{
  return m_radius;
}

float SSAOPass::getBias() const
{
  return m_bias;
}

float SSAOPass::getIntensity() const
{
  return m_intensity;
}

int SSAOPass::getSampleCount() const
{
  return m_sampleCount;
}

void SSAOPass::render(RenderBuffers &b, int stageId)
{
  if (!b.depth) {
    return; // Cannot do SSAO without depth buffer
  }

  if (!m_initialized) {
    initializeKernel();
    initializeNoise();
    m_initialized = true;
  }

  generateAOBuffer(b);
  blurAOBuffer();
  applyAOToColor(b);
}

void SSAOPass::updateSize()
{
  const auto size = getDimensions();
  const size_t totalPixels = size_t(size.x) * size_t(size.y);

  if (m_aoBuffer) {
    detail::free(m_aoBuffer);
  }
  if (m_blurBuffer) {
    detail::free(m_blurBuffer);
  }

  m_aoBuffer = detail::allocate<float>(totalPixels);
  m_blurBuffer = detail::allocate<float>(totalPixels);
}

void SSAOPass::initializeKernel()
{
  if (m_kernel) {
    detail::free(m_kernel);
  }

  m_kernel = detail::allocate<float>(m_kernelSize * 3); // 3 floats per vec3

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Generate random hemisphere samples
  for (int i = 0; i < m_kernelSize; ++i) {
    float x = dis(gen) * 2.0f - 1.0f; // x: [-1, 1]
    float y = dis(gen) * 2.0f - 1.0f; // y: [-1, 1]
    float z = dis(gen);               // z: [0, 1] (hemisphere)

    // Normalize
    float length = std::sqrt(x * x + y * y + z * z);
    x /= length;
    y /= length;
    z /= length;

    // Scale by random factor with bias toward center
    float scale = static_cast<float>(i) / static_cast<float>(m_kernelSize);
    scale = 0.1f + scale * scale * 0.9f; // Lerp from 0.1 to 1.0
    x *= scale;
    y *= scale;
    z *= scale;

    m_kernel[i * 3 + 0] = x;
    m_kernel[i * 3 + 1] = y;
    m_kernel[i * 3 + 2] = z;
  }
}

void SSAOPass::initializeNoise()
{
  if (m_noise) {
    detail::free(m_noise);
  }

  const int noisePixels = m_noiseSize * m_noiseSize;
  m_noise = detail::allocate<float>(noisePixels * 3); // 3 floats per vec3

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Generate random rotation vectors
  for (int i = 0; i < noisePixels; ++i) {
    m_noise[i * 3 + 0] = dis(gen) * 2.0f - 1.0f; // x: [-1, 1]
    m_noise[i * 3 + 1] = dis(gen) * 2.0f - 1.0f; // y: [-1, 1]
    m_noise[i * 3 + 2] = 0.0f;                    // z: 0 (tangent space)
  }
}

void SSAOPass::generateAOBuffer(RenderBuffers &b)
{
  const auto size = getDimensions();
  
  // Simple CPU-based SSAO implementation
  // In a real implementation, this would be done on GPU with shaders
  
  for (uint32_t y = 0; y < size.y; ++y) {
    for (uint32_t x = 0; x < size.x; ++x) {
      const size_t pixelIndex = y * size.x + x;
      
      // Get depth at current pixel
      const float centerDepth = b.depth[pixelIndex];
      
      if (centerDepth >= 1.0f) {
        // Sky/background pixel - no occlusion
        m_aoBuffer[pixelIndex] = 1.0f;
        continue;
      }
      
      float occlusion = 0.0f;
      int validSamples = 0;
      
      // Sample around the pixel
      const int sampleRadius = static_cast<int>(m_radius * 100.0f); // Scale to pixels
      
      for (int i = 0; i < m_sampleCount; ++i) {
        // Get sample offset (simplified - using kernel as 2D offsets)
        const int kernelIndex = i % m_kernelSize;
        const float kernelX = m_kernel[kernelIndex * 3 + 0];
        const float kernelY = m_kernel[kernelIndex * 3 + 1];
        
        const int sampleX = static_cast<int>(x + kernelX * sampleRadius);
        const int sampleY = static_cast<int>(y + kernelY * sampleRadius);
        
        // Check bounds
        if (sampleX < 0 || sampleX >= static_cast<int>(size.x) ||
            sampleY < 0 || sampleY >= static_cast<int>(size.y)) {
          continue;
        }
        
        const size_t sampleIndex = sampleY * size.x + sampleX;
        const float sampleDepth = b.depth[sampleIndex];
        
        // Compare depths to determine occlusion
        const float depthDiff = centerDepth - sampleDepth;
        
        if (depthDiff > m_bias) {
          // Sample is closer to camera - contributes to occlusion
          const float rangeCheck = std::abs(depthDiff) < m_radius ? 1.0f : 0.0f;
          occlusion += rangeCheck;
        }
        
        validSamples++;
      }
      
      if (validSamples > 0) {
        occlusion = 1.0f - (occlusion / validSamples);
      } else {
        occlusion = 1.0f;
      }
      
      m_aoBuffer[pixelIndex] = occlusion;
    }
  }
}

void SSAOPass::blurAOBuffer()
{
  const auto size = getDimensions();
  
  // Simple box blur to smooth the AO
  const int blurRadius = 2;
  
  for (uint32_t y = 0; y < size.y; ++y) {
    for (uint32_t x = 0; x < size.x; ++x) {
      const size_t pixelIndex = y * size.x + x;
      
      float sum = 0.0f;
      int count = 0;
      
      for (int dy = -blurRadius; dy <= blurRadius; ++dy) {
        for (int dx = -blurRadius; dx <= blurRadius; ++dx) {
          const int sampleX = static_cast<int>(x) + dx;
          const int sampleY = static_cast<int>(y) + dy;
          
          if (sampleX >= 0 && sampleX < static_cast<int>(size.x) &&
              sampleY >= 0 && sampleY < static_cast<int>(size.y)) {
            const size_t sampleIndex = sampleY * size.x + sampleX;
            sum += m_aoBuffer[sampleIndex];
            count++;
          }
        }
      }
      
      m_blurBuffer[pixelIndex] = count > 0 ? sum / count : 1.0f;
    }
  }
}

void SSAOPass::applyAOToColor(RenderBuffers &b)
{
  const auto size = getDimensions();
  
  for (uint32_t y = 0; y < size.y; ++y) {
    for (uint32_t x = 0; x < size.x; ++x) {
      const size_t pixelIndex = y * size.x + x;
      
      // Get AO factor (blurred)
      const float aoFactor = m_blurBuffer[pixelIndex];
      
      // Apply intensity
      const float finalAO = 1.0f - (1.0f - aoFactor) * m_intensity;
      
      // Get current color
      const uint32_t color = b.color[pixelIndex];
      
      // Extract RGB components
      const uint8_t r = (color >> 24) & 0xFF;
      const uint8_t g = (color >> 16) & 0xFF;
      const uint8_t blue = (color >> 8) & 0xFF;
      const uint8_t a = color & 0xFF;
      
      // Apply AO
      const uint8_t newR = static_cast<uint8_t>(r * finalAO);
      const uint8_t newG = static_cast<uint8_t>(g * finalAO);
      const uint8_t newB = static_cast<uint8_t>(blue * finalAO);
      
      // Pack back to uint32_t
      b.color[pixelIndex] = (newR << 24) | (newG << 16) | (newB << 8) | a;
    }
  }
}

} // namespace tsd::rendering