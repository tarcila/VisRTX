// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "SimulationControls.h"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
// tsd_algorithms
#include <tsd/algorithms/metal/runtime.hpp>
// std
#include <algorithm>
#include <random>

namespace tsd::demo {

SimulationControls::SimulationControls(
    tsd::ui::imgui::Application *app, const char *name)
    : tsd::ui::imgui::Window(app, name)
{
  initMetalParticleSystem();
}

SimulationControls::~SimulationControls()
{
  releaseMetalBuffers();
  shutdownMetalParticleSystem();
}

void SimulationControls::buildUI()
{
  if (!m_particleGeom) {
    ImGui::Text("{simulation window not setup correctly}");
    return;
  }

  if (ImGui::Button("reset"))
    resetSimulation();

  ImGui::SameLine();

  ImGui::Text(" | ");

  ImGui::SameLine();

  if (ImGui::Button(m_playing ? "stop" : "play"))
    m_playing = !m_playing;

  ImGui::SameLine();

  ImGui::Text(" | ");

  ImGui::SameLine();

  ImGui::BeginDisabled(m_playing);
  if (ImGui::Button("iterate") || m_playing)
    iterateSimulation();
  ImGui::EndDisabled();

  ImGui::BeginDisabled(m_playing);

  ImGui::DragInt("particles-per-side", &m_particlesPerSide, 1, 1024);

  if (ImGui::Checkbox("randomize velocities", &m_randomizeInitialVelocities))
    resetSimulation();

  ImGui::EndDisabled();

  ImGui::Separator();

  ImGui::DragFloat("rotation speed", &m_rotationSpeed, 1.f);
  ImGui::DragFloat("gravity", &m_params.gravity, 1.f);
  ImGui::DragFloat("particle mass", &m_params.particleMass, 0.1f);
  if (ImGui::DragFloat("max distance", &m_params.maxDistance, 1.f))
    updateColorMapScale();
  if (ImGui::DragFloat("color map scale", &m_colorMapScaleFactor, 1.f))
    updateColorMapScale();
  ImGui::InputFloat("delta T", &m_params.deltaT);
}

void SimulationControls::setGeometry(tsd::scene::GeometryRef particles,
    tsd::scene::GeometryRef blackHoles,
    tsd::scene::SamplerRef sampler)
{
  m_particleGeom = particles;
  m_bhGeom = blackHoles;
  m_particleColorSampler = sampler;
  updateColorMapScale();
  resetSimulation();
}

void SimulationControls::releaseMetalBuffers()
{
  namespace mtl = tsd::algorithms::metal;
  if (m_posBuffer) {
    mtl::releaseBuffer(m_posBuffer);
    m_posBuffer = nullptr;
  }
  if (m_velBuffer) {
    mtl::releaseBuffer(m_velBuffer);
    m_velBuffer = nullptr;
  }
  if (m_distBuffer) {
    mtl::releaseBuffer(m_distBuffer);
    m_distBuffer = nullptr;
  }
}

void SimulationControls::remakeDataArrays()
{
  namespace mtl = tsd::algorithms::metal;

  auto &scene = appContext()->tsd.scene;
  const int numParticles =
      m_particlesPerSide * m_particlesPerSide * m_particlesPerSide;
  releaseMetalBuffers();

  const size_t vec3Bytes = numParticles * sizeof(tsd::math::float3);
  const size_t floatBytes = numParticles * sizeof(float);

  m_posBuffer = mtl::newPrivateBuffer(vec3Bytes);
  m_velBuffer = mtl::newPrivateBuffer(vec3Bytes);
  m_distBuffer = mtl::newPrivateBuffer(floatBytes);

  m_dataPoints = scene.createArrayMetal(
      ANARI_FLOAT32_VEC3, numParticles, m_posBuffer, nullptr);
  m_dataDistances = scene.createArrayMetal(
      ANARI_FLOAT32, numParticles, m_distBuffer, nullptr);
  m_dataVelocities = scene.createArrayMetal(
      ANARI_FLOAT32_VEC3, numParticles, m_velBuffer, nullptr);

  m_dataBhPoints = scene.createArray(ANARI_FLOAT32_VEC3, 2);
}

void SimulationControls::resetSimulation()
{
  namespace mtl = tsd::algorithms::metal;

  m_playing = false;
  m_angle = 0.f;

  remakeDataArrays();
  updateBhPoints();

  const int numParticles =
      m_particlesPerSide * m_particlesPerSide * m_particlesPerSide;
  const size_t vec3Bytes = numParticles * sizeof(tsd::math::float3);
  const size_t floatBytes = numParticles * sizeof(float);

  auto *stagingPos = mtl::newSharedBuffer(vec3Bytes);
  auto *stagingVel = mtl::newSharedBuffer(vec3Bytes);
  auto *stagingDist = mtl::newSharedBuffer(floatBytes);

  auto *positions =
      static_cast<tsd::math::float3 *>(mtl::bufferContents(stagingPos));
  auto *velocities =
      static_cast<tsd::math::float3 *>(mtl::bufferContents(stagingVel));
  auto *distances = static_cast<float *>(mtl::bufferContents(stagingDist));

  if (m_randomizeInitialVelocities) {
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(-0.1f, 0.1f);
    std::for_each((float *)velocities,
        (float *)(velocities + numParticles),
        [&](auto &v) { v = dist(rng) * 255; });
  } else {
    std::fill(velocities, velocities + numParticles, tsd::math::float3(0.f));
  }

  const float d = 2.0f / m_particlesPerSide;
  size_t i = 0;
  for (int x = 0; x < m_particlesPerSide; x++) {
    for (int y = 0; y < m_particlesPerSide; y++) {
      for (int z = 0; z < m_particlesPerSide; z++) {
        auto p = tsd::math::float3(d * x - 1.f, d * y - 1.f, d * z - 1.f);
        positions[i] = p;
        distances[i] = tsd::math::length(p);
        i++;
      }
    }
  }

  mtl::blitToBuffer(stagingPos, m_posBuffer, vec3Bytes);
  mtl::blitToBuffer(stagingVel, m_velBuffer, vec3Bytes);
  mtl::blitToBuffer(stagingDist, m_distBuffer, floatBytes);

  mtl::releaseBuffer(stagingPos);
  mtl::releaseBuffer(stagingVel);
  mtl::releaseBuffer(stagingDist);

  m_particleGeom->setParameterObject("vertex.position", *m_dataPoints);
  m_particleGeom->setParameterObject("vertex.attribute0", *m_dataDistances);

  m_bhGeom->setParameterObject("vertex.position", *m_dataBhPoints);

  updateColorMapScale();
}

void SimulationControls::updateColorMapScale()
{
  m_particleColorSampler->setParameter("inTransform",
      tsd::math::makeValueRangeTransform(
          0.f, m_params.maxDistance / m_colorMapScaleFactor));
}

std::pair<tsd::math::float3, tsd::math::float3>
SimulationControls::updateBhPoints()
{
  const auto rot = tsd::math::rotation_matrix(
      tsd::math::rotation_quat(tsd::math::float3(0, 0, 1), m_angle));
  tsd::math::float4 bh1_ =
      tsd::math::mul(rot, tsd::math::float4(5.f, 0.f, 0.f, 1.f));
  tsd::math::float4 bh2_ =
      tsd::math::mul(rot, tsd::math::float4(-5.f, 0.f, 0.f, 1.f));

  tsd::math::float3 bh1(bh1_.x, bh1_.y, bh1_.z);
  tsd::math::float3 bh2(bh2_.x, bh2_.y, bh2_.z);

  auto *bhPoints = m_dataBhPoints->mapAs<tsd::math::float3>();
  bhPoints[0] = bh1;
  bhPoints[1] = bh2;
  m_dataBhPoints->unmap();

  return std::make_pair(bh1, bh2);
}

void SimulationControls::iterateSimulation()
{
  m_angle += m_rotationSpeed * 1e-4f;
  auto [bh1, bh2] = updateBhPoints();

  const int numParticles =
      m_particlesPerSide * m_particlesPerSide * m_particlesPerSide;

  tsd::demo::particlesComputeTimestepMetal(numParticles,
      m_posBuffer,
      m_velBuffer,
      m_distBuffer,
      tsd::math::float3(bh1.x, bh1.y, bh1.z),
      tsd::math::float3(bh2.x, bh2.y, bh2.z),
      m_params);

  m_dataPoints->notifyChanged();
  m_dataDistances->notifyChanged();
}

} // namespace tsd::demo
