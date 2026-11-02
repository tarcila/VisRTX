/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "TransferFunction1D.h"
#include "utility/ScopedReferenceRounding.h"
#include "utility/colorMapHelpers.h"

#include <stdexcept>

namespace visrtx {

TransferFunction1D::TransferFunction1D(DeviceGlobalState *d)
    : Volume(d), m_color(this), m_opacity(this), m_field(this)
{}

TransferFunction1D::~TransferFunction1D()
{
  cleanup();
}

void TransferFunction1D::commitParameters()
{
  Volume::commitParameters();
  m_color = getParamObject<Array1D>("color");
  m_uniformColor = vec4(1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &m_uniformColor);
  getParam("color", ANARI_FLOAT32_VEC4, &m_uniformColor);
  m_opacity = getParamObject<Array1D>("opacity");
  // Raw scalar. ANARI 1.1 §5.12.1: final α = color.w · opacity. The
  // multiplication happens once per bin in discritizeTFData (m_tf[i].w =
  // c.w * o); folding c.w in here would double-count whenever c.w comes
  // from m_uniformColor.
  m_uniformOpacity = getParam<float>("opacity", 1.f);
  m_unitDistance = getParam<float>("unitDistance", 1.f);
  const SpatialField *previousField = m_field.get();
  m_field = getParamObject<SpatialField>("value");
  // AABB comes from m_field. Only a field-pointer swap moves it; field
  // data changes hit SpatialField::markFinalized.
  if (m_field.get() != previousField)
    deviceState()->objectUpdates.lastVolumeBLASChange = helium::newTimeStamp();
  getParam("valueRange", ANARI_FLOAT32_VEC2, &m_valueRange);
  getParam("valueRange", ANARI_FLOAT32_BOX1, &m_valueRange);
  double valueRange_d[2] = {0.0, 1.0};
  if (getParam("valueRange", ANARI_FLOAT64_BOX1, valueRange_d)) {
    m_valueRange.lower = float(valueRange_d[0]);
    m_valueRange.upper = float(valueRange_d[1]);
  }
}

void TransferFunction1D::finalize()
{
  m_referenceAttempted = false;
  m_reference = {};
  m_referencePayload.reset();
  m_referenceColors.reset();
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing parameter 'value' on transferFunction1D ANARIVolume");
    return;
  }

  if (!m_field->isValid() || m_field->valueType() != FieldValueType::Scalar) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "transferFunction1D requires a valid scalar spatial field");
    return;
  }

  if (m_unitDistance <= 0.f) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "unitDistance must be positive on transferFunction1D volume, "
        "clamping to minimum");
    m_unitDistance = 1e-6f;
  }

  discritizeTFData();
  createTFTexture();
  m_field->m_uniformGrid.computeOpacityBounds(
      deviceState()->stream, m_textureObject, m_tfDim, m_valueRange);
  upload();
}

bool TransferFunction1D::isValid() const
{
  return m_field && m_field->isValid()
      && m_field->valueType() == FieldValueType::Scalar;
}

ReferenceVolumeGPUData TransferFunction1D::referenceVolume()
{
  if (!isValid())
    return {};
  if (m_referenceAttempted)
    return m_reference;
  const auto unavailable = [&]() {
    m_referenceAttempted = true;
    return ReferenceVolumeGPUData{};
  };
  const ScopedReferenceRounding rounding;
  const auto *field = m_field->referenceField();
  if (!field || m_tf.empty())
    return unavailable();
  const double limit =
      optical_reference::promote(std::numeric_limits<float>::max());
  if (!optical_reference::validRange(
          {m_valueRange.lower, m_valueRange.upper}, -limit, limit, false)
      || !(optical_reference::promote(m_valueRange.lower)
          < optical_reference::promote(m_valueRange.upper)))
    return unavailable();
  optical_reference::RawRange<float> alpha;
  double maxColor = 0.;
  for (const auto &color : m_tf) {
    alpha = optical_reference::extendRange(alpha, color.w);
    for (int c = 0; c < 3; ++c)
      maxColor = std::max(maxColor, std::abs(double(color[c])));
  }
  TransferFunctionReferenceGPUData data;
  if (!optical_reference::buildTransferBounds({alpha.lower, alpha.upper},
          {m_unitDistance, m_unitDistance},
          data.opticalBounds))
    return unavailable();
  data.field = field->gpuData();
  data.valueRange = m_valueRange;
  data.unitDistance = m_unitDistance;
  data.colorCount = uint32_t(m_tf.size());
  m_referenceColors.reserve(m_tf.size() * sizeof(vec4));
  auto error = cudaMemcpy(m_referenceColors.ptr(),
      m_tf.data(),
      m_tf.size() * sizeof(vec4),
      cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  data.colors = m_referenceColors.ptrAs<vec4>();
  m_referencePayload.reserve(sizeof(data));
  error = cudaMemcpy(
      m_referencePayload.ptr(), &data, sizeof(data), cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  error = cudaStreamSynchronize(nullptr);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  double step=std::numeric_limits<double>::max();
  const auto domain=field->descriptor().domain;
  for(int c=0;c<3;++c)
    step=std::min(step,.5*(optical_reference::promote(domain.upper[c])-optical_reference::promote(domain.lower[c])));
  const double hint=optical_reference::promote(m_field->stepSize());
  if(hint>0. && std::isfinite(hint)) step=std::min(step,hint);
  m_reference = {VolumeType::TF1D,
      field->descriptor().domain,
      nullptr,
      m_referencePayload.ptrAs<TransferFunctionReferenceGPUData>(),
      step,
      index()};
  bool preview = field->supportsFloatPreview() && m_unitDistance >= 0x1p-20f
      && double(m_valueRange.upper) - double(m_valueRange.lower) >= 0x1p-20
      && std::abs(double(m_valueRange.lower)) < 0x1p20
      && std::abs(double(m_valueRange.upper)) < 0x1p20
      && optical_reference::majorant(data.opticalBounds) < 0x1p40;
  m_reference.previewFloatEligible = preview && maxColor < 0x1p20;
  m_reference.mightEmit = maxColor > 0. && alpha.upper > 0.f;
  m_referenceAttempted = true;
  return m_reference;
}

VolumeGPUData TransferFunction1D::gpuData() const
{
  VolumeGPUData retval = Volume::gpuData();
  retval.type = VolumeType::TF1D;
  retval.bounds = m_field->bounds();
  retval.stepSize = m_field->stepSize();
  retval.data.tf1d.tfTex = m_textureObject;
  retval.data.tf1d.valueRange = m_valueRange;
  retval.data.tf1d.oneOverUnitDistance = 1.0f / m_unitDistance;
  retval.data.tf1d.field = m_field->index();
  retval.data.tf1d.uniformColor = vec3(m_uniformColor);
  // Fold c.w · o here for the null-tfTex fallback in classifySample (that
  // path doesn't see discritizeTFData's per-bin fold).
  retval.data.tf1d.uniformOpacity = m_uniformOpacity * m_uniformColor.w;
  return retval;
}

void TransferFunction1D::discritizeTFData()
{
  m_tf.resize(m_tfDim);

  Span<float> cPositions;
  Span<float> oPositions;

  std::vector<float> linearColorPositions;
  std::vector<float> linearOpacityPositions;

  if (m_color) {
    linearColorPositions =
        generateLinearPositions(m_color->totalSize(), m_valueRange);
    cPositions =
        make_Span(linearColorPositions.data(), linearColorPositions.size());
  }

  if (m_opacity) {
    linearOpacityPositions =
        generateLinearPositions(m_opacity->totalSize(), m_valueRange);
    oPositions =
        make_Span(linearOpacityPositions.data(), linearOpacityPositions.size());
  }

  bool warnOnce = false;
  for (size_t i = 0; i < m_tf.size(); i++) {
    const float p = float(i) / (m_tf.size() - 1);
    vec4 c = m_uniformColor;
    if (m_color) {
      if (m_color->elementType() == ANARI_FLOAT32_VEC3) {
        c = vec4(getInterpolatedValue(
                     m_color->beginAs<vec3>(), cPositions, m_valueRange, p),
            1.f);
      } else if (m_color->elementType() == ANARI_FLOAT32_VEC4) {
        c = getInterpolatedValue(
            m_color->beginAs<vec4>(), cPositions, m_valueRange, p);
      } else if (!warnOnce) {
        reportMessage(ANARI_SEVERITY_WARNING,
            "unusable tf1D color array type set (%s)",
            anari::toString(m_color->elementType()));
        warnOnce = true;
      }
    }
    const float o = m_opacity
        ? getInterpolatedValue(
              m_opacity->beginAs<float>(), oPositions, m_valueRange, p)
        : m_uniformOpacity;
    m_tf[i] = vec4(c.x, c.y, c.z, c.w * o);
  }
}

void TransferFunction1D::createTFTexture()
{
  cleanup();

  if (m_tf.empty())
    return;

  auto desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaMallocArray(&m_cudaArray, &desc, m_tfDim);

  cudaMemcpy3DParms copyParams;
  std::memset(&copyParams, 0, sizeof(copyParams));
  copyParams.srcPtr = make_cudaPitchedPtr(
      m_tf.data(), m_tf.size() * sizeof(vec4), m_tf.size(), 0);
  copyParams.dstArray = m_cudaArray;
  copyParams.extent = make_cudaExtent(m_tfDim, 1, 1);
  copyParams.kind = cudaMemcpyHostToDevice;

  cudaMemcpy3D(&copyParams);

  cudaResourceDesc resDesc;
  std::memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_cudaArray;

  cudaTextureDesc texDesc;
  std::memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaCreateTextureObject(&m_textureObject, &resDesc, &texDesc, nullptr);
}

void TransferFunction1D::cleanup()
{
  if (m_textureObject)
    cudaDestroyTextureObject(m_textureObject);
  if (m_cudaArray)
    cudaFreeArray(m_cudaArray);
  m_textureObject = {};
  m_cudaArray = {};
}

} // namespace visrtx
