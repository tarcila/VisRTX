// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "SourceImportance.h"
#include "gpu/aggregateVolume.h"
#include "gpu/blackbody/blackbody.h"
#include "utility/ScopedReferenceRounding.h"

namespace visrtx {
namespace {
void checked(cudaError_t error)
{
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
struct Temporary
{
  double *values{nullptr};
  ~Temporary()
  {
    if (values)
      cudaFree(values);
  }
};
__global__ void evaluateSourceImportance(
    ReferenceVolumeGPUData volume, uint32_t n, double *values)
{
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n * n * n)
    return;
  const uint32_t xyz[3]{index % n, (index / n) % n, index / (n * n)};
  vec3 point;
  namespace ref = optical_reference;
  for (int c = 0; c < 3; ++c)
    point[c] = ref::narrow(ref::add(ref::promote(volume.domain.lower[c]),
        ref::multiply(ref::divide(double(xyz[c]) + .5, double(n)),
            ref::subtract(ref::promote(volume.domain.upper[c]),
                ref::promote(volume.domain.lower[c])))));
  ref::Coefficients coefficient;
  if (volume.type == VolumeType::PRINCIPLED) {
    const auto &v = *volume.principled;
    ReferenceFieldSampler sampler{v.fields};
    // The guide is a heuristic proposal, not an optical certificate. Use the
    // same bounded source evaluator as Quality instead of repeating the cold
    // spectral integration at every pilot point on animated source rebuilds.
    QualityBlackbody blackbody{v.qualityCold};
    coefficient = ref::evaluateBindings(v.bindings,
        point,
        sampler,
        blackbody,
        v.hasConstantTransmissionLog ? v.constantTransmissionLog : nullptr);
  } else if (volume.type == VolumeType::TF1D) {
    const auto &v = *volume.transfer;
    ReferenceFieldSampler sampler{&v.field};
    coefficient = ref::evaluateTransfer(
        aggregate_volume::classify(v, sampler.sample<float>(0, point)),
        v.unitDistance);
  }
  values[index] = .2126 * coefficient.emission[0]
      + .7152 * coefficient.emission[1] + .0722 * coefficient.emission[2];
}
} // namespace
SourceImportance::~SourceImportance()
{
  reset();
}
void SourceImportance::reset()
{
  if (m_cdf)
    cudaFree(m_cdf);
  m_cdf = nullptr;
  m_resolution = 0;
  m_attempted = false;
}
SourceImportanceGPUData SourceImportance::prepare(
    const ReferenceVolumeGPUData &volume)
{
  if (m_attempted)
    return {m_cdf, m_resolution};
  m_attempted = true;
  if (!volume.mightEmit || !optical_reference::nonemptyDomain(volume.domain))
    return {};
  uint32_t n = 64;
  if (volume.type == VolumeType::PRINCIPLED) {
    PrincipledVolumeGPUData data;
    checked(cudaMemcpy(
        &data, volume.principled, sizeof(data), cudaMemcpyDeviceToHost));
    if (data.fieldCount == 0)
      n = 1; // metadata proves spatially constant source
    else {
      std::vector<StoredFieldReferenceGPUData> fields(data.fieldCount);
      checked(cudaMemcpy(fields.data(),
          data.fields,
          fields.size() * sizeof(fields[0]),
          cudaMemcpyDeviceToHost));
      for (const auto &field : fields)
        if (!field.cudaSampling)
          return {};
    }
  } else if (volume.type == VolumeType::TF1D) {
    TransferFunctionReferenceGPUData data;
    checked(cudaMemcpy(
        &data, volume.transfer, sizeof(data), cudaMemcpyDeviceToHost));
    if (!data.field.cudaSampling)
      return {};
  }
  const size_t count = size_t(n) * n * n;
  Temporary temporary;
  auto error = cudaMalloc(&temporary.values, count * sizeof(double));
  if (error == cudaErrorMemoryAllocation) {
    cudaGetLastError();
    return {};
  }
  checked(error);
  evaluateSourceImportance<<<unsigned((count + 127) / 128), 128>>>(
      volume, n, temporary.values);
  checked(cudaGetLastError());
  std::vector<double> weights(count);
  checked(cudaMemcpy(weights.data(),
      temporary.values,
      count * sizeof(double),
      cudaMemcpyDeviceToHost));
  // Algorithmic importance construction, not a field-content validity scan.
  // Normalize by the largest pilot weight; PDFs later use realized CDF widths.
  const ScopedReferenceRounding rounding;
  const double maximum = *std::max_element(weights.begin(), weights.end());
  if (!(maximum > 0.) || !std::isfinite(maximum))
    return {};
  std::vector<double> cdf(count + 1, 0.);
  for (size_t i = 0; i < count; ++i)
    cdf[i + 1] = cdf[i] + weights[i] / maximum;
  const double total = cdf.back();
  if (!(total > 0.) || !std::isfinite(total))
    return {};
  for (double &value : cdf)
    value /= total;
  cdf.back() = 1.;
  error = cudaMalloc(&m_cdf, cdf.size() * sizeof(double));
  if (error == cudaErrorMemoryAllocation) {
    cudaGetLastError();
    m_cdf = nullptr;
    return {};
  }
  checked(error);
  checked(cudaMemcpy(
      m_cdf, cdf.data(), cdf.size() * sizeof(double), cudaMemcpyHostToDevice));
  checked(cudaStreamSynchronize(nullptr));
  m_resolution = n;
  return {m_cdf, m_resolution};
}
} // namespace visrtx
