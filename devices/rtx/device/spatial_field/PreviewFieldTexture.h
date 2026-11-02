// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime_api.h>
namespace visrtx {
struct StoredFieldReferenceGPUData;
// Optional immutable scalar-field representation, independent of lights.
// Canonical reference sampling never consumes this hardware-filtered view.
class PreviewFieldTexture
{
 public:
  PreviewFieldTexture() = default;
  ~PreviewFieldTexture();
  PreviewFieldTexture(const PreviewFieldTexture &) = delete;
  PreviewFieldTexture &operator=(const PreviewFieldTexture &) = delete;
  bool prepare(const StoredFieldReferenceGPUData &field);
  cudaTextureObject_t texture() const;

 private:
  cudaArray_t m_array{nullptr};
  cudaTextureObject_t m_texture{0};
  bool m_attempted{false};
};
} // namespace visrtx
