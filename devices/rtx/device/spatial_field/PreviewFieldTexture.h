// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime_api.h>
namespace visrtx {
struct StoredFieldReferenceGPUData;
// Optional immutable scalar-field representation, independent of lights.
// Reference uses neither view; linear is shadow-only, point is preview optics.
class PreviewFieldTexture
{
 public:
  PreviewFieldTexture() = default;
  ~PreviewFieldTexture();
  PreviewFieldTexture(const PreviewFieldTexture &) = delete;
  PreviewFieldTexture &operator=(const PreviewFieldTexture &) = delete;
  bool prepare(const StoredFieldReferenceGPUData &field);
  cudaTextureObject_t texture() const;
  cudaTextureObject_t pointTexture() const;

 private:
  cudaArray_t m_array{nullptr};
  cudaTextureObject_t m_texture{0};
  cudaTextureObject_t m_pointTexture{0};
  bool m_attempted{false};
};
} // namespace visrtx
