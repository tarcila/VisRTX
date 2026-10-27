// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#define VOLUME_REFERENCE_CUDA
#include "volumeReference/testReference.h"

int main()
{
  return volume_reference_test::run(VOLUME_LOG_ORACLE);
}
