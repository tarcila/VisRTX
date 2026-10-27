// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#define BLACKBODY_CUDA
#include "blackbody/testBlackbody.h"

int main()
try {
  return blackbody_test::run(BLACKBODY_ORACLE);
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
