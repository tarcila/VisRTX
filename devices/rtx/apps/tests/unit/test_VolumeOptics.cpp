/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "volumeOpticsTestCases.h"

#include <cstdio>

int main()
{
  int failures = 0;
  for (int i = 0; i < visrtx::test::VOLUME_OPTICS_CASES; ++i) {
    const int line = visrtx::test::volumeOpticsTestCase(i);
    if (line) {
      std::printf("FAIL case %d at volumeOpticsTestCases.h:%d\n", i, line);
      ++failures;
    }
  }
  std::printf("Volume optics: %d cases, %d failures\n",
      visrtx::test::VOLUME_OPTICS_CASES,
      failures);
  return failures ? 1 : 0;
}
