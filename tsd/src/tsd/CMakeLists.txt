## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(app)
add_subdirectory(core)
add_subdirectory(io)
if (TSD_USE_MPI)
  add_subdirectory(mpi)
endif()
add_subdirectory(rendering)
if (TSD_BUILD_UI_LIBRARY)
  add_subdirectory(ui)
endif()
