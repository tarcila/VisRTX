# VisRTX Devices + TSD

![VisRTX Teaser](teaser.png)

This repository contains multiple implementations of the [Khronos ANARI
standard](https://www.khronos.org/anari) developed by the HPC Visualization
Developer Technology team at NVIDIA. Additionally, an emerging family of
applications are actively developed to help users explore ANARI rendering
capabilities, primarily for scalable, scientific visualizations.

The following ANARI devices are available:

- [RTX device](devices/rtx/) based on OptiX
- [OpenGL device](devices/gl/) (experimental)

For any new feature requests or bugs found in extensions that are implemented,
do not hesitate to [open an issue](https://github.com/NVIDIA/VisRTX/issues/new)!

### Provided Sample Applications (TSD)

There is an interactive collection of applications which provide easy ways
to load and interact with ANARI scenes for the sake of exploring ANARI's
capabilities, which is located in the [tsd/](tsd/) subdirectory. For more
details on TSD, please refer to its [README](tsd/README.md) document.

## Build + Install

[VisRTX](devices/rtx/) and [VisGL](devices/gl/) are supported on both Linux and
Windows. [TSD](tsd/) is supported on Linux, Windows, and macOS.

Each device and TSD can be built stand alone (separately invoking CMake on
their repsective subdirectories), or as a combined build (invoking CMake on the
repository's root directory).

Please refer to each subproject's README for more details.

Note that the devices are installable to `CMAKE_INSTALL_PREFIX`, but TSD is not
(yet) configured to be an installable set of libraries and applications. Please
use them (or copy) from of your build directory directly.
