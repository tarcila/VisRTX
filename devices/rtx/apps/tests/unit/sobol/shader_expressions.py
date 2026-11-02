#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compile actual production shader functions on the host (not a branch model).

Only translation-unit includes/OptiX-only entry points are omitted. Function
bodies are copied verbatim, with #line locations back to their source files.
"""
import pathlib
import re
import sys

root, output = map(pathlib.Path, sys.argv[1:])
root = root.resolve()

def functions(path, predicate):
    text = path.read_text()
    result = []
    for match in re.finditer(r"^VISRTX_(?:DEVICE|CALLABLE)\s+[^;{]+\{", text, re.M):
        signature = match.group()
        if not predicate(signature):
            continue
        start = match.start()
        end = match.end()
        depth = 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        line = text.count("\n", 0, start) + 1
        result.append(f'#line {line} "{path}"\n' + text[start:end] + "\n")
    if not result:
        raise RuntimeError(f"No production expressions found in {path}")
    return "\n".join(result)

gpu = root / "devices/rtx/device/gpu"
shader = root / "devices/rtx/device/material/shaders/PhysicallyBasedShader_ptx.cu"
content = '#include "gpu/gpu_math.h"\n#include "gpu/shadingState.h"\n#include <glm/trigonometric.hpp>\nusing namespace visrtx;\n'
content += functions(gpu / "gpu_util.h", lambda s: any(
    re.search(r"\b" + name + r"\(", s) for name in
    ["luminance", "computeOrthonormalBasis", "sampleHemisphere"]))
content += re.search(r"^constexpr float CLEARCOAT_F0 = [^;]+;", shader.read_text(), re.M).group() + "\n"
content += functions(shader, lambda s: (s.startswith("VISRTX_DEVICE") and not any(
    name in s for name in ["applyNormalMap(", "sampleNormalMap("])) or
    "NextRay __direct_callable__nextRay(" in s)
# Observe the actual assignment consumed by mdlBsdf_sample, without requiring
# the proprietary SDK on the host. The generated wrapper only supplies its
# destination and returns it; all draws/component construction are verbatim.
mdl = root / "devices/rtx/device/material/shaders/MDLShader_ptx.cu"
text = mdl.read_text()
start = text.index("  int qmcCursor = 0;", text.index("NextRay __direct_callable__nextRay("))
end = text.index("\n\n  mdlBsdf_sample", start)
content += '#include <vector_functions.h>\n'
content += 'VISRTX_DEVICE float4 mdlXi(RandState *rs, const NextRayQmc *qmc) {\nstruct { float4 xi; } sample_data;\n'
content += f'#line {text.count(chr(10), 0, start) + 1} "{mdl}"\n'
content += text[start:end] + '\nreturn sample_data.xi;\n}\n'
# Rectangle coordinate expression is also compiled by the host, whose C++
# argument evaluation order differs from nvcc in the regression environment.
light = gpu / "sampleLight.h"
text = light.read_text()
start = text.index("  auto u =", text.index("VISRTX_DEVICE LightSample sampleRectLight("))
end = text.index("\n\n  // Uniform sampling on rectangle", start)
content += '#include "gpu/owenSobol.h"\n'
content += 'VISRTX_DEVICE vec2 rectCoordinates(RandState &rs, bool useQmc = false, uint32_t sampleIndex = 0, uint32_t pixelSeed = 0) {\n'
content += f'#line {text.count(chr(10), 0, start) + 1} "{light}"\n'
content += text[start:end] + '\nreturn uv;\n}\n'
content += '#include <cub/thread/thread_search.cuh>\n'
content += 'template <typename T>\n' + functions(light, lambda s: 'int inverseSampleCDF(' in s)
start = text.index("  auto u =", text.index("VISRTX_DEVICE LightSample sampleHDRILight(const LightGPUData &ld,\n    const mat4 &xfm,\n    RandState &rs,"))
end = text.index("\n\n  // Convert UV coordinates", start)
content += 'VISRTX_DEVICE vec2 hdriCoordinates(const LightGPUData &ld, RandState &rs, bool useQmc = false, uint32_t sampleIndex = 0, uint32_t pixelSeed = 0) {\n'
content += f'#line {text.count(chr(10), 0, start) + 1} "{light}"\n'
content += text[start:end] + '\nreturn uv;\n}\n'
output.write_text(content)
