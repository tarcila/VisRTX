# Calibrated blackbody: bounded production-provider qualification

The production `gpu/blackbody/blackbody.h` provider is tested directly on CPU,
CUDA, and CUDA with the production fast-math option. The provider and its numeric
lookup/spectral tables were **not changed** in this qualification slice.
This record supplements the earlier calibration experiment; it is not an
advertisement of complete principled-volume renderer support.

## Independent reference and coverage

`tools/principled_volume/blackbody_reference.py` reads SHA256-pinned original CIE
observer/D65 CSVs, reconstructs the fixed D65 normalization, and performs direct
0.25 nm Planck/CIE quadrature with interpolated observer values over 360–830 nm.
It does not import or evaluate the production LUT. Its generated `reference.txt`
is separately attributed CC BY-SA 4.0 data.

The 1,297 float-representable temperature controls include:

- a logarithmic sweep, every LUT knot and quarter-cell, and 512 fixed-seed probes;
- adjacent representable temperatures on both sides of 300 K and 100000 K;
- zero, a float subnormal, cold tails at 25/30/50/100 K, and out-of-table values
  through the largest finite float temperature.

The newly added 25/30 K cases exposed an oracle-only intermediate underflow:
`RGB * Y / projectedY` lost the product before division. Forming the dimensionless
`Y / projectedY` scale first fixes this without changing the mathematical gamut
convention. Neither production tables nor test tolerances were adjusted to fit
results. CPU and GPU independently compared against the corrected spectral data.

`apps/tests/unit/blackbody/testBlackbody.h` checks finite, nonnegative RGB,
reference luminance, and actual production absorption-coupled source evaluation.
The coupled controls use colored extinction, scattering albedos 0/0.5/1, and at
100 K intensity and tint of float32 `1e30`, detecting premature float narrowing
of cold radiance. Zero-density explicit emission remains exactly `(2,4,8)` and
albedo-one thermal emission is exactly zero.

## Acceptance criteria and measured results

Criteria were selected before running production-provider tests:

- LUT RGB absolute error / reference Y <= `2e-4`;
- fallback RGB absolute error / reference Y <= `1e-3`;
- relative luminance and normalized coupled-source error use the respective limit;
- exact zero where the reference underflows to zero;
- adjacent-float seam jump / center maximum RGB <= `5e-5`.

The RGB criteria retain the previous calibration limits. Unlike that experiment's
float-output check, cold tails below `1e-30` do not receive a large absolute-error
floor here: representable double radiance is checked relatively.

| Mode | Max LUT RGB error / Y | Max fallback RGB error / Y | Max relative Y error | Max seam jump / center max RGB |
|---|---:|---:|---:|---:|
| CPU | 3.05765025e-5 | 4.01058607e-5 | 8.52942756e-6 | 7.58816652e-6 |
| CUDA | 3.05765025e-5 | 4.01058607e-5 | 8.52942756e-6 | 7.58816652e-6 |
| CUDA fast math | 5.92855135e-5 | 4.01058607e-5 | 1.26084189e-5 | 1.06376279e-5 |

All 1,297 controls passed in each mode. These are errors against a sampled,
finite-resolution spectral oracle, not exact spectral integration, perceptual
color differences, per-channel relative bounds, or a full-float-domain proof.
Subnormal-underflow references do not establish retention of mathematically
unrepresentable radiance. No invalid-content scans, temperature clamps, optical
clamps, or new runtime failure policy were added.

## Installed attribution

`rtx::BlackbodyInstall` installs the `blackbody-data` component into a fresh
build-local prefix, then verifies all eight files byte-for-byte against source:

- `ATTRIBUTION.md`, `LICENSE.md`, `tables.h`, `reference.txt`;
- both original CIE CSVs and their two metadata files.

They install under `${CMAKE_INSTALL_DATAROOTDIR}/VisRTX/blackbody` (normally
`share/VisRTX/blackbody`). The normal unfiltered CMake install includes this
component; component-selective distributors must include `blackbody-data` with
binaries embedding these numbers. The test verifies this component, not every
external packaging system or complete binary package. `LICENSE.md` contains the
CC BY-SA notice and official legal-terms link, not a bundled full legal-text copy.

## Reproduce

From a configured RTX build:

```sh
cmake --build _out/_cmake --config Release --target \
  rtx_test_Blackbody rtx_test_BlackbodyCUDA rtx_test_BlackbodyCUDAFastMath
ctest --test-dir _out/_cmake -C Release -R 'rtx::Blackbody' --output-on-failure
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
  _out/_cmake/Release/rtx_test_BlackbodyCUDAFastMath
```

Five new CTests cover the three evaluators, oracle reproducibility/source hashes,
and installed attribution. A combined ten-test selection also passed the public
Interactive thermal controls and the CPU/CUDA/fast-math optical-reference and
log-oracle tests. Compute Sanitizer reported zero errors and zero leaked
allocations for the fast-math blackbody executable.

No throughput/default-rate benchmark, exhaustive rendered-path qualification,
Quality transport, or extension advertisement follows from these results.
