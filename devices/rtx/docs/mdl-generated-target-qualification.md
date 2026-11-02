# Generated MDL surface targets — bounded qualification

Status: first bounded manifest passed; not generic MDL source-replacement approval.

## Frozen first manifest

Use the existing `Core::getCompiledMaterial` class compilation and
`Core::getPtxTargetCode` backend configuration. Stitch the generated target with
production MDL resource and wrapper PTX, plus a test-only OptiX probe. Do not use
an alternate SDK example backend or a hand-written scattering implementation.

Six constant materials: black, Lambertian diffuse, specular reflection,
thin-walled specular transmission, equal diffuse/specular normalized mix, and
anisotropic GGX Smith reflection. Three states each: canonical front, canonical
back (IOR order), and a fixed post-init normal `(0.98,0,0.2)` normalized. The last
state deliberately bypasses normal adaptation to expose geometric rejection; it
is not a claim that production adaptation produces that state.

65536 fixed xi tuples per state, PCG seed recipe in `mdlTargets/probe_ptx.cu`.
No retry seeds, adaptive sample counts or fixture-normal search. Capture raw
sample outputs and the actual wrapper's result with identical xi. Invoke real
sample/evaluate/pdf and production init/evaluate/pdf/next-ray callables.

Checks fixed before execution:

- Exact event bits, dead initialization, live sample preservation and unchanged
  PCG state under the four-value xi overlay.
- Reproduce positive stale outputs followed by ABSORB in the tilted real mirror
  target; inability to reproduce is missing coverage, not a stub substitute.
- Continuous `weight*sample_pdf` agrees with evaluated diffuse+glossy and the PDF
  query, absolute 2e-6 plus relative 5e-5. Singular events are not compared to a
  continuous-density estimator.
- Canonical diffuse/mirror/mix weights agree with analytic tint; canonical diffuse
  and mixed continuous PDFs agree with cosine/pi and half cosine/pi respectively.
- Exact zero contribution for black, diffuse/specular event masses; mixed singular mass agrees with 1/2
  within six binomial standard errors plus 1e-4.
- Record the loaded SDK image and runtime version, device/CUDA/OptiX identity,
  source, generated PTX, production wrappers/resources and test probe. Hash the
  captured artifacts and loaded SDK for the local evidence ledger.

The initial black-diffuse event-mass assertion was corrected after execution:
zero tint does not imply an ABSORB event in the SDK. The fixture is unchanged;
its valid invariant is exact zero contribution, not a particular event tag.
Numeric gates, xi samples and the real stale-ABSORB requirement are unchanged.

This first manifest is not the complete R5 closure matrix: it does not qualify
measured/hair resources, arbitrary layered normals, the full BTDF, MaterialX
transcoding or all continuous source-query support. No runtime material allowlist
or source-suppression policy is introduced by passing these controls.

## Results

All eighteen fixture/state pairs pass (1,179,648 xi tuples, each sampled both
raw and through the production wrapper). Observed examples:

| Target/state | Absorbed | Positive stale weights | Live singular | Live continuous |
|---|---:|---:|---:|---:|
| Black diffuse, canonical front | 0 | 0 | 0 | 65536 |
| Mirror, canonical front | 0 | 0 | 65536 | 0 |
| Mirror, post-init tilted normal | 65536 | 65536 | 0 | 0 |
| Mixed diffuse/specular, canonical front | 0 | 0 | 32580 | 32956 |
| Mixed, post-init tilted normal | 32580 | 32580 | 0 | 32956 |
| GGX, canonical front | 5837 | 5837 | 0 | 59699 |
| GGX, post-init tilted normal | 43765 | 43765 | 0 | 21771 |

The canonical GGX fixture also demonstrates stale positive outputs without the
post-init normal override. Every absorbed record becomes a fully initialized Dead
continuation; live delta records survive. The consumer fix is now supported by
real generated-target execution, not only the earlier stub.

The loaded SDK reports **MDL SDK 2025.0.3, build 387700.2665, 10 Dec 2025,
linux-x86-64**. Its package directory is named `mdl-sdk-2026.0.2`; that name is not
its runtime version. The loaded SDK SHA256 is
`2f5f84421fa29234a23d5993eb56a8f30967ba45a21ac816e9e82b6cdc519de5`.

Local evidence resides in
`_out/_cmake/devices/rtx/apps/tests/unit/mdl-target-qualification/Release/`:

- `identity.txt`: loaded SDK/device-library paths, runtime version and GPU/CUDA/
  OptiX identities;
- `manifest.json`: SHA256 values for the loaded SDK/device library, executable,
  sources, production wrapper/resources, generated targets and sample records;
- `results.csv`, per-target/state sample excerpts and captured PTX/MDL source.

Verification: full Release build; **45/45 targeted MDL/MaterialX/native/Quality
regressions passed** (66.30 s), excluding the compile benchmark. The complete
renderer suite was not repeated in this test-only slice. Compute Sanitizer
memcheck/full-leak checking of the generated-target test reports zero errors and
zero leaked allocations. A test transaction-lifetime error was corrected by
releasing the module handle before commit; it was not ignored as a backend warning.
Logs use `/tmp/visrtx-mdl-generated-*`.

No production shader behavior changed in this qualification slice. The normal
adaptation override is explicitly a source-local test fixture, not evidence that
production adaptation creates the same state. These checks establish bounded
leaf/event and sample/evaluate/PDF consistency, not full angular-bin/source-query
support or arbitrary layered/material/resource correctness. The broader R5 MDL
and real MaterialX-generated matrix remains open; no provider/material scope has
been removed. No source replacement, advertisement, commit, install or push is
included.
