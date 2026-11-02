# Quality source-importance grid

Status: implemented; bounded correctness/performance qualification passed.
This is a source proposal, not an optical majorant or an empty-space mask.

## Design

Each emitting volume can lazily own an object-space importance CDF. A CUDA kernel
samples the Quality source evaluator at64^3 cell centers, including
explicit and absorption-coupled thermal emission with independent field domains.
The pilot weight is linear luminance. CPU normalization uses scaled binary64
weights and publishes the realized CDF including0/1 endpoints. Metadata-constant
principled volumes use one uniform cell instead of allocating a64^3 grid.

The sampler mixes **90% spatial importance with10% of the previous full-support
angular-distance proposal**. It evaluates the full mixture density for either
origin. Spatial density is converted by the instance's absolute affine Jacobian
and by r² to the shared dω dr measure. Coefficients/source are never scaled by the
instance determinant. Sampling uses actual CDF interval widths, not ideal pilot
weights; zero-width CDF cells are valid proposal holes.

The angular term retains full Medium Domain support and its near-source protection.
Dark/missed pilot cells cannot suppress actual emission. An all-zero pilot,
allocation failure or unusable guide Jacobian falls back to angular-distance
sampling, without changing source eligibility/ownership or dropping occurrences.
There is no radiance/distance clamp, source majorant, validation scan or new
provider restriction. The Jacobian arithmetic envelope applies only to the optional
guide; the fallback remains available outside it.

The immutable CDF is owned by the volume and shared by its occurrences. Source or
binding finalization resets it; camera, light and instance edits reuse it. World
publication distinguishes requests with/without importance. Interactive does not
request these grids. An explicitly opted-in custom field without CUDA sampling
skips only this optional pilot; its OptiX source and angular proposal remain
available. Field snapshots and newly published CDFs follow the existing
supported wait/update/prepare lifecycle.

## Controls

Existing source/ownership gates are unchanged. CPU/CUDA/fast-math source fixtures
now also exercise uniform importance, a deliberately misleading CDF with zero mass
on emitting cells, near-source bounded weights, duplicates and non-unit-determinant
transforms. These estimate the same independent analytic integrals; no test seed,
sample count or numeric tolerance was retuned.

Public-commit World tests verify request-mode separation, unchanged-guide reuse,
RGB emission-field edit invalidation without a volume recommit, all-zero-pilot
fallback without source removal, and recovery. Public receiver/pane/absorption and
Quality source controls remain active.160 selected tests passed in720.14 s.
Source CUDA fast-math, committed World, public source lighting and a32-square
thermal-plume memcheck each reported zero errors/leaks.

## Matched-work checkpoint

Evidence: `_out/principled_volume/quality-importance/`. Relative to the angular
proposal, plume split256-vs256 NRMS fell29.61%->10.64%;128-square32-spp batch
time changed2025.06->2110.63 ms. Native receiver512-spp analytic NRMS fell
5.423%->1.478%, with mean0.010026835 versus0.010027149; batch time fell
20.270->19.218 ms. Receiver squared-error times cost improved about14x.
The plume split-error cost diagnostic improved about7.5x, not a universal
variance guarantee: disjoint QMC blocks are not independent replicates.

The plume guide kernel took10.18 ms once per profiled process. A64-cubed CDF
uses2,097,160 persistent device bytes and2,097,152 temporary device bytes, plus
host normalization storage. Metadata-constant sources use one cell. GPU-wide
memory includes CUDA module/context overhead and must not all be attributed to
the grid. The frozen timing library precedes the optional extreme-Jacobian
guard; the guard does not affect these ordinary fixtures.

Subsequent [Quality hot-path rounds](quality-volume-optimization.md) preserve
the full-support guide and reach roughly30 ms cloud/plume640-square samples.
Neither checkpoint establishes a heterogeneous convergence golden or universal
frame-rate guarantee.
