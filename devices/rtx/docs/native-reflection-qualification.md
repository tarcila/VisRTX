# Native reflection repair — implementation/qualification checkpoint

Status: implemented with bounded qualification passed. Source replacement
remains disabled. This follows the accepted R5 reflection kernel/full-support
proposal, retaining the old transmission approximation.

## Frozen bounded controls

Before executing the new measure tests:

- Ten fixtures: diffuse, rough metal, dielectric, independent coat-only support,
  mixed normals/base/coat, sheen-only, rough transmission, exact eta-one atom,
  TIR/fold and a tilted base normal relative to the geometric normal.
- 262144 fixed-seed draws per fixture, on CPU, CUDA and CUDA fast math. Seed recipe
  is fixed in `nativeReflection/fixtures.h`; no retry seeds or adaptive sample count.
- Eight directional bins on the geometric-front hemisphere (two z bands, four
  azimuth sectors). Rejected and transmitted draws remain in the denominator.
- Independent double-precision kernel and proposal algebra; the oracle does not
  call production BSDF helpers. Bin references use midpoint grids 256x512 and
  512x1024. Required agreement: 2e-4 absolute in every probability/RGB integral.
- Pointwise PDF and K/q: absolute 2e-6 plus relative 5e-4. Points within 1e-5 of
  base/coat support boundaries are counted separately, not used for smooth-point
  comparisons; they remain in all sampling/bin tests.
- Sampled probability/weighted-RGB bin means: six estimated standard errors plus
  5e-4 absolute. These stochastic gates do not replace image or optical gates.
- Exact tags, dead initialization and conditional RNG consumption; transmitted
  O outcomes retain direction/measure and receive only inverse 7/8 compensation.
- No claim of complete iridescence, near-singular or all-input float qualification
  from these bounded fixtures. Additional boundary/actual-renderer controls are
  needed before promoting the complete source-replacement design.

The production common kernel, PDF, sampler and native Quality support gate change
together. No volume-source owner is enabled by this candidate. Any unresolved
measure/reference discrepancy blocks calling this qualification successful.

## Implemented target and proposal

The native shader now shares `pbrReflectionKernel` between ordinary-light queries
and all reflected continuation weights. It applies geometric-front support,
independent base/coat gates, nonnegative diffuse Fresnel complement, the retained
TIR/macro-wrong-side fold, coat-exit attenuation and additive sheen. The accepted
cosine/Smith/Charlie regularizers remain target-model approximations.

`pbrBsdfPdf` is the unconditional mixture of the retained directional proposal
(7/8) and a uniform geometric hemisphere (1/8). Its VNDF density uses the ideal
visible-normal Jacobian, not the target's floored Smith helper. Reflected weights
are total K/q, without probability-denominator floors. This supplies sheen
continuation even when the old proposal's base split is entirely dead.

The retained O sampler preserves its conditional draw order and old transmitted
weight formula, including its approximation/floors. Transmitted weights acquire
only inverse 7/8 compensation. Exact eta=1 is classified Singular with direction
-V; adjacent represented eta values remain Continuous. No new eta-squared factor
or physical-transmission claim is introduced. Native reflections, transmissions
and dead outcomes now carry explicit event/component tags; MDL retains its raw
SDK bits and separate MDL component.

Quality delegates native support to the common kernel instead of additionally
rejecting ordinary-light directions behind the old Ns. Non-native gates and
surface HDRI/cosine proposal axes remain unchanged. The geometric normal is now
retained in the native shading state. Compatible built-in/MDL wrapper/caller PTX
were rebuilt for the changed internal state/NextRay layouts.

## Important integration finding: independent outer choice

The first public four-pane test was about 14% too dark although the local
CPU/CUDA measure controls passed. The initial implementation took the new 1/8
outer coin from a Sobol dimension reused with per-bounce scrambling. Its dyadic
prefix intervals were correlated across successive panes, so survival was not
the product of the per-vertex probabilities.

The outer coin now uses the path's **PCG stream** independently of the Sobol
overlay. The retained O proposal keeps its old overlay order. No mixture weights,
PDFs, mean thresholds or test seeds were changed. The four-pane red mean moved
from 0.177168 to 0.20749 against the analytic 0.20736. This does not claim to
qualify every existing multidimensional Sobol path strategy.

A separate one-sample HDR assertion assumed every reflection was a narrow mirror
sample. The intended 1/8 broad proposal invalidates that premise. Its replacement
requires the narrow branch's bright population at six binomial standard
deviations; all-pixel finite checks and the existing paired HDR/fog arithmetic
limits remain unchanged. This is a sampling-contract update, not a looser affine
radiance tolerance.

## Results and remaining coverage

- All ten frozen measure fixtures pass CPU, CUDA and CUDA-fast-math tests with
  the original reference/point/bin gates. Near-support point comparisons are
  reported separately; every draw still participates in the bin tests.
- Additional fixed controls pass for target cosine/Smith/roughness boundaries,
  forced tiny coat/specular strata, sheen-only support, exact/neighboring eta-one
  classifications, conditional transmitted compensation and overlay/PCG ordering.
- Public Quality sheen-only rendering under a constant HDRI matches the independent
  normal-view Charlie integral `sheenColor / (8*(4+1e-6))`. Existing native and MDL
  public-renderer compatibility tests pass.
- Full Release build; **155/155 selected CTests passed**, excluding only the MDL
  compile benchmark (655.33 s). Final metadata assertions and targeted tests were
  rebuilt/rechecked afterward.
- Native CUDA-fast-math measure tests and public `TestPrincipledQuality` pass
  memcheck/full-leak checks: zero errors, zero leaked allocations.
- Logs: `/tmp/visrtx-native-qualified-tests.log`,
  `/tmp/visrtx-native-final-{build,tests}.log`, `/tmp/visrtx-native-*-memcheck.log`.

This remains bounded evidence, not universal finite-precision/proposal or physical
energy-conservation certification. Complete iridescence/near-singular/material
coverage and the actual generated MDL/MaterialX source-query qualification remain
open. No source replacement, material allowlist, extension advertisement, install,
commit or push is introduced by this step.
