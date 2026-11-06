# Opt-in Quality field sampling

Quality accepts `volumeFastSampling` (`ANARI_BOOL`, default **false**). This is an
experimental numerical approximation of eligible scalar fields, not a change to
the optical input model or an unbiased estimator of the canonical interpolant.
It does not advertise the unfinished principled-volume extension.

```cpp
const int enabled = 1;
anariSetParameter(device, qualityRenderer, "volumeFastSampling", ANARI_BOOL, &enabled);
anariCommitParameters(device, qualityRenderer);
```

## Scope and fallback

The fast path requires the aggregate scene's existing `previewFloatEligible`
envelope. That includes ordinary raw/effective range endpoints and bounded field
coordinate mappings. Every other scene uses reference sampling. Within eligible
scenes, acceleration additionally requires:

- Scalar stored data, linear interpolation, no rectilinear axes/custom provider.
- A prepared point texture containing canonical scalar endpoints.
- Dense canonical scalar storage or a NanoVDB Float grid; packed NanoVDB formats
  and all RGB fields retain reference sampling.
- Indices within the existing +/-2^18 mapping envelope.

Missing/oversized/unavailable textures fall back without changing validity.
There is no new field scan, allocation policy, runtime invalid-data recovery,
source-support restriction, work cap or overlap cap. The point arrays were already
prepared by eligible principled volumes; their immutable snapshot ownership and
array-update invalidation remain unchanged. The frame-local renderer flag never
mutates shared field views. Interactive sampling is unchanged.

## Numerical contract and conservative bounds

`QualityFieldSampling.h` maps coordinates and performs software interpolation in
FP32 using **point** texture fetches. It never uses hardware linear filtering.
For each blend, it computes `a*(1-w) + b*w` and clamps rounding back between the
endpoints. Successive blends therefore stay in the global canonical raw hull
for ordinary values.

CUDA fast-math can flush interior subnormals to zero. Eligibility matters here:
raw hull endpoints must be zero or ordinary finite values. If that hull contains
subnormal values, it necessarily also contains zero. A blend of same-sign normal
endpoints that underflows intermediate products is clamped back to the normal
endpoint interval. Thus flushing does not escape an **eligible global raw hull**;
this is not a claim that every subnormal local endpoint interval is preserved.
Coordinate clamping fetches only stored index-domain endpoints. Point arrays use
the same canonical decoder as reference range construction, including sparse
inactive/tile/background values.

Binding-domain membership, outside values, effective-value multiplication,
extinction logarithms, A/S partitioning, thermal coupling, RGB arithmetic,
ray/flight state, optical-bound construction and proposal rates retain their
reference graphs. Because sampled scalar values stay inside the certified raw
hulls, the unchanged binding/optical bounds remain conservative for the new
interpolant. Overlap bounds still sum conservatively. This argument is for the
current global bounds, not future local-cell certificates: FP32 mapping can
select a different interpolation stencil and would need matching local bounds.
The approximate preview shadow cache is **not** a tracking majorant.

This argument depends on the ordinary-hull rule in `PreviewEligibility.h` and on
canonical endpoint preparation. Changing those rules requires revisiting this
path, not merely relaxing an optimization heuristic. Small tests alone are not
the proof. Conversely, the bound argument is not a universal image-error bound:
coordinate/interpolation error can be amplified by sharp fields or thermal
emission. Users needing canonical reference sampling should leave the option off.

## Bounded qualification

See [experiment 5](principled-volume-approximation-experiments.md) for measured
speed, statistical comparisons and verification. Tests cover:

- 3,993 blends over signed ordinary, zero and interior-subnormal endpoints, on
  CPU and CUDA fast-math, checking the eligible global hull.
- Point-texture fixtures with negative indices, background values, both
  centerings, dense array revisions, and packed-format fallback. The fast path
  directly exercises 20,480 points across NanoVDB Float and dense fixtures,
  including subnormal endpoints inside an ordinary signed raw hull.
- Public metadata default, opt-in rendering, return to exact default behavior,
  density array updates, nearest-filter fallback and restoration to linear.
- Fixed-work seed-randomized comparisons and error-cost diagnostics, retaining
  finite reference uncertainty rather than claiming converged goldens.

The qualification is bounded to the recorded fixtures and hardware. It does not
certify every extreme coordinate, provider, thermal scale or scene distribution.
