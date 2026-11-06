# Quality volume-source resampling

Quality accepts `volumeSourceResampling` (`ANARI_BOOL`, default **false**).
It draws two independent candidates from the existing full-support volume-source
proposal, selects one using its unshadowed source weight, and evaluates the
receiver kernel/visibility only for that selection. There is at most one visibility
query. Nothing is reused between pixels or frames; this is not temporal resampling.

```cpp
const int enabled = 1;
anariSetParameter(device, qualityRenderer,
    "volumeSourceResampling", ANARI_BOOL, &enabled);
anariCommitParameters(device, qualityRenderer);
```

This option can reduce time-to-error while increasing milliseconds per pixel
sample. It does not change optical coefficients, tracking rates, phase functions,
source support, or continuation/source ownership. It is independent of
`volumeFastSampling`; both may be enabled, with the latter's approximation caveats
still applying. Existing unsupported material/provider/source-query cases retain
their existing fallback; this does not expand source-lighting qualification or
advertise the unfinished principled-volume extension.

## Estimator

Let `a` and `b` be independent outputs of the existing labeled-occurrence source
sampler. Their vector weights already include emission, occurrence probability,
and the complete angular/spatial mixture density. There is no additional inverse
square factor to apply.

Define scores `sa=max(abs(a.weight))` and `sb=max(abs(b.weight))`, treating invalid
candidates as zero. Scale by `max(sa,sb)` before normalizing to avoid overflowing
the score sum. If both scores are zero, return zero after the two draws; do not
retry until a nonzero sample appears.

Select `a` with probability:

```text
p = 1/8 + (3/4) * sa / (sa + sb)
```

Select `b` otherwise. This mixes score-proportional selection with 25% uniform
selection, so each candidate has probability at least 1/8 even if a normalized
score underflows. Divide the chosen vector weight by twice its selection
probability. The correction is at most four.

For any receiver kernel/visibility evaluation `F`, the conditional expectation in
real arithmetic is:

```text
p * [a.weight * F(a) / (2*p)]
+ (1-p) * [b.weight * F(b) / (2*(1-p))]
= (a.weight * F(a) + b.weight * F(b)) / 2
```

Taking expectation over candidates therefore preserves the original estimator
mean. A stochastic visibility query uses its separate random stream after
selection. No visibility outcome or source draw changes source ownership.
Invalid/zero candidates remain legitimate zero samples with positive selection
probability when paired with a nonzero candidate. Signed RGB weights, if present
in an internal sampling context, are scored by magnitude and retain their signs.

The returned weight includes this selection correction; it must not be treated
as an ordinary marginal PDF in a future MIS scheme. Current qualified source
connections use disjoint ownership, not such a marginal-PDF interpretation.

## Scope and verification

The proposal is deliberately simple: the score excludes BSDF/phase orientation,
occlusion and medium attenuation. Highly directional or heavily occluded scenes
may benefit less or regress. Default sampling remains available. Numerical/RNG
limitations of the original source sampler are unchanged; this is not a claim of
bitwise equality when enabled.

CPU, CUDA and CUDA fast-math controls cover original and resampled estimators for
far/inside, duplicate, transformed and deliberately poor-guide fixtures. The
original analytic means and statistical tolerances are unchanged; the resampled
per-sample bound includes the mathematically required factor-four correction.
Public tests exercise metadata, mode toggling/default restoration, Matte/native
receivers, duplicate sources, holes, covered transmission, blockers, absorbing
media, emission updates, an empty source set and source restoration.

The [experiment record](principled-volume-approximation-experiments.md) records
paired randomizations, independent finite-reference error-cost comparisons,
combined fast-field sampling, timings and memory checks. These are bounded
results, not a universal convergence-speed guarantee.
