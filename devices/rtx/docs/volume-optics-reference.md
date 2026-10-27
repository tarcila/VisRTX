# Reference optical evaluation and bounds

`device/gpu/volumeOpticsReference.h` is an isolated CPU/CUDA seam in
`visrtx::optical_reference`. This record describes its initial numerical
qualification. Experimental Interactive/default now call it through the
[aggregate layer](aggregate-volume-reference.md) and
[preview transport](interactive-volume-preview.md); other renderers retain legacy
transport. Optical-core tests alone do not qualify complete renderer behavior.

## Interface and preconditions

- `evaluate(values)` returns double absorption/scattering and explicit emission.
  `evaluate(values, blackbody)` additionally performs absorption-coupled thermal
  emission with the existing injected calibrated-RGB provider convention.
- Phase parameters are unchanged and remain in `PrincipledVolumeValues`; this
  slice does not introduce phase sampling or narrow coefficients into float phase
  kernels.
- Inputs are valid **effective float32 properties**: finite density >=0,
  transmission color in (0,1], distance >0 and albedo in [0,1]. Source products
  must be finite/representable where consumed. There is no per-sample validation.
- `EffectiveRanges` enclose those effective values, not raw field samples.
  `buildBounds` checks interval metadata and publishes only on success. Failure
  means the supplied certificate is unavailable, not detected invalid contents.
  Even a zero density does not waive required interval validity.
- `Bounds` retains absorption/scattering bounds separately. `combine` and
  `combineBounds` must process the same occurrence order/tree, counting repeated
  occurrences. `majorant` bounds the recombined RGB extinction. Callers must
  establish a finite usable aggregate before tracking; this helper has no scene
  eligibility/failure policy.
- `evaluateTransfer` takes committed **unit distance**, not its float reciprocal.
  It retains the legacy TF opacity ceiling `1-2^-24` and epsilon `1e-7`; nonpositive
  distance returns zero. `buildTransferBounds` requires a positive finite distance
  range. No existing TF caller has migrated to this path yet.

Host execution requires IEEE binary32/binary64, round-to-nearest, gradual
underflow, and no unsafe reassociation/fast-math. Host RN operations use volatile
binary64 intermediates; host upper operations expand the RN result one ULP
upward. CUDA uses explicit RN/RU double intrinsics. Float promotion expands bits
into binary64 directly, preserving subnormals under CUDA fast math; required
float narrowing uses explicit non-FTZ `cvt.rn.f32.f64`. Neither vector operator
contraction nor approximate float logarithms define this new arithmetic graph.
Host bounds can be looser than GPU bounds; evaluated coefficients should agree
bitwise under the stated environments. A different host/compiler environment
needs qualification, not an assumption that `volatile` overrides its FP mode.

## Physical log approximation versus execution enclosure

For positive float32 C<1, bit reduction gives exactly

```
C = m * 2^-k,  .5 <= m < 1,  0 <= k <= 148
z = (1-m)/(1+m)
P20(z) = 2 * sum(j=0..19) z^(2j+1)/(2j+1)
-ln(C) = k*ln(2) + P20(z) + R20(z)
0 <= R20(z) <= 2*z^41 / (41*(1-z*z))
```

`1-m` and `1+m` are exactly representable in double for these float mantissas.
The RN graph computes z, z squared, successive odd powers, positive terms and
sums separately. ln2 uses the same polynomial at RN(1/3), not host `libm` or a
separately rounded decimal constant. C==1 returns exactly zero. The expression is
an approximation to the physical log, **not correctly rounded libm**.

At z<=1/3, the tail is at most 1.505e-21; its relative bound using P>=2z is
2.257e-21. For a polynomial term of degree 2j+1, z error contributes that many
rounding factors, z-squared/power operations at most 2j further factors, division
one, and the positive sum at most twenty. Including the ln2 path and final
multiply/add remains below a conservative gamma512 allowance
(`gamma_n=n*2^-53/(1-n*2^-53)`). The nonzero powers through degree39 remain normal
binary64 for float32 C<1. Tail plus this allowance is below 2^-40 relative.
This argument is specific to this operation graph and input format.

The majorant does **not** treat that accuracy bound as a fudge factor. Within each
exponent bin, the RN evaluation is monotone decreasing with C: numerator falls,
denominator rises, and the remaining polynomial operations are monotone positive
operations. `negativeLogUpper` enumerates all intersecting exponent bins and
upper-evaluates each bin's smallest contained float. Its RU graph encloses the RN
graph directly. Max across bins avoids assuming global RN monotonicity across
range-reduction boundaries. The singleton C1 case has upper bound exactly zero.

## Extinction and aggregate argument

The fixed double execution is

```
E = RN(RN(Lhat * density) / distance)
S = RN(E * albedo)
A = RN(E * RN(1-albedo))
EU = RU(RU(LU * densityUpper) / distanceLower)
AU = SU = EU
```

Each represented partition multiplier is in [0,1], so each RN product is <=E.
This simple construction deliberately bounds A and S independently; the
recombined bound can be nearly twice E. No albedo-range capability is needed.
Exact density-zero and C1 branches stay exactly transparent. Effective float32
magnitudes keep these per-volume intermediates normal and finite in binary64;
this does not certify arbitrary emission products or an arbitrary aggregate.

For nonnegative terms, monotonicity of RN and RU addition gives an inductive
bound on separately accumulated A/S and their final sum. Changing reduction
order, narrowing the data, or evaluating through a different sampler/log graph
invalidates that specific argument. It is not repaired by clamping a null weight.

TF classification performs the retained alpha policy and RN32(1-alpha), then
uses the same log graph and double distance division. Its bound encloses all
exponent bins of the resulting transmission interval, including the epsilon
branch. TF scattering is zero; coupled emission is evaluated from absorption.

## Tests and evidence

Sources: `apps/tests/unit/volumeReference/` and thin
`test_VolumeOpticsReference.cpp/.cu` entry points.

- `oracle.py` independently computes Decimal90 logarithms, without the production
  polynomial. The checked-in fixture covers 701 float values: every power-of-two
  boundary and representable neighbors, subnormal/normal transitions, C1, and a
  fixed-seed sample of the remaining positive float bit range. `--write` explicitly
  regenerates it; ordinary CTest verifies the existing fixture.
- Every case checks the physical-log error, singleton and range log enclosures,
  physical partition coefficients, explicit/thermal source, TF policy and coupled
  source. Extreme density/distance/albedo combinations exercise overflow-prone
  float inputs that have finite double results.
- Ordered reductions include seventeen repeated principled occurrences and nine
  TF occurrences, checking component bounds and final majorant. Metadata rejection
  and zero-bound tests check publication and exact-zero behavior on host.
- CUDA runs the actual header helpers, both with and without `--use_fast_math`.
  Each GPU result is checked against the independent expected values and enclosure
  conditions, and RN log/partition/thermal/TF values against host results exactly.
  Missing GPU is a failed new CUDA qualification, not a successful skip.
- Physical coefficient and TF expectations also use independent host long-double
  arithmetic/log, with a predeclared relative 2^-38 allowance and exact zero checks.
  That finite test tolerance does not replace the above execution-bound argument.

Commands (existing Release RTX build):

```
cmake --build _out/_cmake --config Release --target \
  rtx_test_VolumeOpticsReference rtx_test_VolumeOpticsReferenceCUDA \
  rtx_test_VolumeOpticsReferenceCUDAFastMath -j 4
ctest --test-dir _out/_cmake -C Release \
  -R '^rtx::(VolumeOpticsReference.*|VolumeLogOracle|VolumeOptics.*|Sobol|SobolTableHeader|SampleOrdinal)$' \
  --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 \
  _out/_cmake/Release/rtx_test_VolumeOpticsReferenceCUDAFastMath
```

Initial execution passed all ten selected tests and memcheck reported zero errors.
The fast-math test translation unit was also compiled to compute_89 PTX with
nvcc13.4.92/O3 using its build command's includes/definitions. Inspected output
contained explicit f64 RN/RP operations and non-FTZ narrowing; no `.ftz`,
`lg2`, approximate reciprocal or fused multiply-add appeared. This is qualification
of the test-kernel seam, **not by itself a qualification of integrated OptiX transport**.
Future consumers/compiler changes must preserve that graph and requalify it.

## Explicitly not completed

The subsequent [reference field-binding bridge](reference-field-bindings.md)
adds controlled raw-to-effective enclosures and typed revision-checked preparation.
The [stored reference adapters](stored-field-reference.md) now add controlled
backend decoding/sampling, field-owned resources and finalized revisions. The
[principledVolume object](principled-volume-object.md) now owns prepared bindings
and bounds. Prepared-scene and experimental Interactive integration are now
covered by their separate records. Quality flight/source-light transport remains
pending. Legacy hardware-filtered fields are not certified by passing these
optical-core tests.
No full-format progress guarantee, performance result, universal finite-test proof,
extra validation scan or new runtime frame-status mechanism is introduced.
