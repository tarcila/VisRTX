# Preview lattice: finite-precision assessment

This record describes the actual Interactive/default preview selector in
`gpu/renderer/volumePreviewLattice.h`, shared by production camera and shadow
integration. It is a bounded assessment, not a universal refinement/error proof.
No rate cap, optical clamp, content scan, new frame-failure API, iteration cutoff,
or multiword arithmetic was introduced.

## Selection rule

Each active occurrence anchors its lattice to the entry of its full object domain,
not to the current ray clipping plane, surface endpoint, or field ROI boundary.
Object-space metadata supplies the step hint. The hint divided by sampling rate
and inverse-transformed direction length determines world-distance spacing.
Inverse-transformed directions are not renormalized.

The selector finds a candidate after the cursor for every active occurrence and
returns the earliest candidate or the current represented span endpoint. A rounded
tie tries the next integer cell identity. If neither candidate advances, that
occurrence supplies no additional split. Other overlapping occurrences can still
refine the span. This is not an epsilon jump or a dropped remainder interval.

## Corrected avoidable overflow

The old direction length squared its components directly. For a large finite
inverse transform, this could overflow to infinity and make the step zero even
when the requested world-distance cells were representable. A reproducible case
failed identically on CPU, CUDA and CUDA fast math before the fix.

The fixture uses the exact inverse of this finite float affine linear part:

```
[ 2^-110  2^110       0 ]
[      0  2^-110  2^110 ]
[      0       0  2^-110]
```

For a world ray along +Z, the object direction contains `2^550`. Squaring it
overflows double. Nevertheless, the tested interval `[2^-550, 2^-549]` and its
requested quarter-sized cells are representable. The old selector returned the
span end; the corrected selector returns `5 * 2^-552`.

The implementation now scales direction components by their maximum magnitude,
computes the norm of those scaled components, and divides the hint by rate, scale,
and scaled norm in stages. It never needs to form the large squared norm or
`rate * speed`. This fixes the demonstrated arithmetic issue without changing
sampling-rate semantics, reinterpreting world distance, or altering the optical
reference arithmetic. The fixture qualifies the selector, not OptiX rendering of
that ill-conditioned transform end to end.

## Limits that remain

1. **Unresolvable cell identities or boundaries.** At large distance-to-cell-size
   ratios, incrementing a double cell index by one or two may not change it. Even
   with a distinguishable index, adding a tiny offset to a large anchor may round
   back to the cursor. The selector then uses another occurrence's candidate or
   the span endpoint. Constant source integration preserves the represented
   length, but heterogeneous source/extinction can be under-resolved. No bound on
   that heterogeneous quadrature error is claimed.
2. **Collapsed intersection endpoints.** For example, a unit domain seen from
   `-2^54` along +X has distinct exact entry/exit distances that round to the same
   double. The aggregate slab intersection then reports no nonempty interval.
   This occurs before lattice selection, so remainder integration cannot repair
   it. One test explicitly characterizes this limitation; it does not certify the
   physical result as correct or declare the medium intrinsically empty.
3. **Sample-position precision.** Object-ray arithmetic and final float sample
   positions have their own precision limits. Preserving a distance interval does
   not prove accurate field lookup at every transformed location.
4. **Cost of fine representable lattices.** An extremely high rate can still
   request extremely many distinct cells. There is no hidden iteration budget.
   The high-rate test inspects one step, not completion time for an enormous march.

Use reasonably local scene/ray coordinates where practical. These observations do
not add a public scene restriction, input clamp, automatic recovery mechanism, or
new promise about all finite transforms.

## Bounded checks

`volumeReference/testPreviewLattice.h` exercises 18 cases through the production
selector on CPU, CUDA and CUDA fast math:

- ordinary boundaries, represented ties and the preceding representable cursor;
- clipping without reanchoring, endpoint clipping, reversed direction;
- merged translated occurrence lattices and permutation invariance;
- object/world scale and a non-axis direction of length five;
- a cell-index precision fallback and a large-anchor boundary fallback;
- an unresolved occurrence alongside another occurrence that can refine;
- inactive occurrences excluded from step selection;
- a finite `2^127` rate without an artificial ceiling;
- the large-inverse overflow regression;
- a complete four-cell partition with constant-source integration;
- explicit characterization of the pre-step endpoint-collapse limitation.

The selected next endpoints match independent dyadic/analytic expectations;
finite endpoints strictly advance and stay within the represented span. Vacuum
source integration is checked at the selected lengths. The test-only loop guard
for the four-cell partition is not present in production.

Final validation: all three lattice tests and eleven related public transport,
material, sample-addressing, fog, aggregate and optical-reference regressions
passed (**14/14**). Compute Sanitizer memcheck/full leak checks passed both the
CUDA-fast-math lattice executable and the expanded public Interactive test:
**zero errors and zero leaked allocations**. No performance or convergence
campaign was performed. Extension advertisement and Quality/custom-provider work
remain separate.
