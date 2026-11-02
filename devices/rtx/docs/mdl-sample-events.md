# MDL sample consumption and event metadata

This implements the narrow absorbed-event prerequisite from the accepted R5
material/source design. It does **not** implement the native reflection repair,
volume-source replacement, or complete MDL/MaterialX closure qualification.

## Changes

`material/shaders/MDLSample.h::consumeMdlSample()` is used directly by the MDL
next-ray callable. It checks `BSDF_EVENT_ABSORB` before normalizing the normal or
consuming sampled direction/weight/PDF. An absorbed result is a fully initialized
Dead record: zero direction, weight, PDF, flags and raw event mask.

A generated target can write a candidate direction/positive weight and then reject
it as ABSORB. Zero-initializing the input sample record does not prevent stale
outputs in that case. Conversely, a live specular event with raw PDF zero is not
absorption and must survive.

`NextRay` now carries a separate event classification and raw MDL event mask:

- MDL live samples are Continuous or Singular according to the SPECULAR bit.
- The old through-surface test and environment-MIS PDF convention are retained.
  Continuous transmission may therefore have an infinite legacy PDF without
  being classified Singular. Raw SDK bits are not replaced by the geometric
  through flag.
- Matte and missing-callable fallbacks return explicit initialized Dead records.
- Native PBR was Unclassified at this checkpoint. The subsequent
  [native repair](native-reflection-qualification.md) adds explicit reflection/
  transmission components and measure tags. This MDL step itself did not alter
  native weights or authorize source replacement.

Quality and Interactive check explicit Dead before consuming the returned weight,
side flag or direction. Local direct lighting has already been evaluated at that
point, so Matte remains DirectOnly rather than losing its direct contribution.
No source-owner suppression, new MDL DF flags, parameter clamps, validity scans or
material allowlist was introduced.

The internal NextRay callable ABI changed; callers and built-in/MDL wrapper PTX
were rebuilt together. This is not a stable external material-plugin ABI promise.

## Evidence

The exact production consumer is compiled on CPU, CUDA and CUDA fast math. Nine
fixed records cover:

- rejection after writing a positive weight/direction;
- absorbed records with signaling-NaN output payloads and a zero normal;
- live diffuse/glossy reflection;
- live specular reflection/transmission with raw PDF zero;
- continuous transmission with an infinite legacy ownership PDF;
- mixed event-bit preservation and live zero-weight classification.

Event tags, flags, raw bits, live sample preservation and complete dead
initialization use exact checks. Host exception flags additionally detect invalid
arithmetic/division through poisoned absorbed inputs. The rejected-output producer
was an explicit **test stub** at this checkpoint. The subsequent
[generated-target qualification](mdl-generated-target-qualification.md) now
reproduces stale-ABSORB outputs in actual compiled mirror and GGX targets, using
the production wrapper/resources. Broader closure/source-query coverage remains
open before generic source replacement.

Verification:

- Full Release build; **152/152 selected CTests passed**, excluding only the MDL
  compile benchmark (690.20 s).
- CUDA-fast-math consumer and public `TestMdlSceneData` memcheck/full-leak checks:
  zero errors and zero leaked allocations. Existing MDL/MaterialX public tests
  remain compatibility coverage, not complete scattering-closure qualification.
- Logs: `/tmp/visrtx-mdl-event-qualified-{build,tests}.log`,
  `/tmp/visrtx-mdl-event-*-memcheck.log`.

The build still resolves the SDK root ending `mdl-sdk-2026.0.2`, whose installed
header/CMake version is 2025.0.3. This checkpoint does not replace the required
runtime generator identity/compiled-target ledger with the package basename.

The native kernel/proposal/PDF repair now has its own bounded qualification.
A first six-closure generated-target manifest now passes; broader MDL/MaterialX
source-query qualification remains ahead of generic MDL source replacement.
