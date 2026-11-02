# Seed-randomized Quality convergence and error-cost checkpoint

> Sampling integration status: recorded images, convergence and timings predate
> the Halton-camera / compact Sobol-transport split. They are historical evidence,
> not qualification of the current sampling policy. Seed zero preserves the new
> Halton baseline, not the former all-Sobol images. See the [sampling contract](../apps/tests/unit/sobol/README.md)
> for the seed policy and outstanding rendering requalification. Optical/unit
> checks that do not consume camera or transport samples remain applicable.

## Reproducible randomization

Quality now accepts `randomSeed` (`ANARI_UINT32`, default0). It randomizes the
camera/surface Owen keys, general PCG state, and the separate medium, shadow,
volume-source and coverage streams. Zero preserves the previous sequences;
the matched plume512-spp HDR average is bit-identical. Changing/restoring the
seed resets accumulation and is reproducible against a fresh scene. The default
center-camera first sample is retained; this study excludes the initial two
launches rather than treating that fixed sample as randomized pixel quadrature.
Other renderers keep their existing default-zero behavior.

This is deterministic pseudo-random replication, not a claim that hash-based
scrambling is an ideal independent random permutation. Existing seeds and
numeric/statistical regression tolerances were not retuned.

## Frozen experiment

`tools/principled_volume/run_quality_replicates.py` writes and checks a frozen
`plan.json` before launching. Each of six cases uses eight512-spp candidate
replicates and eight separate4096-spp reference replicates, at128-square,
32 samples/launch and depth32. Seeds are SHA256-derived from fixed candidate/
reference labels and indices, not selected from output. Binary/asset hashes are
part of the plan. The study completed96 process runs.

`analyze_quality_replicates.py` reconstructs retained samples from actual frame
counts, estimates reference-mean noise, and subtracts that estimated noise from
candidate mean-squared reference differences. It reports signed corrected MSE;
negative estimates are not manufactured into a successful NRMS result. Global
RGB mean checks use the predeclared6 replicate standard errors +5e-5 absolute
criterion. All candidate/reference checks and analytic overlap/receiver checks
passed. Heterogeneous references remain noisy estimates, **not convergence
goldens**. Error-cost timings are128-square study timings, not FPS evidence.

| Case |32-spp NRMS |128-spp NRMS |512-spp NRMS | Estimated reference-mean noise |
|---|---:|---:|---:|---:|
| Cloud |21.45%|10.80%|5.37%|0.673%|
| Thermal plume |26.42%|14.64%|7.10%|0.844%|
| Separate support |7.93%|3.84%|1.99%|0.233%|
| Homogeneous overlap |10.09%|5.03%|2.52%|0.316%|
| Matte receiver |5.93%|2.96%|1.48%|0.186%|
| Native receiver |5.93%|2.96%|1.48%|0.186%|

These curves broadly follow Monte Carlo convergence; they do not establish an
asymptotic floor or prove absence of rare-event tails. Cloud/plume remain the
largest residual error-cost workloads.

## Exact-graph optimization retained

The common achromatic-media path repeatedly divided identical RGB event masses.
`volumeTracking.h` now reuses those divisions and continuation factors, and
reuses shadow factors when executed extinction is exactly equal across RGB.
It preserves all three original rounded mass additions, event ordering, CDF
endpoints, represented probabilities, colored throughput/source and RNG draws.
Chromatic inputs retain the generic path. No optical arithmetic certificate,
proposal rate, support policy or estimator law changes.

A frozen pre-optimization probability graph checks exact CDF, event, throughput,
source and shadow equivalence across scales, ownership and RGB cases on CPU/CUDA/
fast-math. `compare_quality_replicates.py` rerendered all48 candidate replicates:
all768 retained accumulation images match the baseline byte-for-byte. Therefore
the measured time ratios are also error-cost ratios for these samples:

| Case | New / old GPU error-cost |
|---|---:|
| Cloud |0.967|
| Plume |0.965|
| Separate support |0.919|
| Overlap |1.009|
| Matte receiver |0.999|
| Native receiver |0.981|

This is a modest improvement, not a universal win. Separate640-square one-sample
runs measured cloud29.95->28.42 ms and plume29.35->28.37 ms; optimized maxima
were29.24/30.25 ms over18 warm launches. Graphics telemetry throughout the main
replicate study stayed2400 MHz. External clock settings were untouched.

## Qualification and remaining work

-165/165 selected tests passed in652.09 s (compile benchmark excluded as before).
- Memcheck with OptiX tracking passed for tracking CUDA fast-math, single-device
  dynamic/seed updates, public Quality and public source lighting: zero errors/
  leaks. The separately reproduced multi-context sanitizer issue remains open;
  see `quality-volume-dynamic.md`.
- Main evidence: `_out/principled_volume/quality-replicates/` (`plan.json`,
  `analysis.json`, raw frames, timings, telemetry and frozen binaries).
- Optimization evidence: `_out/principled_volume/quality-replicates-gray/`
  (`comparison-plan.json`, `comparison.json`, raw matching frames and timings).

The next ordered implementation stage is custom-field typed opt-in. More
aggressive variance reduction for cloud/plume remains worthwhile, but requires a
separate estimator experiment; this checkpoint deliberately keeps the qualified
transport law rather than trading away the sample-speed target or support.
