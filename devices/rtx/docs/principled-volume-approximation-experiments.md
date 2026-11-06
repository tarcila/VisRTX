# Principled volume approximation experiments

Baseline: `02e66fce` (ray/achromatic reuse and point-texture endpoints), with
`2658e96a` animation tooling. Experiments are sequential: retain and commit a
qualified improvement before starting the next; revert unsuccessful candidates.
Software interpolation and reference Quality transport remain the defaults.

## Acceptance policy

Compare against the committed baseline, not merely the slower disabled mode of a
new candidate. Check both enabled and disabled modes. A useful initial performance
signal is at least 5% less time on representative workloads; investigate default
path regressions above 3%. These are screening criteria, not universal guarantees.

Initial Interactive HDR gates remain relative L1 <= 0.005, normalized RMS <= 0.01,
and maximum alpha error <= 0.002. Black references require exact equality. These
bounded scene checks do not establish universal sampling error or motion quality.
Quality experiments must also assess time-to-error when the estimator changes.
Any deliberately looser approximation needs its own explicit error budget, not a
retroactive relaxation of a failed test.

Use bounded GPU-wide clock leases, reversed execution order, and telemetry.
Clock-policy configuration is not exposed here; do not infer its maximum duration.

## 1. Hardware-filtered Interactive optics — rejected

Implemented an opt-in `volumeHardwareFiltering` renderer parameter for the
practical FP32 principled camera integrator. Eligible scalar fields reused the
existing linear texture with canonical coordinate mapping. RGB, custom,
rectilinear, nearest-filtered, and texture-ineligible fields retained software
sampling. Quality, TF1D classification and shadow integration were unchanged.
The default remained point-texture endpoints plus software interpolation.

Two candidates were tested:

1. Runtime sampling-policy propagation through the camera coefficient evaluator.
2. Compile-time camera-integration specializations selected once per segment,
   avoiding the runtime policy in the inlined common NanoVDB sampling path.

The second improved the enabled path but still failed the benefit/regression
screen. All production and measurement-driver changes were reverted. No dormant
parameter, shader branch, new sampling policy or metadata remains.

### Clock-leased timing results

RTX 5880 Ada, 2400 MHz active graphics clocks; 640x640, one sample, rates 1/1,
no denoising/checkerboarding/firefly reduction. Two reversed-order repetitions;
discard two warmups from 24 subway launches and 12 canonical launches.
Ranges below are the two medians, not confidence intervals.

| Scene | Baseline ms (round 2) | Candidate disabled ms | Candidate enabled ms |
| --- | ---: | ---: | ---: |
| Subway | 23.76–23.96 | 23.99–24.20 | 23.27–23.56 |
| Cloud | 28.80–29.04 | 29.41–29.44 | 28.04–28.08 |
| Plume | 17.02–17.11 | 17.28–17.28 | 16.43–16.45 |
| Separate support | 6.73–6.74 | 7.23–7.29 | 6.89–6.91 |
| Overlap | 0.81–0.87 | 0.87–0.88 | 0.81–0.83 |

The first candidate measured 23.42–23.56 ms subway, 28.76–28.77 ms cloud and
16.86–16.97 ms plume when enabled, also insufficient. The second candidate's
roughly 2–4% gains on representative scenes do not justify the added policy and
approximately 8% disabled-path regression on separate support. This does not
prove hardware interpolation cannot help; it rejects these opt-in integrations.
Earlier unconditional experimental filtering had better results, but is not an
acceptable silent replacement of the reference preview policy.

### Fidelity and checks

The first candidate's 42 retained 128-square frames across subway, cloud, plume,
separate support, overlap and native/Matte receivers passed the unchanged gates:
maximum relative L1 0.000160, NRMS 0.000222, alpha error 0.000109. All 42 disabled
frames matched the baseline exactly. Release builds passed; stored-reference,
public Interactive and public Quality tests passed before benchmarking. The
second candidate was rejected on performance; no broad correctness/sanitizer or
motion qualification is claimed for it.

Both timing leases and the first candidate HDR lease exited successfully after
release/reset. Evidence: `_out/principled_volume/hardware-optics/` contains the
runner, gates, raw captures/timings/telemetry, lease logs, second-candidate patch
and frozen library. The first candidate binary was not separately frozen.

## Remaining sequence

2. Coarser/error-controlled camera integration: pending.
3. Reduced-frequency lighting evaluation: pending.
4. Reduced-resolution volume rendering: pending.
5. Mixed-precision Quality sampling: pending.
6. Baked optical representation: pending.
7. Improved Quality proposals/control variates: pending.
8. Approximate multiple-scattering tail: pending.
