# Profiling a Vela archive

This harness uses Vela to preserve geometry, materials, textures, layer transforms
and enabled states rather than recreating a simplified scene. It requires Linux,
a compatible `vsrLua`, `vsrPrint`, `vsrConvert`, ANARI and CUDA. Build with
`BUILD_QUALITY_BASELINE=ON` as described in `QUALITY_BASELINE.md`; this also builds
`libarchive_capture.so`.

A viewer application-state file is not accepted by `loadSceneArchive`. Extract its
embedded scene without altering the original:

```sh
python3 tools/principled_volume/extract_viewer_scene.py \
  test-principled-volume.vsr _out/archive-profile/scene.vsr
```

The canonical text intermediate can be several times larger than the binary;
provision temporary disk space. Existing output is refused. Input/output hashes
are written next to the extracted scene. Standalone scene archives need no
extraction.

Freeze the library before measuring. Example for controlled full-frame work:

```sh
mkdir -p _out/archive-profile/quality
LD_LIBRARY_PATH=/absolute/frozen/library:$LD_LIBRARY_PATH \
LD_PRELOAD="$PWD/_out/quality-baseline-build/libarchive_capture.so" \
vsrLua tools/principled_volume/profile_archive.lua \
  _out/archive-profile/scene.vsr "$PWD/_out/archive-profile/quality" \
  quality 640 24 1 static 0 controlled \
  > _out/archive-profile/quality/run.log 2>&1
```

Repeat for `interactive`. Arguments after renderer are image size, launches,
samples/launch, edit mode, uint32 random seed and settings profile. Edit modes are
`static`, `density`, `temperature`, plus diagnostic `volume-off` and `source-off`
ablations. The ablations change images and must never count as speedups.
Set `VISRTX_CAPTURE_CAMERA=1` to animate camera X with a one-scene-unit amplitude;
this is a public camera parameter edit before every render.

The controlled profile uses depth32, rates1/1, black transparent backdrop, no
ambient, checkerboard, denoising or firefly filtering. Camera0 is taken from the
archive; CameraSetup takes degrees while its stored ANARI FOV is in radians.
The optional `viewer` profile is **specific to the recorded subway target's
settings**, not automatic renderer-settings discovery. It enables Quality's
recorded checkerboarding/denoising/depth12 and Interactive's ambient/tonemap
settings. Report those partial-update latencies separately from full-frame work.

`CAPTURE,index,samples,gpu_ms,wall_ms,gpu_used_bytes` records are printed to stderr.
The wall interval covers ANARI render/prepare/wait, not Lua parameter setters,
image mapping, encoding or disk I/O. Camera edits performed by the interposer are
inside that interval. GPU memory is a GPU-wide reading; collect process/clock
telemetry and check external contention before attributing differences.

For raw HDR, set `VISRTX_CAPTURE_HDR` to the absolute output directory. The
interposer requests ANARI float4 color, saves the linear values on CUDA mapping,
and supplies an sRGB8 display copy to Vela. Display conversion and file writes
are outside render timing. This mode deliberately changes output conversion;
use the unchanged-format mode for throughput. Seed marshalling converts Lua's
signed integer parameter into Quality's declared uint32 type, without changing
its bits. Capture logs print the requested seed.

For Nsight Compute use `--clock-control none`, preserve external clock locks, and
profile one warmed `regex:optixLaunch` after skipping two launches. Nsight Systems
CUDA traces distinguish guide construction, acceleration, rendering and conversion.
Profiler replay time is not a throughput result. Source PC shares are not elapsed
time partitions.

The subway campaign freezes its gates in `qualification-plan.json`, compares
8 paired512-spp randomized Quality estimates, and keeps deterministic Interactive
HDR gates unchanged. `analyze_archive_performance.py ROOT` analyzes its evidence
layout and raises on its predeclared target fidelity failures. See
`devices/rtx/docs/principled-subway-performance.md` for results and rejected
candidates. The original archive is not redistributed by these tools.
