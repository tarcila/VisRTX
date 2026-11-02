#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
if [ "$#" -ne 6 ]; then
  echo "usage: run_quality_baseline.sh binary library assets output phase library-directory" >&2
  exit 2
fi
binary=$(realpath "$1"); library=$2; assets=$(realpath "$3"); base=$(realpath -m "$4"); phase=$5
export LD_LIBRARY_PATH="$(realpath "$6"):${LD_LIBRARY_PATH:-}"
mkdir -p "$base"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$base/$phase-processes-before.csv"
nvidia-smi --query-gpu=timestamp,clocks.gr,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv -l 1 > "$base/$phase-telemetry.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
run() {
  local name=$1 renderer=$2 scene=$3 size=$4 spp=$5 frames=$6 depth=$7 rate=$8 profile=$9
  local output="$base/$name" status=0
  mkdir -p "$output"
  local args=("$binary" "$library" "$renderer" "$scene" "$assets" "$output" "$size" "$spp" "$frames" "$depth" "$rate")
  if [ "$profile" = yes ]; then
    timeout 240 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o "$output/trace" "${args[@]}" > "$output/run.log" 2>&1 || status=$?
    if [ "$status" = 0 ]; then nsys stats --report cuda_gpu_kern_sum --format csv "$output/trace.nsys-rep" > "$output/kernels.csv";fi
  else
    timeout 240 "${args[@]}" > "$output/run.log" 2>&1 || status=$?
  fi
  printf '%s,%s\n' "$name" "$status" | tee -a "$base/statuses.csv"
}
case "$phase" in
  perf)
    for repetition in 1 2;do
      scenes='cloud plume support overlap receiver-matte receiver-native'
      if [ "$repetition" = 2 ];then scenes='receiver-native receiver-matte overlap support plume cloud';fi
      for scene in $scenes;do
        profile=no
        if [ "$repetition" = 1 ] && { [ "$scene" = cloud ] || [ "$scene" = plume ]; };then profile=yes;fi
        run "perf-$scene-$repetition" quality "$scene" 640 1 12 32 1 "$profile"
      done
    done;;
  convergence)
    for scene in cloud plume support overlap receiver-matte receiver-native;do
      run "convergence-$scene" quality "$scene" 128 32 18 32 1 no
    done;;
  depth)
    for depth in 12 64;do
      for scene in cloud plume;do
        run "depth-$scene-$depth" quality "$scene" 128 32 18 "$depth" 1 no
      done
    done;;
  support)
    for rate in 2 4;do
      run "support-preview-$rate" interactive support 128 32 18 32 "$rate" no
    done;;
  profile)
    for scene in cloud plume;do
      for kind in profile fp64;do
        output="$base/$kind-$scene";mkdir -p "$output"
        sections=(--metrics sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed)
        if [ "$kind" = profile ];then sections=(--section SpeedOfLight --section Occupancy --section InstructionStats --section SourceCounters);fi
        timeout 240 ncu --clock-control none --target-processes all --kernel-name 'regex:optixLaunch' --launch-skip 2 --launch-count 1 \
          "${sections[@]}" -o "$output/counters" "$binary" "$library" quality "$scene" "$assets" "$output" 640 1 4 32 1 > "$output/run.log" 2>&1
        reports=("$output"/counters.ncu-rep*)
        if [ "${#reports[@]}" -ne 1 ] || [ ! -f "${reports[0]}" ];then echo "missing/ambiguous NCU report" >&2;exit 1;fi
        if [ "$kind" = profile ];then
          ncu --import "${reports[0]}" --page details --csv > "$output/details.csv"
          ncu --import "${reports[0]}" --page source --print-source sass --csv > "$output/source.csv"
        else ncu --import "${reports[0]}" --page raw --csv > "$output/metrics.csv";fi
      done
    done;;
  *) echo "unknown phase: $phase" >&2;exit 2;;
esac
