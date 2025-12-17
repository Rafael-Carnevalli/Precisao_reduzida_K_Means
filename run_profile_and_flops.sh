#!/usr/bin/env bash
set -euo pipefail

# run_profile_and_flops.sh
# Usage: edit variables below, then run once: ./run_profile_and_flops.sh
# What it does:
# - compiles the CUDA binary (nvcc)
# - starts GPU power logger (samples every 0.5s)
# - runs the K-means binary and captures its stdout
# - stops the logger, cleans the log, computes avg/max power, and estimates energy
# - computes an estimated total floating-point operations (MFLOP) and MFLOPS

# ------------------- Configuration -------------------
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT_DIR/kmeans_1d_cuda_fp64.cu"
BIN="$ROOT_DIR/kmeans_1d_cuda_fp64"
DATA="$ROOT_DIR/../DadosEntrada/dados.csv"
CENTROIDS="$ROOT_DIR/../DadosEntrada/centroides_iniciais.csv"
ASSIGN_OUT="$ROOT_DIR/assign.csv"
CENTROIDS_OUT="$ROOT_DIR/centroids.csv"
GPU_LOG="$ROOT_DIR/gpu_power_log.csv"
GPU_CLEAN="$ROOT_DIR/gpu_power_clean.csv"
RUN_LOG="$ROOT_DIR/run_log.txt"
SUMMARY="$ROOT_DIR/run_summary.txt"
SAMPLE_INTERVAL=0.5
NVCC_ARCH="-arch=sm_89"

# ------------------- Safe summary writer -------------------
write_summary() {
  # first arg may be exit code
  local exit_code=${1:-${EXIT_CODE:-0}}
  N=${N:-0}
  K=${K:-0}
  it=${it:-0}
  elapsed_s=${elapsed_s:-0}
  elapsed_ms=${elapsed_ms:-0}
  flops_per_iter=${flops_per_iter:-0}
  total_flops=${total_flops:-0}
  total_mflop=${total_mflop:-0}
  mflops_per_s=${mflops_per_s:-0}
  samples=${samples:-0}
  avg=${avg:-0}
  max=${max:-0}
  energy_Wh=${energy_Wh:-0}

  # If GPU clean log exists, recompute samples/avg/max to ensure valid numbers
  if [ -f "$GPU_CLEAN" ]; then
    read samples avg max < <(LC_NUMERIC=C awk -F, 'NR>1{sum+=$2; if($2>max) max=$2; count++} END{if(count) printf "%d %.6f %.6f",count,sum/count,max; else printf "0 0 0"}' "$GPU_CLEAN") || true
  elif [ -f "$GPU_LOG" ]; then
    awk 'NR==1{print; next} {n=$0; split(n,a,","); gsub(/,/,".",a[2]); print a[1]","a[2]}' FS=, OFS=, "$GPU_LOG" > "$GPU_CLEAN" 2>/dev/null || true
    read samples avg max < <(LC_NUMERIC=C awk -F, 'NR>1{sum+=$2; if($2>max) max=$2; count++} END{if(count) printf "%d %.6f %.6f",count,sum/count,max; else printf "0 0 0"}' "$GPU_CLEAN") || true
  fi

  # Ensure numeric defaults (avoid empty strings)
  samples=${samples:-0}
  avg=${avg:-0}
  max=${max:-0}

  # Recompute elapsed times if missing
  elapsed_s=${elapsed_s:-0}
  elapsed_ms=${elapsed_ms:-0}

  # Recompute energy if possible
  if [ -z "$energy_Wh" ] || [ "$energy_Wh" = "0" ] || [ "$energy_Wh" = "0.000000" ]; then
    energy_Wh=$(LC_NUMERIC=C awk -v avg="$avg" -v s="$elapsed_s" 'BEGIN{if(s>0) printf "%.6f", avg * s / 3600; else print "0"}')
  fi

  # Compute MFLOPS per Watt and total MFLOP per Wh if possible
  mflops_per_w="NaN"
  total_mflop_per_Wh="NaN"
  if [ -n "$mflops_per_s" ] && [ -n "$avg" ] && awk "BEGIN{exit!(($avg+0)>0)}"; then
    mflops_per_w=$(LC_NUMERIC=C awk -v mps="$mflops_per_s" -v avg="$avg" 'BEGIN{if(avg>0) printf "%.6f", mps/avg; else print "NaN"}')
  fi
  if [ -n "$total_mflop" ] && [ -n "$energy_Wh" ] && awk "BEGIN{exit!(($energy_Wh+0)>0)}"; then
    total_mflop_per_Wh=$(LC_NUMERIC=C awk -v tm="$total_mflop" -v e="$energy_Wh" 'BEGIN{if(e>0) printf "%.3f", tm/e; else print "NaN"}')
  fi

  cat > "$SUMMARY" <<EOF
Run summary
===========
BIN: $BIN
DATA: $DATA
CENTROIDS: $CENTROIDS
N=$N
K=$K
iterations=$it
elapsed_s=$elapsed_s
elapsed_ms=$elapsed_ms

flops_per_iter=$flops_per_iter
total_flops=$total_flops
total_mflop=$total_mflop
mflops_per_s=$mflops_per_s

GPU_power_samples=$samples
GPU_avg_power_w=$avg
GPU_max_power_w=$max
estimated_energy_Wh=$energy_Wh

mflops_per_w=$mflops_per_w
total_mflop_per_Wh=$total_mflop_per_Wh

exit_code=$exit_code

Files produced:
- run log: $RUN_LOG
- gpu raw log: $GPU_LOG
- gpu clean log: $GPU_CLEAN
- summary: $SUMMARY
EOF

  # also print a short notice
  echo "run_summary written to: $SUMMARY"
}

# ensure we always attempt to write a summary on exit
trap 'status=$?; write_summary "$status"' EXIT

# ------------------- Compile -------------------
echo "Compiling $SRC -> $BIN"
nvcc -O2 $NVCC_ARCH "$SRC" -o "$BIN"

# ------------------- Start logger -------------------
echo "Starting GPU power logger (sampling every ${SAMPLE_INTERVAL}s) -> $GPU_LOG"
echo "timestamp,power_watts" > "$GPU_LOG"
(
  while true; do
    # timestamp in dd/mm/yyyy HH:MM:SS.mmm
    ts=$(date '+%d/%m/%Y %H:%M:%S.%3N')
    # normalize decimal separator from nvidia-smi (some locales use comma)
    val=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | tr ',' '.' | tr -d '\r') || val="0"
    echo "$ts,$val"
    sleep "$SAMPLE_INTERVAL"
  done
) >> "$GPU_LOG" 2>&1 &
LOGGER_PID=$!
echo "Logger started (pid=$LOGGER_PID)"
sleep 0.1

# ------------------- Run K-means -------------------
start_ts=$(date +%s.%3N)

echo "Running: $BIN $DATA $CENTROIDS 50 0.0001 $ASSIGN_OUT $CENTROIDS_OUT"
# capture stdout/stderr to run log
"$BIN" "$DATA" "$CENTROIDS" 50 0.0001 "$ASSIGN_OUT" "$CENTROIDS_OUT" 2>&1 | tee "$RUN_LOG"
EXIT_CODE=${PIPESTATUS[0]}

end_ts=$(date +%s.%3N)

# stop logger
kill "$LOGGER_PID" || true
wait "$LOGGER_PID" 2>/dev/null || true

# ------------------- Timings -------------------
elapsed_s=$(LC_NUMERIC=C awk "BEGIN {print ($end_ts - $start_ts)}")
elapsed_ms=$(LC_NUMERIC=C awk "BEGIN {print ($end_ts - $start_ts)*1000}")

# ------------------- Problem size (N,K) -------------------
# N: number of data points = lines in data file
# K: number of initial centroids = lines in centroid file
if [ -f "$DATA" ]; then
  N=$(wc -l < "$DATA" | tr -d '[:space:]')
else
  echo "Data file $DATA not found" >&2
  exit 2
fi
if [ -f "$CENTROIDS" ]; then
  K=$(wc -l < "$CENTROIDS" | tr -d '[:space:]')
else
  echo "Centroid file $CENTROIDS not found" >&2
  exit 2
fi

# ------------------- Iteration count -------------------
# Parse the run log for the line that contains the iteration count (supports Portuguese and English)
it=1
if grep -q 'Iterações:' "$RUN_LOG" 2>/dev/null; then
  it=$(sed -n 's/.*Iterações:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$RUN_LOG" | head -n1)
elif grep -q 'Iterations:' "$RUN_LOG" 2>/dev/null; then
  it=$(sed -n 's/.*Iterations:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$RUN_LOG" | head -n1)
else
  it=1
fi
# ensure numeric
if ! printf '%s' "$it" | grep -Eq '^[0-9]+$'; then
  it=1
fi

# ------------------- FLOP model -------------------
# Model (approximate) per iteration:
# - assignment: for each point, compute distance to each centroid: diff (1 flop) + multiply (1 flop) => 2 flops per point-centroid
# - after choosing nearest centroid: add point value to centroid sum (1 flop) + add squared distance to SSE (1 flop) => 2 flops per point
# - centroid update: one division per centroid => 1 flop per centroid
# So: flops_per_iter = 2*N*K + 2*N + K

flops_per_iter=$(LC_NUMERIC=C awk -v N="$N" -v K="$K" 'BEGIN{printf "%.0f", 2*N*K + 2*N + K}')
# total flops = it * flops_per_iter
total_flops=$(LC_NUMERIC=C awk -v it="$it" -v fpi="$flops_per_iter" 'BEGIN{printf "%.0f", it * fpi}')

# convert to MegaFlop (C locale)
total_mflop=$(LC_NUMERIC=C awk -v tf="$total_flops" 'BEGIN{printf "%.3f", tf/1e6}')
mflops_per_s=$(LC_NUMERIC=C awk -v tf="$total_flops" -v s="$elapsed_s" 'BEGIN{if(s>0) printf "%.3f", (tf/1e6)/s; else print "NaN"}')

# ------------------- GPU power stats -------------------
# normalize decimal and remove any stray stderr lines
awk 'NR==1{print; next} {n=$0; split(n,a,","); gsub(/,/ , ".", a[2]); print a[1]","a[2]}' FS=, OFS=, "$GPU_LOG" > "$GPU_CLEAN" || true
read samples avg max < <(LC_NUMERIC=C awk -F, 'NR>1{sum+=$2; if($2>max) max=$2; count++} END{if(count) printf "%d %.3f %.3f",count,sum/count,max; else printf "0 0 0"}' "$GPU_CLEAN")

# approximate energy in Wh = avg_power(W) * elapsed_seconds / 3600
energy_Wh=$(LC_NUMERIC=C awk -v avg="$avg" -v s="$elapsed_s" 'BEGIN{printf "%.6f", avg * s / 3600}')

# Finalize and write summary (uses write_summary which also recomputes missing stats)
write_summary "$EXIT_CODE"

# print brief summary
echo "--- Summary ---"
cat "$SUMMARY"

echo "\nFirst lines of cleaned GPU log:"
head -n 20 "$GPU_CLEAN" || true

echo "\nTo run again: edit variables at top of this script and run once."

# exit with same code as binary
exit $EXIT_CODE
