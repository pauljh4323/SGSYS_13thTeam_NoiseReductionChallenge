#!/usr/bin/env bash
# Round-2 ensemble batch (exp/r2-ens): sequential, --jobs 1, resumable (finished runs skipped; partial per-file CSVs resumed).
set -u
cd "$(dirname "$0")/.."
PY="conda run --no-capture-output -n nrc python"
for c in ens_e1 ens_e1g ens_e2; do
  for s in validation val_remix; do
    [ -f "outputs/$c/eval_$s.csv" ] && { echo "skip $c $s"; continue; }
    echo "$(date +%T) eval $c $s"
    $PY scripts/eval.py --config configs/dl/$c.yaml --split $s --jobs 1 --resume > logs/eval_${c}_$s.log 2>&1 || echo "FAILED $c $s"
  done
  for g in -30 -20 -10 0; do
    [ -f "outputs/$c/eval_validation_gain${g}dB.csv" ] && { echo "skip $c gain $g"; continue; }
    echo "$(date +%T) probe $c $g dB"
    $PY scripts/eval.py --config configs/dl/$c.yaml --split validation --input-gain-db $g --jobs 1 --resume > logs/eval_${c}_validation_gain${g}dB.log 2>&1 || echo "FAILED $c gain $g"
  done
done
for c in gru_v8p2x_rmsn ens_e1 ens_e1g ens_e2; do
  [ -f "logs/time_r2ens_$c.log" ] && { echo "skip time $c"; continue; }
  echo "$(date +%T) time $c"
  rm -rf "outputs/time_r2ens_$c"
  $PY main.py --input data/noise_reduction_challenge/validation/noisy --output "outputs/time_r2ens_$c" --config configs/dl/$c.yaml 2>&1 | tail -1 > "logs/time_r2ens_$c.log"
done
echo "$(date +%T) ALL DONE"
