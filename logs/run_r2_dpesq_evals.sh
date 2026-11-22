#!/bin/bash
# exp/r2-dpesq eval batch: run from the repo root; one heavy job at a time, --jobs ${JOBS:-1}; finished runs are skipped (resumable).
set -u
PY="conda run --no-capture-output -n nrc python"
rm -f logs/run_r2_dpesq_evals.done
run() { # $1 model, $2 csv tag, rest = eval args
  local m=$1 t=$2; shift 2
  if [ -f outputs/$m/eval_$t.csv ] && grep -q "ok" logs/eval_${m}_$t.log 2>/dev/null; then echo "skip $m $t"; return; fi
  $PY scripts/eval.py --config configs/dl/$m.yaml "$@" --jobs ${JOBS:-1} > logs/eval_${m}_$t.log 2>&1; echo "ok rc=$?" >> logs/eval_${m}_$t.log
}
for m in gru_v8p1 gru_v8p2 gru_v8p3; do
  run $m validation --split validation
  run $m val_remix --split val_remix
  for g in -30 -20 -10 0; do run $m validation_gain${g}dB --split validation --input-gain-db $g; done
done
for m in gru_v5b_off gru_v8p1 gru_v8p2 gru_v8p3; do
  [ -f logs/time_r2dpesq_$m.log ] && grep -q processed=164 logs/time_r2dpesq_$m.log && continue
  rm -rf outputs/time_r2dpesq_$m
  $PY main.py --input data/noise_reduction_challenge/validation/noisy --output outputs/time_r2dpesq_$m --config configs/dl/$m.yaml > logs/time_r2dpesq_$m.log 2>&1
done
echo DONE > logs/run_r2_dpesq_evals.done
