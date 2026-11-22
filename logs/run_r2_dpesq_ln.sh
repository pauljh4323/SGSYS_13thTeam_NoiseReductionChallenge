#!/bin/bash
# exp/r2-dpesq: gru_v8p2 level norm ON (gru_v8p2_lnrb) - validation, val_remix, -30/-20/-10/0 dB probe; --jobs ${JOBS:-1}; resumable.
set -u
PY="conda run --no-capture-output -n nrc python"
rm -f logs/run_r2_dpesq_ln.done
run() { local m=$1 t=$2; shift 2
  if [ -f outputs/$m/eval_$t.csv ] && grep -q "ok" logs/eval_${m}_$t.log 2>/dev/null; then echo "skip $m $t"; return; fi
  echo "start $m $t $(date +%T)"
  $PY scripts/eval.py --config configs/dl/$m.yaml "$@" --jobs ${JOBS:-1} > logs/eval_${m}_$t.log 2>&1; echo "ok rc=$?" >> logs/eval_${m}_$t.log
}
m=gru_v8p2_lnrb
run $m validation --split validation
run $m val_remix --split val_remix
for g in -30 -20 -10 0; do run $m validation_gain${g}dB --split validation --input-gain-db $g; done
echo DONE > logs/run_r2_dpesq_ln.done
