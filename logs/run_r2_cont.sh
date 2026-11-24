#!/bin/bash
# exp/r2-dmix: gru_v8p2z3c / gru_v8p2z3d (rms_norm): validation, val_remix, -30/-20/-10/0 dB probe.
# --jobs 1, --resume; re-run to continue after an interruption.
set -u
PY="conda run --no-capture-output -n nrc python"
rm -f logs/run_r2_cont.done
run() { local m=$1 t=$2; shift 2
  if [ -f outputs/$m/eval_$t.csv ] && grep -q "^ok rc=0" logs/eval_${m}_$t.log 2>/dev/null; then echo "skip $m $t"; return; fi
  echo "start $m $t $(date +%T)"
  $PY scripts/eval.py --config configs/dl/$m.yaml "$@" --jobs 1 --resume >> logs/eval_${m}_$t.log 2>&1; echo "ok rc=$?" >> logs/eval_${m}_$t.log
}
for m in gru_v8p2z3c gru_v8p2z3d; do
  run $m validation --split validation
  run $m val_remix --split val_remix
  for g in -30 -20 -10 0; do run $m validation_gain${g}dB --split validation --input-gain-db $g; done
done
echo DONE > logs/run_r2_cont.done
