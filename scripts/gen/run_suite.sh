#!/usr/bin/env bash
# Generalisation-suite driver (exp/r2-gen): one heavy process at a time, eval.py --jobs 1 --resume.
# Usage: bash scripts/gen/run_suite.sh <model>... ; models: dsp_v8 gru_v5b_off gru_v8p2z3d gru_v8p2z3e gru_v8p2z4w
#        ext_dfn3 ext_mpsenet ext_cmgan. A finished (model, split) leaves logs/gen_suite/<model>_<split>.done and
#        is skipped on a rerun; an interrupted one resumes from eval.py's per-file partial CSV / existing outputs.
set -u
cd "$(dirname "$0")/../.."
SPLITS=${SPLITS:-"validation gen_unseen_test gen_multi_test gen_low"}
C="D:/miniconda3/Scripts/conda.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1  # conda run re-prints child output in cp949 otherwise (crashed on DFN logs)
mkdir -p logs/gen_suite
for m in "$@"; do
  for s in $SPLITS; do
    done_f=logs/gen_suite/${m}_${s}.done
    [ -f "$done_f" ] && { echo "skip $m $s"; continue; }
    log=logs/gen_suite/${m}_${s}.log
    echo "=== $(date +%T) $m $s"
    case $m in
      dsp_v8)
        $C run --no-capture-output -n nrc python scripts/gen/run_dsp_v8.py --split "$s" > "$log" 2>&1 &&
        $C run --no-capture-output -n nrc python scripts/eval.py --config configs/gen/dsp_v8_pred.yaml --split "$s" --jobs 1 --resume \
          --pred-dir "outputs/dsp_v8_pred/$s" --notes "exp/r2-gen dsp-v8 (tag code via worktree)" >> "$log" 2>&1 ;;
      ext_*)
        mm=${m#ext_}
        env=$([ "$mm" = dfn3 ] && echo ext_dfn || echo "ext_$mm")
        $C run --no-capture-output -n "$env" python scripts/gen/ext_infer.py --model "$mm" --split "$s" > "$log" 2>&1 &&
        $C run --no-capture-output -n nrc python scripts/eval.py --config "configs/gen/$m.yaml" --split "$s" --jobs 1 --resume \
          --pred-dir "outputs/$m/$s" --notes "exp/r2-gen pretrained zero-shot" >> "$log" 2>&1 ;;
      *)
        $C run --no-capture-output -n nrc python scripts/eval.py --config "configs/dl/$m.yaml" --split "$s" --jobs 1 --resume \
          --notes "exp/r2-gen" > "$log" 2>&1 ;;
    esac
    if [ $? -eq 0 ]; then touch "$done_f"; grep -m1 -A3 "\[overall\]" "$log" | tail -1; else echo "FAILED $m $s (see $log)"; fi
  done
done
echo SUITE_DONE "$@"
