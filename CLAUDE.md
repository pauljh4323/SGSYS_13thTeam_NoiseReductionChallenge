# CLAUDE.md — Speech Noise Reduction Challenge
## Objective
Maximize wideband PESQ (pesq==0.0.4, fs=16000, 'wb'). Current best: course Wiener baseline / val PESQ 1.598 (not yet reproduced).
## Environment
- `conda env create -f environment.yml`; run Python via `conda run -n nrc python ...` (or `conda activate nrc` before launching claude).
- Conda at D:/miniconda3. Repo path contains spaces: always quote paths.
- Data root: data/noise_reduction_challenge (override with --data-root or $NRC_DATA_ROOT).
## Commands
- Submission run: `python main.py` (input/ -> output/, same filenames)
- Enhance split: `python scripts/enhance.py --config configs/<name>.yaml --split validation`
- Evaluate: `python scripts/eval.py --config configs/<name>.yaml --split validation [--subset 32]`
## Rules
- Only scripts/eval.py produces metrics; append to results/experiments.csv.
- Output wav: same length/fs/filename as input, 16-bit PCM, no clipping. Inference CPU-only, offline.
- Clean audio is never used at inference.
- Never commit data/, outputs/, logs/, reference/, input/*.wav, output/*.wav. Weights in weights/ (<50 MB each, no LFS).
- Branch exp/<name>; merge to main only on val PESQ gain; tags best-vN, sub-YYYYMMDD.
- Commit style feat/fix/exp/refactor/docs/chore; val PESQ in exp: messages.
- After each task append one row to docs/ai_usage_log.md.
- Long outputs -> logs/, show summaries only. No new deps without stating why; keep environment.yml and requirements.txt in sync.
