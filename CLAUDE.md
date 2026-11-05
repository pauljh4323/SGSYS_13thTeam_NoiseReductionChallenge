# CLAUDE.md — Speech Noise Reduction Challenge
## Objective
Maximize wideband PESQ (pesq==0.0.4, fs=16000, 'wb'). Current best / submission: gru_v1_lnrb (configs/dl/gru_v1_lnrb.yaml params = configs/submission.yaml, weights/gru_v1.pt, range_boost input level guard) / val PESQ 2.3598 / best-v4 (fallback: mmse_lsa_top1 1.7505, best-v2) (last submitted: Wiener 1.5987, sub-20261105).
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
- Commit code before the final eval run so results rows record a clean git hash.
- Output wav: same length/fs/filename as input, 16-bit PCM, no clipping. Inference CPU-only, offline.
- Clean audio is never used at inference.
- Never commit data/, outputs/, logs/, reference/, input/*.wav, output/*.wav. Weights in weights/ (<50 MB each, no LFS).
- Branch exp/<name>; merge to main only on val PESQ gain; tags best-vN, sub-YYYYMMDD.
- Commit style feat/fix/exp/refactor/docs/chore; val PESQ in exp: messages.
- After each task append one row to docs/ai_usage_log.md.
- Long outputs -> logs/, show summaries only. No new deps without stating why; keep environment.yml and requirements.txt in sync.
