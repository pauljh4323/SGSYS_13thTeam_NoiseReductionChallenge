# CLAUDE.md — Speech Noise Reduction Challenge 1 (traditional DSP)
## Objective
Maximize PESQ-WB (pesq==0.0.4, 16 kHz) on the hidden test; auxiliary metric SNR. Deadline 2026-11-17 23:59 KST.
Current submission candidate: mmse_lsa_top1 / val PESQ 1.7505 / tag dsp-v1 (first submission, not yet sent).
## Hard rules
- No ML/DL in the submitted algorithm. Grid-searched parameters and rule-based input-adaptive parameters are allowed.
- Clean audio for evaluation only. Output keeps sample rate, channel count, sample count, filename. No absolute paths.
- requirements.txt = pinned, pip-installable, main.py deps only. Dev tools in requirements-dev.txt.
- main = classical submission only; DL work lives on branch dl-main (never merge it into main).
## Environment
- Python via `conda run -n nrc python ...`. Conda at D:/miniconda3. Repo path has spaces: quote paths.
- Data root data/noise_reduction_challenge (--data-root / $NRC_DATA_ROOT). Secondary held-out set: data/val_remix (make_val_remix.py).
## Commands
- Submission run: `python main.py` (input/ -> output/)
- Evaluate: `python scripts/eval.py --config configs/<name>.yaml --split validation|val_remix|train [--subset N]`
- Sweep: `python scripts/sweep.py --grid configs/sweeps/<name>.yaml`
- Checks: scripts/selfcheck.py, scripts/e2e_check.py; reports: scripts/report_val.py, scripts/plot_spectrograms.py
## Rules
- Tune on train (or a train subset), confirm on validation, gate on val_remix. Never select on val_remix.
- Only scripts/eval.py produces metrics; append to results/experiments.csv with a clean hash (commit code before final evals).
- Report PESQ and SNR together (overall, per SNR, per difficulty).
- Branch exp/<name>; merge to main only when acceptance criteria pass; tags dsp-vN (new best), sub-YYYYMMDD (only after the team confirms it was actually submitted).
- After each task append one row to docs/ai_usage_log.md (date | goal | key prompt (summary) | what AI did | verification | references).
- Add every algorithm source to docs/references.md (title, authors, year, DOI/link) marked pending until the team verifies it.
- Long outputs -> logs/. Run heavy evals one at a time. No new deps without stating why.
