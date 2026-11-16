# CLAUDE.md — Speech Noise Reduction Challenge 1 (traditional DSP)
## Objective
Maximize PESQ-WB (pesq==0.0.4, 16 kHz) on the hidden test; auxiliary metric SNR. Deadline 2026-11-17 23:59 KST.
Current submission: dsp_k / val PESQ 1.9884, SNR 12.84 dB / tag dsp-v7, submitted 2026-11-16 (tag sub-20261116); supersedes dsp-v6 1.9637, dsp-v5 1.9570, dsp-v4 1.9497, dsp-v3 1.9352, dsp-v2 1.8746, dsp-v1 1.7505.
## Working policy
- Do not plan around submission limits or deadlines; maximize PESQ with maximum effort.
- Compute efficiency and hardware are secondary: prefer larger grids, full-train tuning (all 1296 files) and more iterations.
- On memory pressure lower --jobs and rerun instead of shrinking the experiment. Run long jobs in the foreground.
## Rounds
- Round 1 (classical DSP only, no ML/DL) ends 2026-11-17 23:59 KST; round 2 (DL) starts 2026-11-18 KST.
- Round 2 is deep-learning speech enhancement (decided; TA notice pending). Ready state = tag r2-ready on branch r2-prep.
- main stays classical until tag round1-final; DL work stays on dl-main until then. Round-2 plan: docs/round2_plan.md.
- Leaderboard: https://docs.google.com/spreadsheets/d/1Jebd63s24CKjO6MOe14fef9j_coWdc87011rAgsibtI/edit?gid=1596638433#gid=1596638433
- Standings live only in docs/leaderboard.md (newest snapshot first); never copy other teams' standings elsewhere.
## Hard rules
- No ML/DL in the submitted algorithm. Grid-searched parameters and rule-based input-adaptive parameters are allowed.
- Clean audio for evaluation only. Output keeps sample rate, channel count, sample count, filename. No absolute paths.
- requirements.txt = pinned, pip-installable, main.py deps only. Dev tools in requirements-dev.txt.
- main = classical submission only; DL work lives on branch dl-main (never merge it into main).
- README.md must be written in Korean (commands, paths, identifiers, numbers unchanged).
## Environment
- Python via `conda run -n nrc python ...`. Conda at D:/miniconda3. Repo path has spaces: quote paths.
- Data root data/noise_reduction_challenge (--data-root / $NRC_DATA_ROOT). Secondary held-out set: data/val_remix (make_val_remix.py).
## Commands
- Submission run: `python main.py` (input/ -> output/)
- Evaluate: `python scripts/eval.py --config configs/<name>.yaml --split validation|val_remix|train [--subset N]`
- Sweep: `python scripts/sweep.py --grid configs/sweeps/<name>.yaml`
- Checks: scripts/selfcheck.py, scripts/e2e_check.py; reports: scripts/report_val.py, scripts/plot_spectrograms.py
## Rules
- Tune on train (all 1296 files preferred), confirm on validation, gate on val_remix. Never select on val_remix.
- Only scripts/eval.py produces metrics; append to results/experiments.csv with a clean hash (commit code before final evals).
- Report PESQ and SNR together (overall, per SNR, per difficulty).
- Promotion rule (round 1): full-train paired delta >= 0.003 AND z >= 3, val PESQ >= current, val_remix PESQ >= current, each probe level >= current - 0.02. No runtime gate; report main.py time.
- Branch exp/<name>; merge to main only when acceptance criteria pass; tags dsp-vN (new best), sub-YYYYMMDD (only after the team confirms it was actually submitted).
- After each task append one row to docs/ai_usage_log.md (date | goal | key prompt (summary) | what AI did | verification | references).
- Published methods: add each to docs/references.md (title, authors, year, DOI/link) with a one-line description of what was used; pending until the team verifies it.
- Long outputs -> logs/. Run heavy evals one at a time. No new deps without stating why.
