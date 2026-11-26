# CLAUDE.md — Speech Noise Reduction Challenge 1 (traditional DSP)
## Objective
Maximize PESQ-WB (pesq==0.0.4, 16 kHz) on the hidden test; auxiliary metric SNR. Round 2 ends 2026-12-13 KST.
Current round-2 candidate (r2-prep): gru_v8p2z3e + rms_norm / val PESQ 3.6534, SNR 1.21 dB / tag r2-ready (round 1: dsp_k 1.9884 / 12.84 dB, dsp-v7, sub-20261116).
## Rounds
- Round 1 (classical DSP only, no ML/DL) ends 2026-11-17 23:59 KST; round 2 (DL) starts 2026-11-18 KST.
- main stays classical until tag round1-final; DL work stays on dl-main until then. Round-2 plan: docs/round2_plan.md.
- Round 2 (DL allowed) 2026-11-18 to 2026-12-13 KST. Final report due 2026-12-14 (A4 ~10 pages, Malgun/Nanum Gothic 11 pt; outline docs/final_report/outline.md).
- Hidden test: 483 files, 16 kHz mono, 48000 samples, includes noise types not in train/validation. Rank = mean PESQ-WB; SNR auxiliary.
- TA env Python 3.12, torch 2.15.0+cpu, pesq 0.0.4; weights load with torch.load(..., weights_only=True); no downloads at inference; weights < 50 MB plain commit (LFS only if large).
- Output: flat output/ with exactly the input WAV names (same sr, channels, sample count), nothing else; scripts/e2e_check.py prints the 8-item notice checklist.
- README (Korean) states Python version, packages, install + inference commands, model/weight paths, main training settings, why these weights, validation PESQ/SNR, team (13조 잡음처리반), round, commit/tag.
- External data, public code, pretrained weights: allowed only if logged in docs/external_data.md (source, version, licence, split/fold, train vs validation use).
- First submission does not count toward the weekly limit; the best verified submission counts; missing/extra files or format/PESQ errors withhold the score.
- Round-2 promotion: primary = pooled paired PESQ delta vs current candidate on G_unseen_test + G_multi_test + G_low > 0, z >= 2; secondary = val_remix delta >= -0.02, validation delta >= -0.03, level probe flat, full e2e PASS.
- Standings live only in docs/leaderboard.md (newest snapshot first); never copy leaderboard numbers elsewhere.
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
- Tune on train (or a train subset), confirm on validation, gate on val_remix. Never select on val_remix.
- Only scripts/eval.py produces metrics; append to results/experiments.csv with a clean hash (commit code before final evals).
- Report PESQ and SNR together (overall, per SNR, per difficulty).
- Round-1 promotion rule (revised 2026-11-11, docs/exp_dsp_g.md): full-train paired delta >= 0.003 and z >= 3, val PESQ and val_remix PESQ >= current, every probe level >= current - 0.02, main.py CPU < 180 s.
- Branch exp/<name>; merge to main only when acceptance criteria pass; tags dsp-vN (new best), sub-YYYYMMDD (only after the team confirms it was actually submitted).
- After each task append one row to docs/ai_usage_log.md (date | goal | key prompt (summary) | what AI did | verification | references).
- Add every algorithm source to docs/references.md (title, authors, year, DOI/link) marked pending until the team verifies it.
- Long outputs -> logs/. Run heavy evals one at a time. No new deps without stating why.
