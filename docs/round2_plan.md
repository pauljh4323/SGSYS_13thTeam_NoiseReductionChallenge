# Round 2 (deep learning) plan

Plan only; no round-2 code lives on `main` until round 1 is closed. Round 2 starts 2026-11-18 KST; rules not yet announced.

## DL assets on branch dl-main (do not modify before round 2)

| item | value |
|---|---|
| lineage | gru_v1 -> v2b -> v3b -> v4b -> v5b (BiGRU magnitude mask, `gru_mask`) |
| best | gru_v5b_off |
| val PESQ | 2.6962 (0/5/10/15 dB: 1.953 / 2.410 / 2.702 / 3.083) |
| val_remix PESQ | 2.5170 |
| size | 2.37M parameters, 9.5 MB weights |
| runtime | CPU ~25-90 s for 164 validation files (torch CPU) |
| training | from scratch on the train split only (no pretrained models, no external data), Colab T4 |
| losses | compressed magnitude (cmag) + multi-resolution STFT + SI-SDR |
| augmentation | mixture gain augmentation |

How it was validated: validation split for selection, held-out val_remix gate, input-level probe (robustness to input
gain), and the end-to-end check (main.py outputs re-scored, local + fresh clone bit-identical, sr/channels/length kept).

## Status 2026-11-12: port done on branch r2-prep (tag r2-ready)

Done ahead of the round-1 close on a separate branch `r2-prep` (from main at dsp-v7, 65891f1; main and dl-main untouched):
- Inference-only port from dl-main (verbatim): `src/nrc/models/`, `src/nrc/methods/gru_mask.py` (registered),
  `configs/dl/gru_v5b_off.yaml`, `weights/gru_v5b.pt` (blob 87799003). Training code stays on dl-main.
- `configs/submission.yaml` = gru_v5b_off; `requirements.txt` + `torch==2.14.1` (CPU extra index).
- `scripts/e2e_check.py`: `ROUND` setting (`--round` override): 1 = classical allow-list + torch-free probe, 2 = classical + gru_mask.
- Code commit 32a68f8: selfcheck OK; e2e PASSED with ROUND 2 (local, fresh clone, other cwd, fresh pip venv with torch CPU:
  pip install ~193 s, ~801 MB; all bit-identical). main.py val PESQ 2.6962 / SNR 16.81 dB, val_remix 2.5170 / 15.45 dB
  (identical to dl-main); main.py 164 files 18.5-32.1 s CPU.
- README section "2차 챌린지 (준비 중)" on r2-prep.

Still depends on the round-2 notice: every open question below; whether r2-prep is submitted as is or rebased as
`r2-main` from `round1-final` (branch plan step 2); main's e2e_check stays classical-only (no ROUND setting) until round 1 closes;
and whether README / submissions.md need a round-2 submission row (only after the team confirms a submission).

## Open questions (confirm from the round-2 notice)

- Are pretrained models allowed?
- Is external data allowed?
- Limits on weight file size and on inference runtime?
- Is a GPU available at evaluation, or CPU only?
- Constraints on requirements.txt (torch allowed, version pins, install size)?

## Branch plan

1. After 2026-11-17 23:59 KST: tag `main` as `round1-final`.
2. Create branch `r2-main` from `round1-final`.
3. Port from `dl-main` onto `r2-main`'s current infrastructure: the `gru_mask` model/method code and the gru_v5b weights,
   wired into the requirements split, e2e/compliance checks, report_val and plot_spectrograms.
4. requirements.txt adds pinned torch (CPU build); dev-only tools stay in requirements-dev.txt.
5. The CLASSICAL_METHODS check becomes an allow-list of round-2 methods.
6. First round-2 submission = gru_v5b, as early as possible (after e2e PASS incl. fresh clone + fresh venv).

## Candidate improvements (decide after the first round-2 submission)

- Model soup (weight averaging) across the v3b -> v5b lineage.
- Complex / phase-aware masking instead of magnitude-only.
- Larger model within the CPU runtime budget.
- Hybrid with the classical front-end (e.g. dsp-v4 high-pass / noise estimate as input features or pre-stage).
