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
