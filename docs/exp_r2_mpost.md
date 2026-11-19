# exp/r2-mpost: SNR-adaptive mask strength for gru_v5b (round 2): FAIL, r2-prep unchanged

Date 2026-11-19. Branch `exp/r2-mpost`, branched from r2-prep at 7193fc7 (tag r2-ready). No training. Code and config at
clean hash 4df01aa. Motivation: in docs/r2_noise_analysis.md (exp/r2-noise-analysis), distortion dominated at 0–5 dB and
residual noise at 10–15 dB. So the idea was to weaken the mask at low estimated SNR and sharpen it at high estimated SNR.

## Method (`mask_post`, src/nrc/methods/gru_mask.py)

Optional block of the gru_mask method, usable by any gru_mask model. It is rule-based and uses the noisy input only:
- SNR_est = 10·log10((Σ|M X|² + ε) / (Σ|(1−M) X|² + ε)), where M is the model's own mask and X the noisy STFT. Per file, or per
  frame with the numerator and denominator smoothed by a centred moving average of 0.25 s (31 frames).
- γ = γ_lo for SNR_est ≤ a, γ_hi for SNR_est ≥ b, linear in between. M' = max(M^γ, floor). Output = iSTFT(M' X).
- Optional 100 Hz zero-phase high-pass, the same one the dsp chain uses (mmse_lsa `_highpass`).
- Off by default. **Identity check (20 validation files): mask_post off vs r2-prep code max|diff| = 0; identity block
  (γ 1, floor 0, hpf off) vs off max|diff| = 0; 16-bit PESQ vs the recorded gru_v5b_off rows max|diff| = 0.**
- `scripts/eval.py`: the method-cache key is now `repr(sorted(params))`, because a dict parameter is unhashable. Behaviour is
  otherwise unchanged.

Tuning tool: `scripts/r2_mpost.py` (stages identity / cache / grid / frame / report). It caches M and X per file once, then
scores every grid point on the cached tensors with the method's own functions, at 16-bit PESQ as main.py writes it. The grid
stage is resumable: one block of per-file rows is flushed per finished point, and complete points are skipped on restart.
Workers load one cached file at a time, and peak RSS is logged.

## Grid (400-file stratified train subset, 360 points)

γ_lo {0.6, 0.7, 0.8, 0.9, 1.0} × γ_hi {1.0, 1.1, 1.25, 1.5} × (a, b) {(0, 10), (3, 12), (5, 15)} × floor {0, 0.03, 0.1} ×
hpf {off, 100 Hz}. Identity (v5b) on this subset: PESQ 3.1497, SNR 17.71 dB. Full table:
`results/r2_mpost_grid_gru_v5b_off_train400.csv`.

| # | γ_lo | γ_hi | a, b | floor | hpf | PESQ | SNR dB | Δ vs identity | SE | z |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0 | 1.25 | 0, 10 | 0 | off | 3.1834 | 17.68 | +0.0336 | 0.0031 | 10.89 |
| 2 | 1.0 | 1.25 | 3, 12 | 0 | off | 3.1822 | 17.68 | +0.0325 | 0.0027 | 11.88 |
| 3 | 1.0 | 1.1 | 0, 10 | 0 | off | 3.1796 | 17.71 | +0.0298 | 0.0024 | 12.42 |
| 4 | 1.0 | 1.25 | 5, 15 | 0 | off | 3.1782 | 17.69 | +0.0285 | 0.0024 | 11.64 |
| 5 | 1.0 | 1.1 | 3, 12 | 0 | off | 3.1765 | 17.71 | +0.0267 | 0.0023 | 11.52 |

- Every γ_lo < 1 lowers PESQ: the best by γ_lo is 0.6 → 3.1350, 0.8 → 3.1661, 1.0 → 3.1834. **Weakening the mask at low SNR does not
  reduce the distortion error; the 0–5 dB "distortion" share is not a mask-strength problem.**
- A floor hurts a lot (best 3.0184 at 0.03, 2.6400 at 0.1). The 100 Hz high-pass hurts (best 3.0988). γ_hi 1.25 > 1.1 > 1.5.
- The best point gains at 5/10/15 dB (+0.027/+0.037/+0.043), not at 0 dB (+0.001).
- Frame-level variant of the best point (0.25 s smoothing): 3.1669, +0.0172 vs identity (z 7.23) and **−0.0164 vs file-level
  (SE 0.0020, z −8.42)**. File-level kept.
- **Train-subset vs validation gap of v5b: 3.1497 vs 2.6962 = +0.4535** (full train 3.1537, +0.4575). v5b was trained on
  train, so the mask is more accurate there than on held-out data.

## Confirmation of the best point (γ 1.0 → 1.25 over SNR_est 0 → 10 dB, file-level; configs/dl/gru_v5b_mpost.yaml)

| set | n | v5b | mask_post | Δ (SE, z) | SNR dB v5b → cand |
|---|---|---|---|---|---|
| full train (keep rule Δ ≥ 0.003, z ≥ 3) | 1296 | 3.1537 | 3.1887 | +0.0350 (0.0018, +18.95) | 17.64 → 17.60 |
| validation | 164 | 2.6962 | **2.6924** | −0.0037 (0.0062, −0.60) | 16.81 → 16.68 |
| val_remix | 328 | 2.5170 | **2.5125** | −0.0046 (0.0042, −1.09) | 15.45 → 15.33 |

Per SNR (Δ, z). Validation: 0 dB −0.0000 (−0.10), 5 dB −0.0013 (−0.18), 10 dB +0.0039 (+0.33), 15 dB −0.0138 (−1.06).
val_remix: 0 dB +0.0002 (+2.15), 5 dB +0.0064 (+1.32), 10 dB −0.0061 (−0.55), 15 dB −0.0189 (−1.65).
Full train: 0 dB +0.0007, 5 dB +0.0369, 10 dB +0.0404, 15 dB +0.0375 (all z ≥ 2.7).

Top-5 headroom types (Δ, z; validation | val_remix): machine −0.0153 (−1.30) | −0.0057 (−0.77); wind −0.0170 (−1.51) |
−0.0130 (−1.38); aircraft_cockpit +0.0070 (+0.23) | −0.0135 (−0.82); broadband_noise −0.0044 (−0.15) | −0.0231 (−1.15);
alarm_siren +0.0249 (+2.33) | **+0.0423 (+4.23)**. Only the tonal, residual-dominant type (alarm_siren) gains on held-out data.

Level probe (validation PESQ at input gain −30/−20/−10/0 dB): v5b 2.6137/2.6700/2.6843/2.6962, mask_post
2.6036/2.6660/2.6818/2.6924. All within 0.02 of v5b.

## Round-2 decision: FAIL

val PESQ 2.6924 < 2.6962 (fail); val_remix Δ −0.0046, z −1.09 (fail); probe within −0.02 (pass). r2-prep, configs/submission.yaml,
README and tag r2-ready are unchanged. The rows are in results/experiments.csv (4df01aa, validation and val_remix).

**Why the train gain does not transfer:** v5b's mask is much more accurate on its own training data (+0.45 PESQ gap).
Sharpening an accurate mask (M^1.25) removes more residual noise at little cost to speech. On held-out data the mask is less
accurate, so sharpening it removes speech as well. Tuning any post-process of a trained model's mask on its training split is
biased this way. Future post-process grids for gru_mask models should tune on a split the model has not seen, but CLAUDE.md
forbids selecting on val_remix, and validation is the confirmation set. Possible options are a train fold held out from
training (v6m*), or using the alarm_siren gain only through a type-specific rule (tonal noise, low per-frame flatness),
which would need its own held-out evidence.

## Runs and resources

The first grid run (`--jobs 4`, all rows held until the end) was killed for low system memory before writing anything. The
resumable grid (`--jobs 2`) then ran to completion with no restarts in 85 min. Peak RSS per worker was 457 / 458 MB and the
main process 164 MB. The full-train cache stage (896 new files) was killed once for low system memory at 1162/1296 files, and
the rerun of the same command resumed from there (restart 1; workers 310 MB). The full-train confirmation used 311–315 MB per
worker. Logs: `logs/r2_mpost_*` (git-ignored).
