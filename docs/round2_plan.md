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

## Status 2026-11-23: r2-prep submission -> gru_v8p2x + level mode rms_norm (tag r2-ready moved)

- gru_v8p2x continues gru_v8p2 with the same loss. It starts from the v8p2 weights with a fresh Adam 5e-5 and runs
  200 epochs (best 194). Branch exp/r2-dpesq.
- Level mode rms_norm (new in src/nrc/methods/gru_mask.py):
  - g = target_rms / max(rms(noisy), 1e-6), over the whole file. The model runs on g * noisy.
  - The output stays at that level (no inverse gain). If its peak is above 0.99, the whole output is scaled to 0.99.
  - target_rms 0.079130 (-22.03 dBFS) is the train noisy median (scripts/rms_target.py).
  - With level norm off, the output is bit-identical to before.
- eval.py at c9a25c7: val PESQ 3.4867 / SNR 2.24 dB, val_remix 3.3453 / 2.09 dB.
  - Paired d vs gru_v8p2: val +0.0884 (z 8.7), val_remix +0.1075 (z 11.9).
  - Probe -30/-20/-10/0 dB: 3.487 each. The input is level-invariant; v8p2x OFF scores 2.640 at -30 dB.
  - main.py outputs at 4b6ca11 give the same numbers. e2e passes for round 2 (fresh torch 2.15.0 venv, bit-identical).
- Round-2 rule over 9 model x level pairs (v8p2/v8p3/v8p2x x OFF/lnrb/rms_norm): v8p3 lnrb, v8p3 rms_norm and v8p2x
  rms_norm pass. v8p2x rms_norm has the highest val PESQ. Details are in docs/exp_r2_dpesq.md (exp/r2-dpesq).
- The promotion brings weights/gru_v8p2x.pt, configs/dl/gru_v8p2x{,_rmsn}.yaml, the rms_norm code and scripts/rms_target.py.
- Fallbacks: gru_v8p2 (configs/dl/gru_v8p2.yaml) and gru_v5b (configs/dl/gru_v5b_off.yaml).
- Still training on Colab: gru_v8p2y (continues v8p2x with input gain U[-40, +6] dB on every sample). Promote it only if it
  passes the same rule and beats gru_v8p2x + rms_norm.

## Status 2026-11-18: r2_diag conclusions and gru_v6 prep (branch exp/r2-v6, from exp/r2-diag)

r2_diag (docs/r2_diag.md): every validation noise_code also occurs in train (validation is a seen-noise test); gru_v5b is
weakest on stationary NOISEX noises (aircraft_cockpit, factory, broadband_noise, destroyer_operations, hf_channel) and at
0 dB; per-category routing / blending with dsp-v8 is bounded at about +0.02 PESQ and is about 0 with a realistic noise
classifier, so it is not pursued. Improvements go into the model.

Phase headroom (oracle: gru_v5b enhanced magnitude + clean STFT phase, 16-bit scoring, scripts/r2_phase_oracle.py,
logs/r2_phase_oracle.md; analysis bound, not a method):

| split | gru_v5b | oracle phase | delta | 0 dB delta | 15 dB delta | weak-5 delta |
|---|---|---|---|---|---|---|
| validation | 2.6962 | 2.9893 | +0.293 | +0.159 | +0.312 | +0.346 (n 38) |
| val_remix | 2.5170 | 2.7776 | +0.261 | +0.162 | +0.308 | +0.328 (n 76) |

Phase is worth about +0.26 to +0.29 at most: largest on the weak stationary categories, smallest at 0 dB (there the
magnitude estimate is still the limit). SNR gains about +2.5 dB.

Two separately testable fine-tunes from gru_v5b (both start output-identical to gru_v5b; configs/dl/gru_v6{a,b}.yaml):
- **gru_v6a, complex ratio mask** (`model.mask_type: crm`): the sigmoid head is the real part M_r, plus an imaginary
  head M_i = mask_imag_scale * tanh(Linear(h)) with zero weights/bias; enhanced = (M_r + j M_i) * Y. mask_imag_scale 0.1:
  with 1.0 the zero-init head drifted under Adam (val8 2.717 -> 2.621 after 10 steps, 2.573 after 40), with 0.1 it holds
  (2.703 / 2.694). Compressed-complex loss term implemented (`lambda_ccplx`) but off this round.
- **gru_v6b, noise-aware inputs** (`model.noise_aware: true`, src/nrc/features.py): 2 x 257 features per model frame,
  normalised with train-set mean/std buffers, added through a zero-initialised Linear(514 -> H) into the input projection.
  - log noise PSD: dsp-v8 stationary estimate (spp_mean, noise_fraction 0.1, 3 iterations, xi_H1 10 dB), before the
    noise bias.
  - log xi: first-step decision-directed a priori SNR of the dsp-v7 path (forward DD, alpha 0.94, xi_min -25 dB, LSA gain,
    gmin -30 dB, TSNR gain in the recursion, noise x two-band bias 1.25/1.5), no NLM.
  - **Grid mapping.** Both are estimated on dsp-v8's own grid: n_fft 1024, hop 256, 513 bins at k x 15.625 Hz, frame j
    centred at sample j x 256 (scipy, zero boundary). The model grid is n_fft 512, hop 128, 257 bins at f x 31.25 Hz,
    frame m centred at sample m x 128 (torch center=True).
    - Frequency: the LOG values are linearly interpolated over frequency in Hz onto the 257 model bins. The model bin
      f falls on dsp bin 2f, so this is an exact resampling of the log spectrum.
    - Time: the log noise PSD is stationary and is broadcast to every model frame. Log xi is linearly interpolated in
      the log domain between dsp frame centres (model frame m = dsp frame m/2, edges held).
  - Training: the features are computed per remixed sample by CPU worker processes (`--loader-workers`), prefetched
    within the epoch. The batch RNG stream is unchanged, so resume stays deterministic. At inference forward() computes
    them from the noisy waveform.
- Shared: init_from weights/gru_v5b.pt (`load_partial`: strict except the new heads), mixture gain augmentation,
  cmag + MR-STFT 0.5 + SI-SDR 0.3, lr 1e-4, 100 epochs, early stop after 8 val points, batch 16, val every 2 epochs.
  Colab: notebooks/colab_train.ipynb (branch exp/r2-v6, MyDrive/nrc_ckpt/<config_stem>/, resume / skip-completed,
  per-run stdout log on Drive).

Checks (2026-11-18, local CPU):
- gru_v5b inference is bit-identical on 20 validation files (max |diff| 0).
- Warm start with --max-steps 0, val subset 8: gru_v5b 2.716825 == gru_v6a == gru_v6b.
- 10-step smoke: losses finite, checkpoints written. Loss / val8 / s per step: v5b-config reference -4.579 / 2.7049 / 6.9;
  v6a (scale 1.0) -4.567 / 2.6207 / 7.5; v6b -4.578 / 2.6968 / 7.6 (loader wait 0.02 s/batch with 4 workers).
- v6b features: 52 ms per 3 s file on one core, about 0.83 s per batch of 16. Loader throughput is 0.92 / 0.49 / 0.29 /
  0.19 s per batch with 1 / 2 / 4 / 8 workers.
- CPU inference per 3 s file: v5b 0.20 s, v6a 0.16 s, v6b 0.22 s (v6b adds the feature cost).

Estimated T4 time per config, 100 epochs as an upper bound. Early stopping usually ends earlier: gru_v5b's best was
epoch 8.
- v6a: about 30 min, close to gru_v5b. The CPU step cost is +9 %. Reference: the v3b T4 log, 15.4 s/epoch incl. val.
- v6b: loader-bound on a standard 2-vCPU Colab runtime. About 0.5-0.65 s/batch, i.e. 40-55 s/epoch, plus about 10 s of
  feature work per val point and about 1 min of aux stats. Total about 75-100 min. With 8 vCPUs about 35 min.
## Status 2026-11-22: r2-prep submission -> gru_v8p2 (differentiable-PESQ fine-tune; tag r2-ready moved)

- gru_v8p2 = gru_v5b fine-tuned on Colab (branch exp/r2-dpesq, docs/exp_r2_dpesq.md) with
  0.2 cmag + 0.32 torch-pesq PesqLoss (torchaudio pieces replaced by src/nrc/tpesq.py). Adam 5e-5, seed 0, best epoch 99 of 100.
- eval.py (exp/r2-dpesq): val PESQ 3.3983 / SNR 4.48 dB, val_remix 3.2378 / 4.31 dB. Paired d vs gru_v5b: val +0.702 (z 27.0),
  val_remix +0.721 (z 35.1). Probe -30/-20/-10/0 dB: 2.736/3.203/3.378/3.398. Round-2 rule PASS; highest val PESQ of v8p1/p2/p3.
- SNR is deliberately traded for PESQ (objective is PESQ-WB only): 16.81 -> 4.48 dB.
- Level norm ON (range_boost, gru_v8p2_lnrb): val +0.0005, val_remix +0.0005, -30 dB probe +0.064 (< 0.1 adoption bar). Kept OFF.
- Promotion brings only weights/gru_v8p2.pt and configs. Inference code is unchanged; the r2-prep code reproduces the exp-branch
  outputs bit for bit. torch-pesq, torchaudio and src/nrc/tpesq.py stay training-only: they are not in requirements.txt, and
  e2e (d) checks that they cannot be imported.
- Fallback: gru_v5b (configs/dl/gru_v5b_off.yaml, weights/gru_v5b.pt).
- (2026-11-22) Still training on Colab then: gru_v8p3 (resume) and gru_v8p2x; both evaluated 2026-11-23 (above).

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
