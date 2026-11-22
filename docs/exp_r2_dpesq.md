# exp/r2-dpesq: gru_v5b fine-tuned with a differentiable PESQ loss (torch-pesq), gru_v8p1/p2/p3. Colab prep

Date 2026-11-19. Branch exp/r2-dpesq from exp/r2-mgan (e1d4d8d, which has the RNG-resume fix and the fail-loud notebook).
This document covers the design, the proxy check and the smoke test. There are no training or evaluation results yet.
main, dl-main and r2-prep are untouched. requirements.txt, main.py and the inference path are unchanged.

## Why
- MetricGAN (docs/exp_r2_mgan.md) is closed. Its learned discriminator never tracked PESQ (corr 0.00 / -0.08), and the
  generator exploited it.
- torch-pesq (audiolabs, MIT) is a fixed, untrained, differentiable re-implementation of the P.862 disturbance model.
  It has no time alignment, and it uses IIR level alignment.
- Because it is never trained, it has no weights the generator can game. It can still be gamed through its own
  approximations, which is the PESQetarian point (de Oliveira et al. 2024: 3.82 on VB-DMD with this loss alone).
- Our objective is PESQ-WB only, so that is acceptable. val_remix stays the generalisation guard, and the train64 vs val
  gap is logged every epoch.

## torch-pesq without torchaudio (src/nrc/tpesq.py; training only, never imported by main.py)
torch-pesq 0.1.2 imports torchaudio, whose last release (2.11.1) has no build for torch >= 2.12; we pin 2.15.0.
torch-pesq uses only three torchaudio pieces, and src/nrc/tpesq.py replaces all three:
- `Resample(16000, 16000)` is the identity. Any other rate raises.
- `Spectrogram` becomes |torch.stft|^2 with the same window and framing.
- `lfilter` is an IIR filter with zero initial state. On an N-sample input it equals the causal convolution with the
  first N impulse-response samples, so it is computed as an FFT convolution with h[0:N]. h is computed in float64 by
  scipy, so the result is exact up to rounding and differentiable.

Checked against scipy.signal.lfilter on a validation file: max |err| 1.3e-11 / 2.2e-14 in float64, 7.3e-8 / 3.0e-7 in float32.
Install with `pip install --no-deps -r requirements-dpesq.txt` (torch-pesq 0.1.2, torchtyping 0.1.5, typeguard 2.13.3).
requirements-dev.txt points to this file. A plain `-r` include would make pip try to resolve torchaudio against torch 2.15
and fail.

## Proxy check (scripts/r2_dpesq_proxy.py; logs/r2_dpesq_proxy.log, per-file logs/r2_dpesq_proxy_files.csv): PASS
Data: gru_v5b (gru_v5b_off, 16-bit like main.py), noisy, and dsp-v8 (cached val_remix outputs), scored on validation
(164 files) + val_remix (328 files).
torch-pesq = `PesqLoss(1.0, sample_rate=16000).mos`; pesq = pesq 0.0.4 WB.

| comparison | Pearson | Spearman | n |
|---|---|---|---|
| per file, v5b | 0.9968 | 0.9962 | 492 |
| per file, noisy | 0.9859 | 0.9766 | 492 |
| per file, dsp-v8 (val_remix) | 0.9973 | 0.9971 | 328 |
| paired diff v5b - noisy, all (**gate >= 0.6**) | 0.9930 | **0.9963** | 492 |
| paired diff v5b - noisy, validation | 0.9970 | 0.9958 | 164 |
| paired diff v5b - noisy, val_remix | 0.9913 | 0.9960 | 328 |
| paired diff v5b - dsp-v8, val_remix | 0.9954 | 0.9892 | 328 |

Mean levels agree within 0.02, for example v5b validation: pesq 2.6962, torch-pesq 2.6824. The paired-difference signs agree
on 99.1-100 % of files.

The gradient of PesqLoss on 3 s inputs (peak < 1) is finite. Exact digital silence is not:
- deg = 0, ref = 0 or both = 0 gives NaN, from 0/0 in the level alignment.
- 1e-9 noise in either signal is finite. The gradient w.r.t. a near-silent estimate is very large (7e6), but grad clip 5
  bounds it.
- The training loss therefore drops rows whose clean or estimate has max|x| < 1e-8 (count logged per epoch).
- If the PESQ term is still non-finite, that step skips the term and counts it. More than 5 in an epoch aborts.
- The quietest training clean file has RMS 0.0076, about 2e-4 after the -30 dB gain augmentation.

## Loss and lambda_pesq
loss = w_cmag cmag + w_mrstft MR-STFT + w_sisdr (-SI-SDR) + lambda_pesq * mean_b(0.1 d_symm + 0.0309 d_asymm).
The last term is `PesqLoss(1.0).forward`. The regression part reuses train_mgan.g_regression, which has the ops and
summation order of train.batch_loss.

**lambda_pesq = 0.32.** At gru_v5b init, over 8 seed-0 remix batches of 16, the parameter-gradient norm of v5b's
regression loss (1.0 / 0.5 / 0.3) is 0.44-2.09. The norm of the PESQ term is 1.08-7.62. The ratio averages 0.3186
(min 0.135, max 0.514, CV 0.39), rounded to 0.32. All three variants use this lambda (x1).

| config | w_cmag | w_mrstft | w_sisdr | lambda_pesq | intent |
|---|---|---|---|---|---|
| gru_v8p1 | 1.0 | 0.5 | 0.3 | 0.32 | v5b loss + PESQ term with equal gradient weight |
| gru_v8p2 | 0.2 | 0 | 0 | 0.32 | weak cmag anchor |
| gru_v8p3 | 0 | 0 | 0 | 0.32 | PESQ only (PESQetarian-style) |

Every variant uses the same training setup:
- Warm start from weights/gru_v5b.pt, magnitude mask.
- Adam lr 5e-5 (constant), grad clip 5, seed 0, 100 epochs, batch 16, gru_v5b's remix stream.
- Validation every epoch on the full validation split: pesq 0.0.4 on 16-bit output (selection) and SNR.
- Early stop after 20 validations without a gain.
- best.pt holds inference weights only (`save_weights`, loads with weights_only=True). last.pt is the full state every
  epoch (~28 MB). `--resume` restores the RNG with train_mgan.restore_torch_rng.

Train log columns (scripts/train_dpesq.py, one row per epoch):
- loss, plus l_cmag / l_mrstft / l_sisdr / l_pesq (unweighted; empty if the term is off)
- **tpesq_mos**: mean torch-pesq MOS of the training batches, also computed (no_grad) when lambda is 0
- pesq_rows_dropped, pesq_nonfinite
- **train64_pesq**: pesq 0.0.4 on a fixed 64-file stratified train subset (seed 0, original mixtures)
- val_pesq, val_snr, **gap** = train64 - val, val_failed
- lr, epoch_s, val_s, best_pesq, early_stop

## CPU smoke test (scripts/r2_dpesq_smoke.py, log logs/r2_dpesq_smoke.log): PASS
| check | result |
|---|---|
| lambda calibration | 0.3186 -> 0.32; the three configs match |
| lambda 0, v8p1 weights (= v5b) | train_dpesq loss == train.batch_loss bitwise (-4.9376459122) |
| v8p1, 2 steps | loss -4.4066 / -3.5763, PESQ term 1.660 / 2.010, finite |
| v8p2, 2 steps | loss 0.5322 / 0.6439, PESQ term 1.660 / 2.005, finite |
| v8p3, 2 steps | loss 0.5311 / 0.6416, PESQ term 1.660 / 2.005, finite |
| train_log row | train64_pesq / val_pesq / gap / tpesq_mos written (3-file subsets, so the values are not meaningful) |
| best.pt (each variant) | loads with weights_only=True (keys config, epoch, format, model_cfg, model_state, pesq) |
| main.py --config <tmp -> best.pt>, 3 validation files | names kept, 48000 samples @ 16 kHz mono in = out (3/3 per variant) |
| resume after 1 step (v8p1) | resumed step-2 loss -3.576293 == uninterrupted step 2 |

Local CPU cost: 6.6-7.2 s/step, of which the PESQ term's forward+backward w.r.t. the wave is 0.53-0.69 s for 16 x 3 s.

## Estimated Colab T4 time (not measured on GPU; epoch 1 prints s/step)
- Steps: gru_v5b took about 0.1 s/step on the T4. The PESQ term adds FFT convolutions (16 x 131072 points), a 49-band Bark
  matrix product and elementwise ops, which we estimate at 0.05-0.2 s on the GPU. That gives 0.15-0.3 s/step, or 12-25 s
  per 81-step epoch.
- PESQ scoring: 164 validation + 64 train64 files of pesq 0.0.4 on Colab's 2 CPUs. Validation alone was 14 s in the
  MetricGAN runs, so about 20 s for both.
- **Per epoch about 35-45 s. Per variant up to about 1-1.3 h for 100 epochs, much less with early stopping
  (patience 20). Whole queue up to about 4 h**, plus about 5 min of setup per session.
- Unlike MetricGAN, training is no longer PESQ-bound: there are no per-step PESQ calls.

## Evaluation plan (after Colab)
1. Copy each best.pt to weights/gru_v8pN.pt (the configs already point there, with level norm off).
2. Run eval.py on validation and val_remix at a clean hash.
3. Report the paired deltas vs gru_v5b, the level probe and the round-2 rule, as in docs/exp_r2_v7.md.
4. The val_remix z must clearly exceed the seed noise; gru_v5b_s2 alone reached z 1.97.
5. Also compare tpesq_mos with the real PESQ over training (PESQetarian check). If torch-pesq rises while train64 stays
   flat, the generator is gaming the proxy.

Files: src/nrc/tpesq.py, scripts/train_dpesq.py, scripts/r2_dpesq_proxy.py, scripts/r2_dpesq_smoke.py,
configs/dl/gru_v8p{1,2,3}.yaml, requirements-dpesq.txt, notebooks/colab_train.ipynb (queue), logs/r2_dpesq_proxy.log,
logs/r2_dpesq_proxy_files.csv, logs/r2_dpesq_smoke.log (force-added).

## Results (2026-11-22; Colab best.pt -> weights/gru_v8pN.pt; eval.py at e42d531 / 3679106)
gru_v8p3 is partial: its Colab session ended at epoch 17 (best so far). It is re-queued and resumes from Drive `last.pt`.
Paired deltas are vs gru_v5b_off. Full tables: logs/r2_dpesq_compare.txt (scripts/r2_v6_compare.py).

| model | val PESQ | val SNR | val d (z) | val_remix PESQ | val_remix SNR | val_remix d (z) | probe -30/-20/-10/0 dB | rule |
|---|---|---|---|---|---|---|---|---|
| gru_v5b_off | 2.6962 | 16.81 | - | 2.5170 | 15.45 | - | 2.614/2.670/2.684/2.696 | ref |
| gru_v8p1 | 2.7849 | 16.85 | +0.089 (13.9) | 2.6189 | 15.46 | +0.102 (23.5) | 2.709/2.761/2.775/2.785 | PASS |
| **gru_v8p2** | **3.3983** | 4.48 | +0.702 (27.0) | **3.2378** | 4.31 | +0.721 (35.1) | 2.736/3.203/3.378/3.398 | PASS |
| gru_v8p3 (ep 17) | 3.0447 | 9.43 | +0.349 (16.3) | 2.8904 | 8.59 | +0.373 (23.6) | 2.839/3.000/3.041/3.045 | PASS |

Top-5 types (validation / val_remix delta vs v5b): v8p2 broadband_noise +0.996 / +1.007, aircraft_cockpit +0.921 / +0.911,
machine +0.742 / +0.773, wind +0.636 / +0.599, alarm_siren +0.441 / +0.490. v8p3: +0.561/+0.632, +0.500/+0.494,
+0.325/+0.338, +0.301/+0.309, +0.235/+0.312.
SNR is traded for PESQ: the PESQ term alone does not constrain level or phase. v8p2's SNR is 4.5 dB, against 16.8 dB for v5b
and 9.6 dB for the noisy input. That is acceptable because the objective is PESQ-WB only.
main.py CPU time on 164 validation files (logs/time_r2dpesq_*.log, sequential): v5b 58.4 s, v8p1 67.9 s, v8p2 63.0 s,
v8p3 61.2 s. The same architecture is used throughout, so the differences come from machine load.

### Level normalisation for gru_v8p2 (configs/dl/gru_v8p2_lnrb.yaml; ON = gru_v1_lnrb range_boost; logs/eval_gru_v8p2_lnrb_*)
| set | OFF | ON | paired d (SE, z) |
|---|---|---|---|
| validation | 3.3983 | 3.3988 | +0.0005 (0.0004, 1.29) |
| val_remix | 3.2378 | 3.2384 | +0.0005 (0.0005, 1.05) |
| probe -30 dB | 2.7360 | 2.7995 | +0.0636 (0.0090, 7.03) |
| probe -20 dB | 3.2034 | 3.2156 | +0.0122 |
| probe -10 dB | 3.3776 | 3.3752 | -0.0025 |
| probe 0 dB | 3.3983 | 3.3988 | +0.0005 |
Adoption rule: val and val_remix d >= -0.003 (met), and the -30 dB probe must gain >= 0.1. It gained +0.064, so the rule
is not met. **Decision: level norm OFF** (configs/dl/gru_v8p2.yaml).

### Queue (Colab, 2026-11-22): gru_v8p3 (resume), gru_v8p2x (init gru_v8p2, fresh Adam 5e-5, 200 ep, patience 30).
