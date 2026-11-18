# exp/r2-v7 — gru_v7a (crm from scratch, compressed spectral loss) + gru_v5b_s2 (seed control): Colab prep

Date 2026-11-18. Branch exp/r2-v7 from exp/r2-v6 (9007ee9). Colab results and evaluation: see "Results" below.
**Decision: no candidate passes the round-2 rule; r2-prep (gru_v5b_off) is unchanged.**

## Why
gru_v6a (crm) was a warm start from v5b with a zero-init imaginary head at mask_imag_scale 0.1 and lr 1e-4. Validation
selected epoch 2, so it recovered 0.04 % of the clean-phase headroom (docs/exp_r2_v6.md). gru_v7a trains the
crm model from random init, so selection cannot fall back to the v5b function. gru_v5b_s2 measures how much a
v5b-style result moves with the seed alone.

## Configs
| | gru_v7a | gru_v5b_s2 |
|---|---|---|
| architecture | v5b (conv2d front end, 16 ch GroupNorm, BiGRU 2 x 256) + crm head, mask_imag_scale 1.0 (2,505,492 params) | = gru_v5b (2,373,651 params) |
| init | random (seed 0, as v5b / v2b) + train-set norm stats; no init_from | init_from weights/gru_v4b.pt (= gru_v5b) |
| seed | 0 | 2 (gru_v5b: 0) |
| loss | 0.7 cmag + 0.3 ccplx (Braun & Tashev 2021, c 0.3, alpha 0.3) + 0.5 MR-STFT + 0.3 (-SI-SDR dB) | 1.0 cmag + 0.5 MR-STFT + 0.3 (-SI-SDR dB) (= gru_v5b) |
| lr / schedule | 1e-3, ReduceLROnPlateau x0.5, patience 3 val points, min 1e-5 | 2e-4, same schedule (= gru_v5b) |
| epochs / early stop | 300 / 75 val points (150 epochs) without gain | 100 / 8 val points (= gru_v5b) |
| augmentation | v5b remix: p_original 0.5, SNR U(-5, 20) dB, mixture gain (p_wide 0.5: U(-30, +6) dB else U(-6, +3) dB) | same |
| selection | best.pt = best full-validation PESQ (every 2 epochs); last.pt every epoch | same |

Epoch budget: gru_v5b itself is the fourth fine-tune stage (v2b 150 -> v3b 92 -> v4b 64 -> v5b 24 epochs, logs/train_log_v*.csv).
The only from-scratch run in its lineage is gru_v2b: 150 epochs, lr 1e-3, no early stop. gru_v7a therefore uses
v2b's lr and 300 epochs (about the 330-epoch cumulative lineage). Its patience of 150 epochs means it always
trains at least as long as v2b did.

Code: `train.lambda_cmag` (default 1.0) and `nrc.losses.compressed_spectral_loss`. The training loss moved into
`scripts/train.py:batch_loss` with the same ops, and the smoke test checks that gru_v5b's loss on one batch is
bit-identical to the exp/r2-v6 inline loss. Unit test: tests/test_losses.py (0 for identical inputs, finite
loss and gradients for silence, the weights match 0.7 cmag + 0.3 ccplx, and the complex term reacts to phase while
cmag does not). weights/gru_v4b.pt was copied from dl-main for the s2 warm start.

## Estimated T4 time (from logs/train_log_v5b.csv: 8.5 s/epoch + 13.5 s per validation; gru_v6a crm ~9 s/epoch)
- gru_v7a: ~16 s/epoch -> ~80 min for 300 epochs. It cannot early-stop before about epoch 152 (~40 min).
- gru_v5b_s2: ~15 s/epoch -> ~6 min if it stops where v5b did (epoch 24), at most ~26 min.
- Plus about 5 min setup per session.

## CPU smoke test (scripts/r2_v7_smoke.py, log logs/r2_v7_smoke.log): PASS
| | gru_v7a | gru_v5b_s2 |
|---|---|---|
| fixed batch (16), loss before / after step 1 / after step 2 | -2.4137 / -2.1134 / -2.4397 (finite; net decrease, not monotone) | -4.7913 / -4.8497 / -4.8841 (finite, monotone) |
| train.py --max-steps 2 --val-subset 3 | step 2, val PESQ 1.5042, best.pt + last.pt + train_log.csv written | step 2, val PESQ 2.3129, same files |
| best.pt load with weights_only=True | OK (model_cfg mask_type crm) | OK |
| last.pt load with weights_only=True | OK | OK |
| inference, 3 validation files | 48000 samples @ 16 kHz mono in = out (3/3) | 3/3 |

The v7a bump after step 1 comes entirely from the SI-SDR term: -3.218 -> -2.886, while cmag+ccplx stays at
0.047 and MR-STFT falls 0.757 -> 0.726. It also appears with the imaginary path disabled (scale 1e-4: -2.414 ->
-2.185), so it is the first Adam step from random init at lr 1e-3, not the crm head. At lr 5e-4 the bump
shrinks to -2.414 -> -2.391. Over 5 steps the loss falls to -2.66 in all three settings. The pass rule was
therefore relaxed from strict monotonicity to "finite and lower after step 2"; the monotone flag is still printed.

## Results (Colab T4 training; evaluation at clean hash ea14cac, 16-bit in-memory, eval.py)
Weights, train logs and inference configs were committed at ea14cac. The inference configs are configs/dl/gru_v7a_off.yaml,
gru_v7a_e300_off.yaml and gru_v5b_s2_off.yaml, all with level norm off like gru_v5b_off. weights/gru_v7a_last.pt is
the full training state (model + optimizer + RNG, 30 MB). It loads with weights_only=True but has no inference format, so
scripts/export_last.py copies its state dict unchanged into weights/gru_v7a_e300.pt (bit-identical, reloads with
weights_only=True). All four checkpoints load with weights_only=True.

### Train logs (val PESQ = full validation every 2 epochs)
- gru_v7a: ran all 300 epochs without an early stop. Best epoch 262 (2.6635); epoch 300 2.6630; since_best 19 val points at the end.
  Curve: ep 2 1.749, 22 2.314, 62 2.535, 102 2.613, 142 2.655, 202 2.660, 262 2.664. The lr fell 1e-3 -> 1e-5 (floor) by
  epoch 160. Mean val PESQ was 2.6572 / 2.6588 / 2.6609 over epochs 151-200 / 201-250 / 251-300, so the run is plateaued
  (about +0.002 per 50 epochs at the lr floor) and still about 0.033 below v5b. ~9.3 s/epoch + 14.7 s per validation.
- gru_v5b_s2: early-stopped at epoch 36; best epoch 20 (2.6990). It is noisy around the warm start (2.670-2.699), like
  v5b (best epoch 8, 2.6962, stopped at 24).

### Main table
| | gru_v5b_off | gru_v5b_s2_off | gru_v7a_off (best, ep 262) | gru_v7a_e300_off (last) |
|---|---|---|---|---|
| val PESQ | 2.6962 | 2.6990 | 2.6635 | 2.6630 |
| val 0/5/10/15 dB | 1.953/2.410/2.702/3.083 | 1.957/2.415/2.707/3.083 | 1.909/2.372/2.671/3.056 | 1.911/2.374/2.670/3.054 |
| val easy/moderate/hard | 2.630/2.708/2.949 | 2.634/2.719/2.933 | 2.594/2.690/2.906 | 2.594/2.689/2.903 |
| val SNR (dB) | 16.81 | 16.89 | 17.19 | 17.19 |
| val paired delta ± SE (z) | — | +0.0028 ± 0.0041 (0.68) | -0.0326 ± 0.0107 (-3.04) | -0.0332 ± 0.0108 (-3.08) |
| val_remix PESQ / SNR | 2.5170 / 15.45 | 2.5216 / 15.52 | 2.4891 / 15.85 | 2.4896 / 15.85 |
| val_remix 0/5/10/15 dB | 1.825/2.318/2.723/3.206 | 1.821/2.325/2.730/3.213 | 1.784/2.270/2.726/3.176 | 1.786/2.271/2.726/3.175 |
| val_remix paired delta ± SE (z) | — | +0.0045 ± 0.0023 (1.97) | -0.0280 ± 0.0075 (-3.71) | -0.0274 ± 0.0076 (-3.63) |
| probe -30/-20/-10/0 dB | 2.614/2.670/2.684/2.696 | 2.616/2.673/2.688/2.699 | 2.584/2.620/2.643/2.664 | 2.579/2.617/2.643/2.663 |
| main.py CPU, 164 val files | 24.0 s (0.146 s/file) | 24.8 s (0.151) | 24.3 s (0.148) | 25.9 s (0.158) |

The CPU times were measured back to back in one session (logs/time_r2v7_*.log).

Per-SNR paired deltas: v7a loses at every validation SNR (-0.027 to -0.044, each |z| < 1.8). On val_remix it loses
at 0 dB (-0.041, z -3.0), 5 dB (-0.048, z -3.1) and 15 dB (-0.030, z -1.9), and ties at 10 dB (+0.003).
v5b_s2 gains +0.007 at 5/10/15 dB on val_remix (z about 1.6 each) and -0.004 at 0 dB.

Weak-5 categories, paired delta vs v5b (validation | val_remix):

| category (n val / val_remix) | v5b_s2 | v7a best | v7a last |
|---|---|---|---|
| aircraft_cockpit (14/28) | +0.003 \| +0.024 | -0.012 \| -0.039 | -0.015 \| -0.041 |
| factory (5/10) | +0.015 \| +0.026 | -0.067 \| -0.044 | -0.066 \| -0.042 |
| broadband_noise (10/20) | +0.003 \| +0.010 | -0.068 \| -0.031 | -0.075 \| -0.032 |
| destroyer_operations (5/10) | +0.017 \| +0.024 | -0.011 \| -0.058 | -0.016 \| -0.061 |
| hf_channel (4/8) | -0.024 \| -0.006 | -0.043 \| +0.006 | -0.041 \| +0.005 |

### Seed noise (training-noise reference)
v5b_s2 - v5b: +0.0028 ± 0.0041 on validation and +0.0045 ± 0.0023 on val_remix. Changing only the seed moves the result
by about 0.003-0.005 PESQ. v7a's deltas (-0.033 val, -0.028 val_remix) are 6-12 times larger and negative, so v7a is
really worse than v5b, not unlucky. The seed control itself misses the val_remix gate by only 0.03 in z
(z 1.97). This shows that the gate's z >= 2 can be reached by re-seeding alone, which argues against promoting on a z near 2.

### Phase use (does the crm path act?)
Phase-headroom fraction (v7a - v5b) / headroom: -0.111 on validation (/0.293) and -0.107 on val_remix (/0.261).
The last checkpoint gives -0.113 / -0.105. v7a recovers none of the clean-phase oracle gain and loses about a tenth of it.

Mask phase on the 164 validation noisy files (scripts/r2_mask_phase.py, logs/r2_v7_mask_phase.txt):

| | imag scale | mean \|angle(M)\| | \|X\|²-weighted | bins > 5° | mean \|M_i\| | mean M_r |
|---|---|---|---|---|---|---|
| gru_v6a | 0.1 | 2.05° | 4.25° | 8.2 % | 0.0023 | 0.502 |
| gru_v7a best | 1.0 | 8.12° | 8.86° | 28.4 % | 0.0136 | 0.492 |
| gru_v7a last | 1.0 | 8.09° | 8.89° | 28.3 % | 0.0137 | 0.494 |

The v7a phase path does act: it rotates about 4 times as much as v6a and turns 28 % of bins by more than 5°. These
rotations do not raise PESQ. Training from scratch cost more magnitude quality (v5b's lineage had about 330 epochs in 4
fine-tune stages) than the phase term returned. v7a has higher SNR (+0.4 dB on both sets), as expected from the
complex/SI-SDR terms, but PESQ falls.

### Promotion check (round-2 rule)
| | val >= 2.6962 | val_remix delta > 0 and z >= 2 | probe >= v5b - 0.02 | result |
|---|---|---|---|---|
| gru_v7a_off | no (2.6635) | no (-0.0280, z -3.71) | no (all four levels, e.g. -20 dB 2.620 < 2.650) | FAIL |
| gru_v7a_e300_off | no (2.6630) | no (-0.0274, z -3.63) | no | FAIL |
| gru_v5b_s2_off | yes (2.6990) | no (+0.0045, z 1.97) | yes | FAIL |

No candidate passes, so r2-prep, configs/submission.yaml and tag r2-ready are untouched.

### Takeaways
- From-scratch crm with the compressed complex loss converges about 0.03 PESQ below v5b and stops improving at the lr floor.
  Training longer at this lr schedule is unlikely to close the gap (+0.002 per 50 epochs).
- A next crm attempt could warm-start the magnitude path from v5b and give the imaginary head its own larger lr or
  scale. The v6a and v7a results bracket that design.
- v5b_s2 is a near-miss (z 1.97) whose gain is the size of seed noise. Seed averaging (for example checkpoint averaging of
  v5b and v5b_s2, same architecture and warm start) is a cheap untested option that needs no retraining. It was not run here.

Files: scripts/r2_v6_compare.py (now takes --ref/--cands/--headroom; defaults reproduce logs/r2_v6_compare.txt exactly),
scripts/r2_mask_phase.py, scripts/export_last.py, logs/r2_v7_compare.txt, logs/r2_v7_mask_phase.txt,
logs/eval_gru_{v7a,v7a_e300,v5b_s2}_off_*.log, logs/time_r2v7_*.log, logs/run_r2_v7_evals.sh (logs/ is git-ignored and
stays local; logs/train_log_v7a.csv and train_log_v5b_s2.csv are force-added).
