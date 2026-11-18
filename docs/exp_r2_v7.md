# exp/r2-v7 — gru_v7a (crm from scratch, compressed spectral loss) + gru_v5b_s2 (seed control): Colab prep

Date 2026-11-18. Branch exp/r2-v7 from exp/r2-v6 (9007ee9). Prep only: no training results yet, no eval rows.

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
