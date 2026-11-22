# exp/r2-mgan — MetricGAN+-style PESQ-discriminator fine-tuning of gru_v5b (gru_v6m1/m2/m3): Colab prep

Date 2026-11-18. Branch exp/r2-mgan from exp/r2-v7 (45c044e).
**Status (2026-11-19): closed, no promotion.** See "Result" at the end; the follow-up is exp/r2-dpesq (docs/exp_r2_dpesq.md).
main, dl-main and r2-prep are untouched. requirements.txt, main.py and the inference path are unchanged.

## Why
The objective is PESQ-WB only. SNR and listening quality may be traded for PESQ, and val_remix generalisation is the only guard.
gru_v7a showed that SNR-type losses (complex spectral, SI-SDR) raise SNR (+0.4 dB) and lower PESQ (-0.03).
gru_v5b is trained only on regression losses (cmag + MR-STFT + SI-SDR), so nothing in its objective tracks PESQ directly.
MetricGAN+ (Fu et al. 2021) learns a differentiable PESQ surrogate D and trains the generator to raise D's score.
The three variants step from "v5b plus a small metric term" (m1) to "mostly metric" (m3), which brackets how much regression
fidelity to give up.

## Discriminator (src/nrc/models/metric_disc.py; training only, never imported by main.py)
- Input: (B, 2, 257, T) stack of |STFT|^0.3 of x and the clean reference. Same STFT as G: n_fft 512, hop 128, periodic hann.
  T = 376 for 3 s segments.
- Body: 4 x [Conv2d 5x5, stride 2, padding 2, spectral norm, LeakyReLU 0.3], with channels 2 -> 16 -> 32 -> 32 -> 64.
- Head: global average pooling, then spectral-norm Linear 64 -> 50 -> 10 (LeakyReLU), Linear 10 -> 1, sigmoid.
  **94,315 parameters.**
- No batch norm. A D batch mixes clean, enhanced, noisy and replay rows, so batch statistics would depend on its composition.
- Target: (PESQ-WB - 1.0) / 3.5, clipped to [0, 1].
- PESQ: pesq 0.0.4 on the 16 kHz waveform after 16-bit quantisation, which is exactly how validation and main.py output are
  scored (train._pesq16). It runs in a process pool of os.cpu_count() workers. A sample whose PESQ fails is dropped from that D batch.
- Replay buffer: a reservoir sample (capacity 512, float16 on CPU) of past (enhanced, clean, target) rows. Replayed rows make up
  20 % of every D batch: 48 current rows (16 clean/clean, 16 enhanced, 16 noisy) + 12 replay rows.

## Training step (scripts/train_mgan.py; scripts/train.py is unchanged)
Each batch uses the same remix stream as train.py: 16 segments of 3 s with v5b augmentation.
1. G forward. The regression terms use exactly train.batch_loss's ops and summation order, and a term with weight 0 is skipped.
   The G loss is w_cmag cmag + w_mrstft MR-STFT + w_sisdr (-SI-SDR) + lambda_metric mean (D(enh, clean) - 1)^2.
   During this step D is in eval mode (no spectral-norm power iteration) and its parameters are frozen.
2. PESQ of the enhanced batch (G before this step's update) and of the noisy batch is submitted to the pool. It runs while G
   back-propagates.
3. G step: Adam, lr 1e-4, grad clip 5.
4. D step: Adam, lr 2e-4, MSE loss with targets (clean, clean) -> 1, (enhanced, clean) -> its PESQ target,
   (noisy, clean) -> its PESQ target, and replay rows -> their stored targets. The enhanced rows are then added to the replay buffer.

**D pretrain:** before any G update, D trains alone for 2 epochs (162 steps) on frozen gru_v5b outputs with the same D step,
`--pretrain-d`. The resulting D weights, D Adam state and replay buffer are shared by all three variants (`--d-init`).
The pretrain log has corr_d_pesq, the per-epoch correlation between D(enhanced) and the true PESQ target, which checks
whether D has learned PESQ before G uses it.

All variants warm-start from weights/gru_v5b.pt and use a magnitude mask, seed 0, 60 epochs and constant learning rates
(no plateau schedule). Validation runs every epoch on the full validation split and reports PESQ (used for selection) and SNR.
Training stops early after 15 validations without a gain.
best.pt holds inference weights only (save_weights; loads with weights_only=True).
last.pt is the full state every epoch: G, D, both optimizers, replay buffer and RNG, about 130 MB at a full buffer.

| config | w_cmag | w_mrstft | w_sisdr | lambda_metric | intent |
|---|---|---|---|---|---|
| gru_v6m1 | 1.0 | 0.5 | 0.3 | 0.05 | v5b loss plus a small metric term |
| gru_v6m2 | 1.0 | 0.2 | 0.0 | 0.3 | drop SI-SDR, weaker MR-STFT, metric term comparable to the regression loss |
| gru_v6m3 | 0.2 | 0.0 | 0.0 | 1.0 | metric-dominated; a weak cmag anchor only |

Train log columns (one row per epoch): g_loss, g_cmag / g_mrstft / g_sisdr (unweighted, empty if the term is off),
g_adv = mean (D(enh) - 1)^2, d_loss, d_enh = mean D(enhanced), d_clean = mean D(clean, clean), train_pesq_enh /
train_pesq_noisy (true PESQ of the training batches), pesq_failed, val_pesq, val_snr, val_failed, lr_g, lr_d, epoch_s,
val_s, best_pesq, early_stop.

## CPU smoke test (scripts/r2_mgan_smoke.py, log logs/r2_mgan_smoke.log): PASS
| check | result |
|---|---|
| D forward/backward (16 x 3 s) | output in [0.4685, 0.4703] at init, finite grads |
| PESQ (3 s files) | 63-76 ms/call serial. One training step = 32 calls: 0.87 s in a 16-process pool, 1.38 s with 2 processes, 2.44 s with 1 |
| lambda 0, gru_v5b weights, v6m1 = v5b loss weights | train_mgan G loss == train.batch_loss bitwise (-4.9376459122) |
| D pretrain, 2 steps | d_loss 0.155 -> 0.135, PESQ failures 0 |
| v6m1, 2 G+D steps | g_loss -4.924 / -4.207, d_loss 0.1345 / 0.1305, all finite |
| v6m2, 2 G+D steps | g_loss 0.199 / 0.241, d_loss 0.1345 / 0.1303, all finite |
| v6m3, 2 G+D steps | g_loss 0.274 / 0.269, d_loss 0.1345 / 0.1303, all finite |
| best.pt (each variant) | loads with weights_only=True (keys config, epoch, format, model_cfg, model_state, pesq) |
| last.pt (each variant) | holds model, disc, opt_g, opt_d, replay |
| main.py --config <tmp config -> best.pt>, 3 validation files | names kept, 48000 samples @ 16 kHz mono in = out (3/3 per variant) |

Local CPU step time: G 2.4-3.8 s, D 0.46-0.51 s, PESQ wait 0 s, because the 16-process PESQ overlaps the CPU backward pass.
That is about 3.3 s/step.

## Estimated Colab T4 time
- G on T4: gru_v5b took 8.5 s per 81-step epoch, about 0.1 s/step. The D forward/backward on 60 rows adds a few tens of ms.
- PESQ: 32 calls per step on Colab's 2 CPUs. That is 1.38 s here and probably ~1.5-1.8 s on Colab's slower cores.
  Only about 0.1 s of it overlaps the G backward pass, so **training is PESQ-bound**.
- Per epoch: 81 x ~1.5-1.9 s plus validation (~15-20 s, 164 files) gives **~2.5-3 min/epoch**.
- Per variant: **up to ~2.5-3 h for 60 epochs.** With early stopping, a run that peaks early ends 15 epochs later.
- D pretrain: 162 steps without a G step, **~5 min**. Whole queue: **up to ~8-9 h**, plus ~5 min of setup per session.
  It survives disconnects through `--resume`.
- A runtime with more CPUs cuts the PESQ part almost linearly. `--pesq-jobs` defaults to os.cpu_count().

## Evaluation plan (after Colab)
Copy each best.pt to weights/gru_v6mN.pt (the configs already point there with level norm off) and run eval.py on validation and
val_remix at a clean hash. Report paired deltas vs gru_v5b, the level probe and the round-2 rule as in docs/exp_r2_v7.md.
Report SNR as well, but it is not a constraint. The val_remix z must clearly exceed the seed noise: gru_v5b_s2 alone reached
z 1.97.

Files: src/nrc/models/metric_disc.py, scripts/train_mgan.py, scripts/r2_mgan_smoke.py, configs/dl/gru_v6m{1,2,3}.yaml,
notebooks/colab_train.ipynb (queue), logs/r2_mgan_smoke.log (force-added).

## Result (2026-11-19, Colab T4): closed, no promotion
Source: the Colab train logs (validation every epoch, full validation split, pesq 0.0.4). No full eval.py run was made on
validation / val_remix, so there are no rows in results/experiments.csv. Logs (force-added):
logs/r2_mgan_d_pretrain_log.csv / _stdout.log, logs/r2_mgan_gru_v6m{1,2}_train_log.csv / _stdout.log,
logs/train_log_v6m3.csv (added 2026-11-22 on exp/r2-dpesq; train log only, no stdout copy).

| config | best epoch | best val PESQ | vs gru_v5b 2.6962 | log |
|---|---|---|---|---|
| gru_v6m1 | 2 | 2.6977 | +0.0015 (one epoch, within seed noise) | 17 epochs, early stop |
| gru_v6m2 | 2 | 2.6688 | -0.0274 | 13 epochs in the downloaded copy (no early_stop row) |
| gru_v6m3 | 2 | 2.6502 | -0.0460 | 17 epochs, early stop (ep 17 val 2.6163) |

Diagnosis: the generator exploited the discriminator.
- The D pretrain never learned PESQ. corr_d_pesq, the correlation between D(enhanced) and the true PESQ target over an epoch,
  was 0.0017 after epoch 1 and -0.0794 after epoch 2.
- In G+D training, D rated the enhanced output 0.69 at epochs 1-3, which corresponds to PESQ 1 + 3.5 * 0.69 = 3.41. The true
  train PESQ of those batches was 2.91-2.99.
- Over training, D(enh) fell to 0.59 for m1, 0.61-0.62 for m2 and 0.60 for m3 (D(clean) 0.62 -> 0.83, train PESQ 2.88 -> 2.86), while D(clean) rose from 0.68 to 0.84. The true train PESQ
  stayed flat (m1 2.99 -> 2.89-2.95) and val PESQ went down after epoch 2 in every variant.
- So the adversarial term mostly measured D's own miscalibration, not PESQ. The best epoch is the second warm-start epoch, and
  that is noise around gru_v5b.

Decision: no full eval, nothing promoted, r2-prep unchanged. The follow-up replaces the learned D with a fixed differentiable
PESQ approximation (torch-pesq), so the generator has no trainable critic to game. Branch exp/r2-dpesq, gru_v8p1/p2/p3.
The weights/gru_v6m{1,2,3}.pt copies stay local and untracked (.git/info/exclude).
