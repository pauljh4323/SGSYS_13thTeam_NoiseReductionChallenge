# exp/r2-dmix: gru_v8p2z1 (train-time rms_norm) and gru_v8p2z2 (+ dynamic mixing). Colab prep

Date 2026-11-23. Branch exp/r2-dmix from r2-prep (tag r2-ready, a94e603). The training stack was brought over from
exp/r2-dpesq @ 791e7fa in commit 6f66fb9; main.py output on 3 validation files is byte-identical and selfcheck is OK.
main, dl-main and r2-prep are untouched. main.py, requirements.txt and the inference path are unchanged.
This document covers the design, the CPU tests and (at the end) the Colab training and evaluation results.

## Why
The submission is gru_v8p2y + rms_norm (val 3.5077, val_remix 3.3633). Its train64 PESQ is 4.07, so the train/val gap
is 0.56, and the training curve was flat for the last 30 epochs. Two mismatches are addressed:
- **Level.** At inference the model sees rms-normalised input (level_mode rms_norm, target_rms 0.079130). In training
  it saw input with a U[-40, +6] dB gain, so most of its capacity for level robustness is unused at inference.
- **Data diversity.** Only half of the training samples are remixed. The other half are the 1296 original pairs,
  which the model can memorise.

## Current augmentation of gru_v8p2y (src/nrc/data/remix.py, configs/dl/gru_v8p2y.yaml)
- With p_original 0.5 the sample is an original (clean, noisy) pair.
- Otherwise it is a random train clean plus a random train noise:
  - The noise is noise = noisy - clean, from files valid in logs/noise_check.csv.
  - It gets a random circular shift and a random sign.
  - It is mixed at SNR U[-5, 20] dB.
  - The noise file can be the same file as the clean one.
- Gain: one U[-40, +6] dB gain on every sample (original pairs included), applied to clean and mix alike.
- Peak guard: clean and mix are scaled down together if |mix| > 0.99.
- One numpy Generator (seed 0) drives all draws; each step is `data.batch(rng, 16)`.

## New options (defaults off; RNG stream and loss unchanged when off)
**Train-time rms_norm** (`train.rms_norm_target`):
- Applied after the peak guard: g = target / max(rms(mix), 1e-6), in float64. Clean and mix are both multiplied by g.
- There is no input peak cap afterwards, as at inference.
- The gain augmentation is off and makes no draws. Passing any gain option together with rms_norm raises an error.
- Validation and train64 use the same normalisation per file, plus the inference output peak cap (0.99). Training
  validation therefore equals `methods.gru_mask.enhance(level_mode rms_norm)` (max |diff| 6e-8 on 3 files).
- train_dpesq checks that the config's inference params use level_mode rms_norm with the same target_rms.

**Dynamic mixing** (`train.remix.p_dm`; Zeghidour & Grangier 2021). Each sample is built with probability p_dm:
1. Draw a random train clean i, then the noise of a random other train file j != i (valid noise only, random circular
   shift, no sign flip).
2. With probability 0.2, add a second noise k not in {i, j} (random shift) at U[-10, 0] dB relative to the first
   noise's power.
3. Scale the combined noise so that the SNR against the clean is U[-5, 20] dB. The SNR is defined against the total
   noise.
4. Apply the peak guard (<= 0.99), then rms_norm. Neither step changes the SNR.

| config | init | rms_norm (train + val) | gain aug | data |
|---|---|---|---|---|
| gru_v8p2y (reference) | gru_v8p2x | no | U[-40, +6] dB | 50 % original, 50 % remix |
| gru_v8p2z1 | gru_v8p2y | yes, 0.079130 | off | 50 % original, 50 % remix |
| gru_v8p2z2 | gru_v8p2y | yes, 0.079130 | off | 100 % dynamic mixing |

Settings shared by both configs:
- Same loss as gru_v8p2y: 0.2 cmag + 0.32 torch-pesq.
- Adam lr 5e-5 (constant), fresh optimizer, weights-only init.
- Seed 0, 200 epochs, patience 30.
- best.pt + last.pt, with train64 PESQ and gap logged every epoch.
- Inference params: level_mode rms_norm, target_rms 0.079130.

## CPU tests: PASS
`python tests/test_dmix.py` uses synthetic data:
- Dynamic mixing reproduces the drawn SNR within 0.1 dB on 100 draws, with rms_norm off and on.
- The second noise appears in 22/100 draws, and j != i always holds.
- rms_norm gives mix RMS = target, and an original pair comes out as exactly g * (clean, noisy) with no gain draw.
- Combining a gain option with rms_norm raises an error.
- With the options off, the RNG stream is unchanged.

`python scripts/r2_dmix_smoke.py --out <scratch>` uses real data (logs/r2_dmix_smoke.log):

| check | result |
|---|---|
| options off: RemixData vs exp/r2-dpesq version, gru_v8p2y settings, 200 draws | identical samples + RNG state |
| options off: gru_v8p2y training loss, 2 Adam steps | bitwise equal (0.3194180750, 0.3156099428) |
| z1 / z2 rms_norm level, 100 real draws | max rel err 1.8e-9 / 8.4e-10 |
| z2 dynamic mixing SNR, 100 real draws | max abs err 0.0000 dB, second noise 23/100 |
| z1, 2 steps | finite (0.3433, 0.3123) |
| z2, 2 steps | finite (0.4492, 0.3894) |
| best.pt (z1, z2) | loads with weights_only=True |
| training rms_norm validation path vs inference enhance(rms_norm), 3 val files | max diff 6.0e-8 |

## Estimated Colab T4 time
- The gru_v8p2y Colab log (183 epochs, early stop) shows a median of 8.6 s training + 20.3 s validation/train64 per
  epoch, about 29 s per epoch.
- The new data path adds about 16 numpy rolls and sums per step, a few ms. This is negligible.
- **Up to about 1.6 h per config** (200 epochs), less with an early stop, and up to about 3.3 h for the queue, plus
  about 5 min setup per session.

## Evaluation plan (after Colab)
1. Copy each best.pt to weights/gru_v8p2z{1,2}.pt; the configs already point there, with rms_norm on.
2. Run eval.py on validation and val_remix at a clean hash.
3. Compare paired deltas against gru_v8p2y + rms_norm, and run the level probe.
4. Compare the train64-val gap with gru_v8p2y's 0.56.

Files: src/nrc/data/remix.py, scripts/train_dpesq.py, configs/dl/gru_v8p2z{1,2}.yaml, tests/test_dmix.py,
scripts/r2_dmix_smoke.py, logs/r2_dmix_smoke.log, notebooks/colab_train.ipynb.

## Results 2026-11-23: gru_v8p2z1 / gru_v8p2z2 vs the submission gru_v8p2y + rms_norm (eval.py at a3e0005, 16-bit, --jobs 1)
Training (Colab; logs/train_log_v8p2z{1,2}.csv; both best.pt load with weights_only=True; 0 non-finite PESQ-loss steps):

| model | best epoch / run | train-log val | val SNR | train64 | gap (v8p2y 0.559) | curve |
|---|---|---|---|---|---|---|
| gru_v8p2z1 (rms_norm) | 140 / 170, early stop | 3.5215 | 1.80 | 4.140 | 0.619 | flat (last 20 ep mean 3.5165, previous 20 3.5160) |
| gru_v8p2z2 (rms_norm + dmix) | 117 / 147, early stop | 3.5452 | 1.35 | 4.026 | **0.481** | flat (3.5371 vs 3.5390) |

z1 fits train harder (gap widens 0.56 -> 0.62). With dynamic mixing (z2), train64 falls and val rises, so the gap
narrows to 0.48. In other words, dynamic mixing regularises as intended.

Evaluation: runner logs/run_r2_dmix.sh, 12 runs with 0 PESQ failures and 0 clipped samples; tables in
logs/r2_dmix_compare.txt. Paired deltas are vs gru_v8p2y + rms_norm.

| model | val PESQ | @0/5/10/15 dB | val SNR | val d (SE, z) | val_remix PESQ | val_remix d (SE, z) | probe -30/-20/-10/0 dB | rule |
|---|---|---|---|---|---|---|---|---|
| v8p2y rms_norm (submission) | 3.5077 | 2.904/3.281/3.555/3.776 | 1.88 | - | 3.3633 | - | 3.508 x4 | ref |
| v8p2z1 | 3.5215 | 2.918/3.306/3.565/3.786 | 1.80 | +0.0138 (0.0029, 4.77) | 3.3843 | +0.0210 (0.0029, 7.26) | 3.522 x4 | PASS |
| **v8p2z2** | **3.5452** | 2.906/3.313/3.597/3.823 | 1.35 | +0.0375 (0.0058, 6.46) | **3.4065** | +0.0432 (0.0039, 11.08) | **3.545 x4** | **PASS** |

Per-SNR paired deltas at 0/5/10/15 dB:

| model | validation | val_remix |
|---|---|---|
| z1 | +0.015/+0.024/+0.010/+0.010 (z 1.6/3.8/2.3/1.9) | +0.029/+0.020/+0.021/+0.015 (z 5.6/3.9/2.9/2.9) |
| z2 | +0.003/+0.032/+0.042/+0.047 (z 0.1/2.4/5.3/4.6) | +0.047/+0.042/+0.056/+0.027 (z 7.4/4.6/7.6/3.3) |

On validation, z2's 0 dB gain is flat (n 15, SE 0.024). On val_remix (n 82) the 0 dB gain is +0.047 at z 7.4.

Per difficulty, validation easy/moderate/hard is 3.449/3.642/3.775 for z2 (submission 3.417/3.605/3.715). On
val_remix it is 3.432/3.327/3.438 (3.390/3.283/3.392).

Top-5 types (delta on validation / val_remix):

| type | z1 | z2 |
|---|---|---|
| machine | +0.011 / +0.025 | +0.029 / +0.047 |
| wind | +0.013 / +0.016 | +0.041 / +0.055 |
| aircraft_cockpit | +0.008 / +0.014 | +0.049 / +0.054 |
| broadband_noise | +0.018 / +0.015 | +0.047 / +0.055 |
| alarm_siren | +0.005 / +0.027 | -0.013 (SE 0.017) / +0.014 |

z2 reverses v8p2y's broadband_noise loss. Its only negative entry is alarm_siren on validation, which is within 1 SE.

Caveat: the training checkpoint is selected on validation, and train-log val equals eval val (3.5215 / 3.5452), so val is
mildly optimistic. val_remix is the held-out gate: z2 gains +0.043 there at z 11.1.

Rule: val >= 3.5077, val_remix d > 0 with z >= 2, every probe >= 3.488. Both pass. **The highest val PESQ is gru_v8p2z2
(3.5452), so it is promoted to r2-prep** (configs/dl/gru_v8p2z2.yaml; level mode rms_norm, target_rms 0.079130).

## Second batch 2026-11-23: gru_v8p2z3a / gru_v8p2z3b (clean-side speaker augmentation). Colab prep
Why: gru_v8p2z2 still has a train64-val gap of 0.48. Every validation noise recording also appears in train, but the
validation speakers do not, so the remaining gap is most likely speaker mismatch. Dynamic mixing only varies the noise
side. These two runs add clean-side diversity. The objective is PESQ-WB only.

New `train.remix` options in src/nrc/data/remix.py. Defaults are off and make no RNG draws.
- They are applied to the drawn clean utterance before mixing: in the dynamic-mixing branch, and in the remixed half
  of the remix branch. Original pairs are never augmented, because their noisy signal is fixed.
- The target is the augmented clean, and the SNR is set against it.

| option | what it does | implementation |
|---|---|---|
| `p_speed`, `speed_range` (Ko et al. 2015) | With probability p_speed, the clean is resampled by f ~ U[0.9, 1.1], which changes tempo and pitch. The result is then randomly cropped (f < 1) or zero-padded at a random offset (f > 1) back to 48000 samples. | f is quantised to 1/200 (41 factors). scipy `resample_poly` with up/down = 200/round(200 f) and its own Kaiser FIR (`kaiserord` 80 dB, passband 0.95 and stopband 1.0 of the lower Nyquist). The 41 filters are cached, about 7 MB. scipy's default filter only gives about 16 dB just above the new Nyquist, so it is not used. Factor 1 returns an exact copy. |
| `p_eq`, `eq_low_hz`, `eq_high_hz`, `eq_gain_db` | With probability p_eq, a low-shelf (corner U[150, 500] Hz) and a high-shelf (corner U[2, 5] kHz) are applied, gains each U[-6, +6] dB. The clean is then rescaled to its original RMS. | RBJ Audio-EQ-Cookbook shelving biquads, shelf slope S = 1, cascaded with `sosfilt` in float64. |

Draw order per sample:
1. Speed: 1 decision draw, then factor and offset.
2. EQ: 1 decision draw, then 2 corners and 2 gains.

| config | init | added to gru_v8p2z2 |
|---|---|---|
| gru_v8p2z3a | weights/gru_v8p2z2.pt (weights only, fresh Adam 5e-5) | p_speed 0.5, U[0.9, 1.1] |
| gru_v8p2z3b | weights/gru_v8p2z2.pt (weights only, fresh Adam 5e-5) | p_speed 0.5 + p_eq 0.5 (150-500 Hz / 2-5 kHz, +-6 dB) |

Everything else is as in gru_v8p2z2:
- Loss 0.2 cmag + 0.32 torch-pesq, lr 5e-5.
- p_dm 1.0, train-time rms_norm 0.079130.
- Seed 0, 200 epochs, patience 30.
- best.pt + last.pt, with train64 PESQ and gap logged.

### CPU tests: PASS
`python tests/test_caug.py` (synthetic):
- Speed 1.0 and EQ gains 0 are identities, max |diff| 0.0.
- A 1 kHz tone resampled by 1.1 / 0.9 peaks at 1100.0 / 900.0 Hz, with resampled lengths 43637 / 53334. The fitted
  length is 48000.
- A 7.8 kHz tone sped up by 1.1 (it would alias to 7.42 kHz) leaves a -75.9 dB residual.
- Shelf gains are +-6 dB at the shelf end and half the gain at the corner.
- EQ keeps the RMS.
- Full pipeline (dynamic mixing + rms_norm + speed + EQ), 100 draws: SNR against the augmented clean is within 0.1 dB.
- With the options off, samples and RNG state are identical.

`tests/test_dmix.py` still passes.

`python scripts/r2_caug_smoke.py --out <scratch>` uses real data (logs/r2_caug_smoke.log):

| check | result |
|---|---|
| options off: RemixData vs pre-augmentation version (363c4e5), gru_v8p2z2 settings + rms_norm, 200 draws | identical samples + RNG state |
| options off: gru_v8p2z2 training loss, 2 Adam steps from weights/gru_v8p2z2.pt | bitwise equal (0.4049389887, 0.3554138992) |
| z3a / z3b, 100 real draws | length 48000, finite, SNR err 0.0000 dB, RMS rel err <= 3.2e-9; speed 43 / 56, EQ 0 / 56 |
| loader time per batch of 16 (local CPU) | z2 13 ms, z3a 81 ms, z3b 101 ms |
| z3a, 2 steps | finite (0.3689, 0.4482) |
| z3b, 2 steps | finite (0.3919, 0.3698) |
| best.pt (z3a, z3b) | loads with weights_only=True |

### Estimated Colab T4 time
- The gru_v8p2z2 Colab log (147 epochs) shows a median of 9.2 s training + 5.4 s validation/train64 per epoch,
  about 14.5 s per epoch.
- The augmentation runs on the CPU inside the training loop. Locally it adds 70 / 90 ms per batch, which is
  +6 / +7 s per epoch over 81 steps. Assuming the Colab CPU is about 2x slower, that becomes about +11 / +15 s.
- That gives about 26 s per epoch for z3a and 29 s for z3b.
- **Up to about 1.5 h for z3a and 1.7 h for z3b** (200 epochs), less with an early stop. About 3.2 h for the
  queue, plus about 5 min setup.

Evaluation plan:
1. Copy each best.pt to weights/gru_v8p2z3{a,b}.pt.
2. Run eval.py on validation and val_remix and the level probe at a clean hash. Use paired deltas vs gru_v8p2z2 +
   rms_norm.
3. Apply the same rule as before: val >= 3.5452, val_remix d > 0 with z >= 2, every probe >= 3.525.

Files: src/nrc/data/remix.py, scripts/train_dpesq.py (options line), configs/dl/gru_v8p2z3{a,b}.yaml,
tests/test_caug.py, scripts/r2_caug_smoke.py, logs/r2_caug_smoke.log, notebooks/colab_train.ipynb.

## Results 2026-11-24: gru_v8p2z3a / gru_v8p2z3b vs the submission gru_v8p2z2 + rms_norm (eval.py at 798ee67, 16-bit, --jobs 1)
Training (Colab; logs/train_log_v8p2z3{a,b}.csv; both best.pt load with weights_only=True; 0 non-finite PESQ-loss steps):

| model | best epoch / run | train-log val | val SNR | train64 | gap (z2 0.481) | curve |
|---|---|---|---|---|---|---|
| gru_v8p2z3a (+ speed) | 73 / 103, early stop | 3.5828 | 1.39 | 4.025 | 0.442 | flat (last 20 ep mean 3.5717, previous 20 3.5759) |
| gru_v8p2z3b (+ speed + EQ) | 179 / 200, epoch cap | 3.6063 | 1.28 | 4.010 | **0.404** | still rising slowly (25-epoch block means 3.557 -> 3.600) |

Clean-side augmentation narrows the gap further (0.48 -> 0.44 / 0.40). train64 stays about the same, while val rises.

Evaluation: runner logs/run_r2_caug.sh, 12 runs with 0 PESQ failures and 0 clipped samples. The first pass was killed
at the z3b -10 dB probe (host low on memory) and resumed with --resume. Tables are in logs/r2_caug_compare.txt
(scripts/r2_v6_compare.py --speakers). Paired deltas are vs gru_v8p2z2 + rms_norm.

| model | val PESQ | @0/5/10/15 dB | val SNR | val d (SE, z) | val_remix PESQ | val_remix d (SE, z) | probe -30/-20/-10/0 dB | rule |
|---|---|---|---|---|---|---|---|---|
| v8p2z2 rms_norm (submission) | 3.5452 | 2.906/3.313/3.597/3.823 | 1.35 | - | 3.4065 | - | 3.545 x4 | ref |
| v8p2z3a | 3.5828 | 2.927/3.366/3.627/3.862 | 1.39 | +0.0376 (0.0044, 8.47) | 3.4371 | +0.0306 (0.0035, 8.83) | 3.583 x4 | PASS |
| **v8p2z3b** | **3.6063** | 2.988/3.375/3.658/3.879 | 1.28 | +0.0612 (0.0052, 11.76) | **3.4628** | +0.0562 (0.0047, 11.89) | **3.606 x4** | **PASS** |

Per-SNR paired deltas at 0/5/10/15 dB:

| model | validation | val_remix |
|---|---|---|
| z3a | +0.021/+0.053/+0.030/+0.039 (z 0.7/6.8/4.8/6.4) | +0.025/+0.036/+0.030/+0.032 (z 3.6/6.5/3.6/5.0) |
| z3b | +0.082/+0.062/+0.061/+0.056 (z 3.7/5.1/7.7/6.6) | +0.059/+0.054/+0.064/+0.047 (z 6.1/6.6/6.1/5.1) |

Per difficulty, validation easy/moderate/hard is 3.511/3.705/3.833 for z3b (submission 3.449/3.642/3.775). On
val_remix it is 3.492/3.381/3.483 (3.432/3.327/3.438).

Top-5 types (delta on validation / val_remix):

| type | z3a | z3b |
|---|---|---|
| machine | +0.043 / +0.040 | +0.068 / +0.074 |
| wind | +0.044 / +0.036 | +0.059 / +0.075 |
| aircraft_cockpit | +0.014 / +0.027 | +0.070 / +0.043 |
| broadband_noise | +0.029 / +0.017 | +0.070 / +0.042 |
| alarm_siren | +0.042 / +0.035 | +0.056 / +0.058 |

Per speaker (delta on validation / val_remix):

| speaker | z3a | z3b |
|---|---|---|
| speaker007 | +0.046 / +0.038 | +0.044 / +0.049 |
| speaker015 | +0.031 / +0.026 | +0.065 / +0.059 |
| speaker016 | +0.057 / +0.054 | +0.105 / +0.102 |
| speaker037 | +0.028 / +0.030 | +0.052 / +0.051 |
| speaker040 | +0.030 / +0.008 | +0.045 / +0.025 |

Every speaker and every top-5 type gains with both models, and with z3b all of these gains are at least 0.025.
z2's alarm_siren loss on validation is reversed. The weakest speaker on val_remix is speaker040 (z3b +0.025, SE
0.006). The largest gain is for speaker016, the lowest-scoring validation speaker. This is consistent with the
speaker-mismatch hypothesis.

Caveat: as before, the checkpoint is selected on validation (train-log val = eval val), so val is mildly optimistic.
val_remix is the held-out gate: z3b gains +0.056 there at z 11.9.

Rule: val >= 3.5452, val_remix d > 0 with z >= 2, every probe >= 3.525. Both pass. **The highest val PESQ is gru_v8p2z3b
(3.6063), so it is promoted to r2-prep** (configs/dl/gru_v8p2z3b.yaml; level mode rms_norm, target_rms 0.079130).

## Third batch 2026-11-24: gru_v8p2z3c / gru_v8p2z3d (continuing the submission gru_v8p2z3b). Colab prep
Why: gru_v8p2z3b ran all 200 epochs and was still rising slowly (25-epoch block means 3.580 / 3.586 / 3.591 / 3.595 /
3.600, best epoch 179). Its gap of 0.40 is the smallest so far, and the clean-side augmentation gave the largest gain
of the batch. Two follow-ups, both init from weights/gru_v8p2z3b.pt (weights only, fresh Adam 5e-5), everything else
as gru_v8p2z3b (loss, full dynamic mixing, train-time rms_norm 0.079130), seed 1 (z3c) / 2 (z3d), 200 epochs, patience 30:

| config | speed perturbation | random EQ |
|---|---|---|
| gru_v8p2z3c (continuation) | p 0.5, U[0.9, 1.1] | p 0.5, gains U[-6, +6] dB |
| gru_v8p2z3d (stronger) | p 0.7, U[0.85, 1.15] | p 0.7, gains U[-9, +9] dB |

EQ corners are unchanged (low-shelf U[150, 500] Hz, high-shelf U[2, 5] kHz).

Filter bank: U[0.85, 1.15] has 61 factors at 1/200 steps. That needs 60 cached Kaiser FIRs (11.2 MB; [0.9, 1.1]:
40, 7.4 MB), well inside the 256-filter cache bound. The bank builds in 1.7 s on first use. No code change was
needed beyond the comment in src/nrc/data/remix.py.

Seeds: z3c uses seed 1 and z3d seed 2. z3b used seed 0, so with seed 0 z3c would replay the same sample and augmentation
sequence from epoch 1. With seeds 1 and 2 both continuations see new draws (2026-11-24 change; no other change). The 1-step CPU
tests above ran with seed 0; only the RNG seed differs.

### CPU tests: PASS
`python tests/test_caug.py`:
- 1 kHz tone x 1.15 / 0.85 peaks at 1150.0 / 850.0 Hz.
- 7.8 kHz tone x 1.15 leaves a -76.9 dB residual.
- U[0.85, 1.15] builds 60 filters.
- The pipeline at z3d settings, 100 draws:
  - SNR against the augmented clean is within 0.1 dB.
  - Speed 59/100, all in [0.855, 1.150].
  - EQ 76/100, max |gain| 9.00 dB.
- With the options off, samples and RNG state are identical.

`tests/test_dmix.py` still passes.

1-step CPU test (logs/r2_caug2_smoke.log): `train_dpesq.py --max-steps 1 --val-subset 3 --train-subset 3`

| config | step-1 loss | exit | best.pt |
|---|---|---|---|
| z3c | 0.3636, finite | 0 | loads with weights_only=True |
| z3d | 0.4262, finite | 0 | loads with weights_only=True |

The data-options line confirms each config's settings.

Notebook: notebooks/colab_train.ipynb queue [gru_v8p2z3c, gru_v8p2z3d], init check on weights/gru_v8p2z3b.pt.
- Resume: Run all again resumes from last.pt.
- Fail-loud: a failed run raises with the config name and stops the queue.

### Estimated Colab T4 time
- The z3a/z3b Colab logs show a median of 19.4 s training + 18.9 s validation/train64, about 38 s per epoch. The
  earlier estimate of 26-29 s was low.
- **About 2.1 h for z3c and about 2.2 h for z3d** (200 epochs), less with an early stop.
- About 4.3 h for the queue, plus about 5 min setup. This may need two sessions; resume handles that.

Evaluation plan: same as for z3a/z3b, against gru_v8p2z3b + rms_norm.
- Gate: val >= 3.6063, val_remix d > 0 with z >= 2, every probe >= 3.586.
- Plus per-speaker and top-5 type deltas.

## Results 2026-11-25: gru_v8p2z3c / gru_v8p2z3d vs the submission gru_v8p2z3b + rms_norm (eval.py at 9f321a6, 16-bit, --jobs 1)
Training (Colab; logs/train_log_v8p2z3{c,d}.csv; both best.pt load with weights_only=True; 0 non-finite PESQ-loss steps):

| model | best epoch / run | train-log val | val SNR | train64 | gap (z3b 0.404) | curve |
|---|---|---|---|---|---|---|
| gru_v8p2z3c (continue z3b, same aug, seed 1) | 51 / 81, early stop | 3.6136 | 1.27 | 4.023 | 0.410 | flat (25-epoch block means 3.601/3.606/3.608/3.605) |
| gru_v8p2z3d (speed p 0.7 U[0.85, 1.15], EQ p 0.7 +-9 dB, seed 2) | 139 / 169, early stop | 3.6359 | 1.24 | 4.007 | **0.371** | flattening (block means 3.606 -> 3.631; last 20 ep mean 3.6310, previous 20 3.6305) |

Continuing with the same augmentation (z3c) adds little and the gap does not narrow. Stronger clean-side augmentation
(z3d) keeps train64 flat while val rises, so the gap narrows again (0.404 -> 0.371).

Evaluation: runner logs/run_r2_cont.sh, 12 runs with 0 PESQ failures and 0 clipped samples. Tables are in
logs/r2_cont_compare.txt (scripts/r2_v6_compare.py --speakers). Paired deltas are vs gru_v8p2z3b + rms_norm.

| model | val PESQ | @0/5/10/15 dB | val SNR | val d (SE, z) | val_remix PESQ | val_remix d (SE, z) | probe -30/-20/-10/0 dB | rule |
|---|---|---|---|---|---|---|---|---|
| v8p2z3b rms_norm (submission) | 3.6063 | 2.988/3.375/3.658/3.879 | 1.28 | - | 3.4628 | - | 3.606 x4 | ref |
| v8p2z3c | 3.6136 | 2.997/3.387/3.671/3.877 | 1.27 | +0.0072 (0.0029, 2.52) | 3.4704 | +0.0077 (0.0019, 4.10) | 3.614 x4 | PASS |
| **v8p2z3d** | **3.6359** | 2.985/3.419/3.697/3.898 | 1.24 | +0.0296 (0.0063, 4.74) | **3.4868** | +0.0240 (0.0034, 7.15) | **3.636 x4** | **PASS** |

Per-SNR paired deltas at 0/5/10/15 dB:

| model | validation | val_remix |
|---|---|---|
| z3c | +0.010/+0.012/+0.013/-0.002 (z 0.8/1.6/2.9/-0.5) | +0.014/+0.010/+0.005/+0.002 (z 3.1/2.3/1.6/0.8) |
| z3d | -0.002/+0.044/+0.039/+0.019 (z -0.1/3.3/3.7/3.5) | +0.024/+0.025/+0.025/+0.022 (z 2.9/3.6/4.3/3.9) |

On validation, z3d's 0 dB delta is flat (n 15, SE 0.041). On val_remix (n 82) it is +0.024 at z 2.9.

Per difficulty, validation easy/moderate/hard is 3.547/3.722/3.857 for z3d (submission 3.511/3.705/3.833). On
val_remix it is 3.512/3.411/3.511 (3.492/3.381/3.483).

Top-5 types (delta on validation / val_remix):

| type | z3c | z3d |
|---|---|---|
| machine | +0.005 / +0.009 | +0.018 / +0.034 |
| wind | +0.009 / +0.006 | +0.045 / +0.040 |
| aircraft_cockpit | +0.009 / +0.007 | +0.000 (SE 0.040) / +0.007 (SE 0.013) |
| broadband_noise | +0.009 / -0.002 (SE 0.006) | +0.058 / +0.007 (SE 0.011) |
| alarm_siren | +0.018 / +0.012 | +0.037 / +0.013 |

Per speaker (delta on validation / val_remix):

| speaker | z3c | z3d |
|---|---|---|
| speaker007 | +0.015 / +0.010 | +0.016 / +0.017 |
| speaker015 | +0.008 / +0.005 | +0.059 / +0.021 |
| speaker016 | +0.015 / +0.019 | +0.038 / +0.061 |
| speaker037 | -0.002 (SE 0.005) / +0.001 | +0.028 / +0.014 |
| speaker040 | +0.002 / +0.006 | +0.007 (SE 0.007) / +0.012 (SE 0.007) |

z3d gains for every speaker on both sets. The largest gain is again for speaker016, the lowest-scoring validation
speaker. z3d's weakest entries (aircraft_cockpit on both sets, broadband_noise on val_remix, speaker040) are positive
but within 1-2 SE; no entry is negative. z3c's only negatives are within 1 SE.

Caveat: as before, the checkpoint is selected on validation (train-log val = eval val), so val is mildly optimistic.
val_remix is the held-out gate: z3d gains +0.024 there at z 7.2.

Rule: val >= 3.6063, val_remix d > 0 with z >= 2, every probe >= 3.586. Both pass. **The highest val PESQ is gru_v8p2z3d
(3.6359), so it is promoted to r2-prep** (configs/dl/gru_v8p2z3d.yaml; level mode rms_norm, target_rms 0.079130).

## Fourth batch 2026-11-25: gru_v8p2z3e (stronger clean-side augmentation) / gru_v8p2z4w (widened BiGRU). Colab prep
Why: starting from z3b's train64-val gap of 0.404, continuing with the same augmentation (z3c) left it at 0.410, while
stronger augmentation (z3d) narrowed it to 0.371. Stronger clean-side augmentation keeps helping; longer training alone
does not. z3e pushes the augmentation further.
z4w tests capacity: with stronger regularisation in place, a wider model may use it. The objective is PESQ-WB only.
Both continue the submission gru_v8p2z3d. Fresh Adam lr 5e-5, 0.2 cmag + 0.32 torch-pesq, full dynamic mixing,
train-time rms_norm 0.079130, 200 epochs, patience 30, best.pt + last.pt, train64 PESQ and gap logged.

| config | init | model | speed perturbation | random EQ | seed |
|---|---|---|---|---|---|
| gru_v8p2z3e | weights/gru_v8p2z3d.pt | as z3d (H 256, 2,373,651 params) | p 0.8, U[0.80, 1.20] | p 0.8, gains U[-12, +12] dB | 3 |
| gru_v8p2z4w | weights/gru_v8p2z3d_w384.pt (z3d widened) | BiGRU H 384, 5,033,363 params | as z3d (p 0.7, U[0.85, 1.15]) | as z3d (p 0.7, +-9 dB) | 4 |

Filter bank (z3e): U[0.80, 1.20] has 81 factors at 1/200 steps, so 80 cached Kaiser FIRs (14.4 MiB, built in 0.5 s; z3d
60 filters, 10.7 MiB). This is inside the existing 256-filter cache bound, so only the comment in src/nrc/data/remix.py changed.

### Net2Net widening (Chen et al. 2016): src/nrc/models/net2net.py, scripts/widen_gru.py
For each GRU layer and direction, units 0..255 keep z3d's weights. Units 256..383 are new:
- Incoming weights (input and recurrent rows of every gate r/z/n) are random, N(0, 1e-2^2), seed 4.
- Biases are the mean of the old bias block of the same gate (b_ih and b_hh separately).
- Every weight FROM a new unit is 0. This covers GRU recurrent columns into old units, layer-1 GRU input columns from new
  units of either direction, and the output head (Linear 2H -> 257).
- The input projection (Linear 1040 -> H), whose output size depends on H, is treated the same way: new rows are random
  with the mean bias, and their columns into old layer-0 units are 0.
- The widened layout of a bidirectional output is [fwd old, fwd new, bwd old, bwd new].

`python scripts/widen_gru.py` (logs/widen_gru_v8p2z3d_w384.log): PASS

| check | result |
|---|---|
| widened checkpoint | weights/gru_v8p2z3d_w384.pt, 5,033,363 params, 19.21 MiB, loads with weights_only=True |
| weights from new units | 1,048,832, all exactly 0 |
| output vs z3d, 20 stratified validation files, inference path (rms_norm) | max abs diff 1.4e-7 (median 4.5e-8), tol 1e-5 |
| gradient, step 1 (MSE on 4 val pairs) | outgoing weights of new units 5.8e-4 (non-zero); incoming 0 (expected: their outputs are muted) |
| gradient after one Adam step (lr 5e-5) | incoming weights of new units 4.2e-5 (non-zero), so the new units start learning |

This is a zero-outgoing variant of Net2WiderNet, not its unit replication with split outgoing weights: replication
keeps copies symmetric unless noise is added, and here the new units start as fresh random features.
GRUMask.MAX_PARAMS was raised from 5M to 6M for H 384 (the round-2 budget is weights < 50 MB; z4w is 19.2 MiB). Older
configs are unaffected. Inference is config- and checkpoint-driven (load_weights builds the model from the
checkpoint's model_cfg), so there is no code path specific to one run.

### CPU tests: PASS (logs/r2_z3e_z4w_smoke.log)
`train_dpesq.py --max-steps 1 --val-subset 3 --train-subset 3 --device cpu --threads 4`

| config | data-options line | step-1 loss | CPU s/step | best.pt |
|---|---|---|---|---|
| z3e | speed p 0.8 (0.8, 1.2), eq p 0.8 gain (-12, 12) dB | 0.3656, finite | 1.82 | 9.07 MiB, loads with weights_only=True |
| z4w | speed p 0.7 (0.85, 1.15), eq p 0.7 gain (-9, 9) dB; 5,033,363 params | 0.3915, finite | 2.54 | 19.21 MiB, loads with weights_only=True |

main.py with z4w params and the 1-step H-384 best.pt on 20 validation files: processed 20, failed 0, clipped 0,
4.8 / 4.9 s (z3d 4.5 / 4.6 s), so about 0.24 s per file on the local CPU.

### Estimated Colab T4 time
The z3d Colab log shows a median of 23.5 s training + 20.5 s validation/train64 per epoch, about 44 s.
- z3e: slightly more resampling and EQ work on the CPU, **about 45 s per epoch, up to about 2.5 h** (200 epochs).
- z4w: one training step takes 1.40x z3d's on the CPU, so **about 54 s per epoch, up to about 3.0 h**.
- About 5.5 h for the queue, plus about 5 min setup. Expect two sessions; resume handles that. last.pt for z4w is about 58 MB.

Notebook: notebooks/colab_train.ipynb queue [gru_v8p2z3e, gru_v8p2z4w]. It checks the init weights gru_v8p2z3d.pt and
gru_v8p2z3d_w384.pt, resumes from last.pt and is fail-loud.

Evaluation plan: as before, against gru_v8p2z3d + rms_norm.
- Gate: val >= 3.6359, val_remix d > 0 with z >= 2, every probe >= 3.616.
- Also check per-speaker and top-5 type deltas, the gap vs 0.371, and CPU time of main.py on the full validation set (< 180 s).

### Results 2026-11-25: gru_v8p2z3e / gru_v8p2z4w (rms_norm) vs the submission gru_v8p2z3d + rms_norm
Training (Colab T4; logs/train_log_v8p2z3e.csv, logs/train_log_v8p2z4w.csv; train-log val = eval validation). Both
checkpoints load with weights_only=True.

| run | epochs (early stop) | best epoch | best val | epoch-1 val | train64 at best | gap at best | gap first 10 / last 10 epochs |
|---|---|---|---|---|---|---|---|
| z3e | 134 (yes) | 104 | 3.6534 | 3.6386 | 3.9864 | 0.333 | 0.369 / 0.327 |
| z4w | 92 (yes) | 62 | 3.6455 | 3.6289 | 4.0017 | 0.356 | 0.374 / 0.359 |

z4w starts near z3d, as Net2Net intends (epoch 1 3.6289 vs z3d 3.6359; one Adam step on the fresh optimizer moves it).
It peaks at epoch 62 and plateaus. z3e's stronger augmentation again pulls train64 down (4.003 -> 3.976) while val
rises, so the gap narrows further (z3d 0.371 -> 0.333). The wider model keeps train64 near 4.00, and its gap narrows
less (0.356).

Evaluation: runner logs/run_r2_z3e_z4w.sh, 12 runs, all rc 0, 0 PESQ failures, 0 clipped samples, clean hash 3905f80.
Tables are in logs/r2_z3e_z4w_compare.txt (scripts/r2_v6_compare.py --ref gru_v8p2z3d --speakers). The deltas are paired vs
gru_v8p2z3d + rms_norm.

| model | val PESQ | @0/5/10/15 dB | val SNR | val d (SE, z) | val_remix PESQ | val_remix d (SE, z) | probe -30/-20/-10/0 dB | main.py CPU, 164 val files | rule |
|---|---|---|---|---|---|---|---|---|---|
| v8p2z3d rms_norm (submission) | 3.6359 | 2.985/3.419/3.697/3.898 | 1.24 | - | 3.4868 | - | 3.636 x4 | 86-88 s (0.53 s/file) | ref |
| **v8p2z3e** | **3.6534** | 3.009/3.431/3.714/3.918 | 1.21 | +0.0174 (0.0034, 5.19) | 3.4961 | +0.0093 (0.0030, 3.13) | **3.653 x4** | 88-96 s (0.56 s/file) | **PASS** |
| v8p2z4w | 3.6455 | 3.005/3.426/3.708/3.905 | 1.20 | +0.0096 (0.0030, 3.25) | **3.4988** | +0.0120 (0.0031, 3.90) | 3.646 x4 | 107-120 s (0.69 s/file) | PASS |

The CPU column gives the wall time of two main.py runs each, quiet machine, local CPU; all are well under 180 s.

Per-SNR paired deltas at 0/5/10/15 dB:

| model | validation | val_remix |
|---|---|---|
| z3e | +0.024/+0.012/+0.017/+0.020 (z 1.3/1.6/3.3/4.4) | +0.021/+0.004/+0.009/+0.003 (z 3.0/0.8/2.3/0.5) |
| z4w | +0.020/+0.008/+0.011/+0.007 (z 1.2/1.5/2.1/1.6) | +0.016/+0.009/+0.012/+0.010 (z 2.3/2.0/1.6/2.7) |

Top-5 types (delta on validation / val_remix):

| type | z3e | z4w |
|---|---|---|
| machine | +0.027 / +0.008 | +0.025 / +0.003 |
| wind | +0.018 / +0.012 | +0.008 / +0.016 |
| aircraft_cockpit | +0.027 (SE 0.020) / +0.016 (SE 0.015) | +0.017 (SE 0.015) / +0.042 |
| broadband_noise | +0.003 (SE 0.010) / +0.019 | -0.018 (SE 0.013) / +0.015 |
| alarm_siren | +0.013 / +0.011 | +0.004 / +0.004 |

Per speaker (delta on validation / val_remix):

| speaker | z3e | z4w |
|---|---|---|
| speaker007 | +0.015 / +0.003 (SE 0.006) | +0.005 / +0.003 |
| speaker015 | +0.010 / -0.009 (SE 0.010) | +0.014 / +0.025 |
| speaker016 | +0.040 / +0.019 | +0.015 / -0.001 (SE 0.007) |
| speaker037 | +0.014 / +0.020 | -0.001 (SE 0.004) / +0.007 |
| speaker040 | +0.011 / +0.014 | +0.015 / +0.023 |

z3e improves on z3d for every type on both sets. Its largest gain is again for speaker016, the weakest validation
speaker. Its only negative entry (speaker015 on val_remix, -0.009) is within 1 SE. z4w's negatives (broadband_noise on
validation, speaker037 on validation, speaker016 on val_remix) are each within about 1.4 SE.

Caveat: the checkpoints are selected on validation, so val is mildly optimistic. On the held-out val_remix, z4w is
+0.0027 ahead of z3e, which is inside one paired SE (about 0.003). The rule ranks on val.

Rule: val >= 3.6359, val_remix d > 0 with z >= 2, every probe >= 3.616. Both pass. **The highest val PESQ is gru_v8p2z3e
(3.6534), so it is promoted to r2-prep** (configs/dl/gru_v8p2z3e.yaml; level mode rms_norm, target_rms 0.079130).
z4w (Net2Net H 384) stays on exp/r2-dmix and is not merged. Next: a z3e-style stronger-augmentation continuation of the
widened model would combine both effects.
