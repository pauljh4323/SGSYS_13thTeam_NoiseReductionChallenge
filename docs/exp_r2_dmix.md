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
