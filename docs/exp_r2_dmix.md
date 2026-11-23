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
