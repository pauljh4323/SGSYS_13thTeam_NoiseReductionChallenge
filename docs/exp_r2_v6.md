# exp/r2-v6 — gru_v6a (complex ratio mask) / gru_v6b (noise-aware inputs) vs gru_v5b

Date 2026-11-18. Branch exp/r2-v6 (r2-prep merged in at 4067056). Weights committed at 9d62a94; all eval rows
in results/experiments.csv carry the clean hash 9d62a94. **Decision: neither model passes the round-2
promotion rule; r2-prep (gru_v5b_off) is unchanged.**

## Models and load check
Both are warm-start fine-tunes of gru_v5b (configs/dl/gru_v6a.yaml, gru_v6b.yaml: lr 1e-4, seed 0, val every 2 epochs,
early stop after 8 val points without a gain, checkpoint = best validation PESQ). Inference configs
configs/dl/gru_v6{a,b}_off.yaml (level norm off).

| model | change | params | file | ckpt epoch | loads with weights_only=True |
|---|---|---|---|---|---|
| gru_v5b | — | 2,373,651 | 9.51 MB | 8 | yes |
| gru_v6a | crm, zero-init imag head, mask_imag_scale 0.1 | 2,505,492 | 10.03 MB | 2 | yes (no re-save) |
| gru_v6b | noise_aware: dsp-v8 log noise PSD + log DD xi inputs, zero-init projection | 2,505,491 | 10.04 MB | 10 | yes (no re-save) |

Checkpoint keys: format, model_cfg, model_state, epoch, pesq, config (tensors and primitives only).

## Train logs (logs/train_log_v6a.csv, logs/train_log_v6b.csv; val PESQ = full validation)
- v6a: best epoch 2 (2.6963), early-stopped at epoch 18 (lr 1e-4 -> 2.5e-5); last 3 val points ep 14/16/18: 2.6852, 2.6867, 2.6855. ~9 s/epoch.
- v6b: best epoch 10 (2.6968), early-stopped at epoch 26 (lr 1e-4 -> 2.5e-5); last 3 val points ep 22/24/26: 2.6881, 2.6912, 2.6942. ~120 s/epoch (on-the-fly features).
- Both curves stay within about ±0.01 of the warm start (2.6962). After the first val point neither model improved on validation, so the selected checkpoints are almost the initial v5b function.

## Results (16-bit in-memory, eval.py, hash 9d62a94)
| | gru_v5b_off | gru_v6a_off | gru_v6b_off |
|---|---|---|---|
| val PESQ | 2.6962 | 2.6963 | 2.6968 |
| val 0/5/10/15 dB | 1.953/2.410/2.702/3.083 | 1.954/2.409/2.703/3.084 | 1.956/2.409/2.704/3.083 |
| val easy/moderate/hard | 2.630/2.708/2.949 | 2.632/2.713/2.934 | 2.635/2.710/2.931 |
| val SNR (dB) | 16.81 | 16.79 | 16.85 |
| val paired delta ± SE (z) | — | +0.0001 ± 0.0035 (0.04) | +0.0007 ± 0.0037 (0.18) |
| val_remix PESQ / SNR | 2.5170 / 15.45 | 2.5191 / 15.41 | 2.5147 / 15.46 |
| val_remix 0/5/10/15 dB | 1.825/2.318/2.723/3.206 | 1.825/2.318/2.725/3.212 | 1.818/2.313/2.722/3.210 |
| val_remix easy/moderate/hard | 2.591/2.364/2.473 | 2.594/2.367/2.471 | 2.590/2.361/2.467 |
| val_remix paired delta ± SE (z) | — | +0.0020 ± 0.0017 (1.20) | -0.0023 ± 0.0024 (-0.98) |
| probe -30/-20/-10/0 dB | 2.614/2.670/2.684/2.696 | 2.611/2.669/2.684/2.696 | 2.628/2.676/2.687/2.697 |
| main.py CPU, 164 val files | 24.1 s (0.147 s/file) | 29.6 s (0.180 s/file) | 31.5 s (0.192 s/file) |

The CPU times were measured back to back in one session (logs/time_r2v6_*.log). The earlier 87-92 s figures for v5b came from a more heavily loaded machine.

Weak-5 categories, paired delta vs v5b (validation | val_remix):

| category (n val / val_remix) | v6a | v6b |
|---|---|---|
| aircraft_cockpit (14/28) | -0.020 \| +0.007 | -0.011 \| -0.011 |
| factory (5/10) | -0.013 \| +0.003 | -0.010 \| +0.006 |
| broadband_noise (10/20) | -0.013 \| -0.004 | -0.025 \| -0.012 |
| destroyer_operations (5/10) | -0.012 \| -0.020 | +0.004 \| -0.011 |
| hf_channel (4/8) | -0.021 \| -0.015 | -0.002 \| -0.008 |

Neither model helps the stationary-noise categories it was aimed at. Most of these deltas are within about 2 SE.

Phase headroom recovered by v6a: (2.6963 - 2.6962) / 0.293 = 0.0004 on validation and (2.5191 - 2.5170) / 0.261 = 0.008 on val_remix. That is under 1 % of the clean-phase oracle gain. The imaginary mask head (scale 0.1, zero-init) barely moved before the validation-selected checkpoint (epoch 2) was saved.

## Promotion check (round-2 rule)
| | val >= 2.6962 | val_remix delta > 0 and z >= 2 | probe >= v5b - 0.02 | result |
|---|---|---|---|---|
| gru_v6a_off | yes (2.6963) | no (+0.0020, z 1.20) | yes | FAIL |
| gru_v6b_off | yes (2.6968) | no (-0.0023, z -0.98) | yes | FAIL |

Neither model passes, so r2-prep, configs/submission.yaml and tag r2-ready are untouched.

## Takeaways
- Warm start with zero-init new heads at lr 1e-4 gives a model that is practically v5b. Validation-based checkpoint selection then picks an early epoch, so these runs test "v5b plus noise", not the new mechanisms.
- For crm to help, the phase path would need a larger effective learning rate or mask_imag_scale (1.0 drifted in smoke tests), a loss that rewards phase (the compressed-complex term was off), or training from scratch. Any of these needs retraining, which was out of scope here.
- v6b is about 30 % slower on CPU for no gain, and it loses on val_remix at 0 dB (-0.0074, z -2.37).

Files: scripts/r2_v6_compare.py, logs/r2_v6_compare.txt, logs/eval_gru_v6{a,b}_off_*.log, logs/time_r2v6_*.log,
logs/run_r2_v6_evals.sh (logs/ is git-ignored and stays local).
