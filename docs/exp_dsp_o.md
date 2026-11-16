# Experiment dsp-o: time-varying noise PSD extensions on dsp-v7 (branch exp/dsp-o)

Goal: recover part of the O1 -> O2 oracle gap (docs/exp_oracle.md) with two blind, non-causal noise-PSD
extensions on dsp-v7 (configs/dsp_k.yaml): tonal-noise track detection (`noise_tonal`) and a bidirectional
SPP-gated tracker (`noise_tv_spp`), then a two-band bias re-tune. All selection on the FULL train set (1296),
paired keep rule d >= 0.002 and >= 2 SE (scripts/sweep.py stage verdicts).

**Decision: NOT promoted** (val_remix gate fails: 1.8924 < 1.8942). main and the submission stay at dsp-v7.

## Changes from the brief (recorded as required)

1. Tonal track classification tightened (`tonal_track_is_noise`): besides the brief rule (span >= tonal_min_frames,
   >= 1 speech-absent frame in the span, detected in >= tonal_min_absent_frac of them) a noise track must be
   detected in >= `tonal_min_absent_frames` (default 8) speech-absent frames and speech-absent frames must make up
   >= `tonal_min_absent_span` (default 0.3) of its span. Reason: the brief rule marked speech-harmonic tracks that
   touch one or two speech-absent frames (first partial O1 run: prom 6 / frac 0.5 gave train PESQ 1.32-1.38 vs 1.944;
   diagnostic below).
2. Tone power for lambda: estimated only from the track's speech-absent frames (|Y|^2 smoothed over +-1 frame at the
   track bins); in speech-present frames linear interpolation between the nearest absent frames (held at the ends),
   capped at |Y|^2. |Y|^2 of speech-present frames is not used directly.
3. O1 full grid replaced by a mechanism check (team decision): 4 settings on train alarm_siren + phone_dialing only;
   the full grid runs only if one is positive there. None was, so the O1 grid was skipped.
4. Stage reference per-file PESQ for O2/O3 reused from `outputs/dsp_k/eval_train.csv` (eval.py, hash c330f12, 1296
   files, mean 1.9440) instead of re-running dsp_k: src/ is unchanged c330f12 -> dsp-v7, scoring code unchanged since,
   and the dsp-o defaults are bit-identical to dsp-v7 (20 train files, max |diff| 0.0; selfcheck). Inserted with
   scripts/sweep_inject_ref.py under sweep.py's own key.
5. O3 edge extensions went two steps for noise_bias_high (2.25 and 2.5 in one run) to save a round.

Defaults remain bit-identical to dsp-v7 (verified after every code change: 20 train files vs tag dsp-v7, selfcheck).

## Tonal tracks: diagnostic (scripts/tonal_diag.py, analysis only; clean used for labels only)

Files: stratified 20 train files + all train alarm_siren (187) and phone_dialing (43). Candidate tracks last
>= 12 frames; truth "noise" = speech energy fraction over the track cells < 0.5. absent_frac 0.7, bins 1.
Precision = truly-noise share of marked tracks, recall = marked share of truly-noise tracks. logs/tonal_diag.txt.

| group | prom dB | tracks | brief prec / rec | tight prec / rec | mean speech frac of marked (brief / tight) |
|---|---|---|---|---|---|
| sub20 | 6 | 825 | 0.43 / 0.67 | 0.73 / 0.38 | 0.58 / 0.31 |
| sub20 | 10 | 302 | 0.56 / 0.73 | 0.86 / 0.45 | 0.46 / 0.18 |
| sub20 | 14 | 143 | 0.67 / 0.73 | 0.97 / 0.50 | 0.34 / 0.04 |
| alarm_siren | 10 | 7034 | 0.76 / 0.82 | 0.93 / 0.34 | 0.26 / 0.09 |
| alarm_siren | 14 | 4521 | 0.81 / 0.83 | 0.97 / 0.39 | 0.20 / 0.05 |
| phone_dialing | 10 | 756 | 0.34 / 0.83 | 0.73 / 0.28 | 0.67 / 0.32 |
| phone_dialing | 14 | 430 | 0.55 / 0.93 | 0.97 / 0.34 | 0.48 / 0.06 |

The brief rule explains the collapse (at prom 6 more than half of the marked tracks are speech). The tightened rule
is precise (0.73-0.97) but recalls only 0.28-0.50 of the noise tracks.

## O1 mechanism check (train alarm_siren + phone_dialing, n=230, paired vs dsp-v7)

`configs/sweeps/dsp_o_o1_mech.yaml`, `sweep.py --cats alarm_siren phone_dialing`; absent frac 0.7, bins 1, tightened rule.

| prom dB | beta | PESQ | d +- SE (z) |
|---|---|---|---|
| 14 | 1.0 | 2.1162 | -0.0110 +- 0.0070 (-1.57) |
| 14 | 1.5 | 2.1026 | -0.0246 +- 0.0084 (-2.93) |
| 10 | 1.0 | 2.0909 | -0.0363 +- 0.0125 (-2.91) |
| 10 | 1.5 | 2.0767 | -0.0505 +- 0.0133 (-3.81) |

Reference dsp-v7 2.1272. Best point by category: alarm_siren -0.0133 +- 0.0086, phone_dialing -0.0009 +- 0.0009
(logs/breakdown_dsp_o_o1_mech_002.txt). None positive -> O1 grid skipped, `noise_tonal` stays off. **Closed.**
Likely reason: a raised lambda on a tone also notches the speech underneath it (the tone and speech share bins),
and the recall at usable precision is low.

## O2: bidirectional SPP tracker on dsp-v7 (full train, logs/sweep_dsp_o_o2_paired.txt)

12 points (alpha {0.8, 0.9, 0.95, 0.98} x {max, mix w 0.5, mix w 1.0}); all lose. Best alpha 0.9 / max
d = -0.0009 +- 0.0014 (z -0.64); runner-up 0.95 / max -0.0019 +- 0.0008; worst 0.8 / mix 1.0 -0.0259 +- 0.0036.
No winner, so no edge extension. Best point by difficulty: easy -0.0048 +- 0.0018, moderate +0.0047 +- 0.0031,
hard +0.0051 +- 0.0033; bell +0.049 +- 0.032, phone_dialing -0.052 +- 0.007 (logs/breakdown_dsp_o_o2_001.txt).
Not adopted.

## O3: two-band bias re-tune on dsp-v7 (full train; O3, O3b, O3c)

| low \ high | 1.25 | 1.5 | 1.75 | 2.0 | 2.25 | 2.5 |
|---|---|---|---|---|---|---|
| 0.75 | | | -0.0059 | -0.0058 | | |
| 1.0 | -0.0102 | +0.0017 | +0.0072 | **+0.0079 +- 0.0016** | +0.0056 | +0.0026 |
| 1.25 | -0.0129 | 0 (dsp-v7) | +0.0056 | +0.0069 | +0.0057 | +0.0028 |
| 1.5 | -0.0222 | -0.0084 | -0.0022 | | | |

Adopted low 1.0 / high 2.0: d = +0.0079 +- 0.0016 (z 4.98); runner-up 1.0 / 1.75 +0.0072 +- 0.0013. Both values
are interior after the extensions. By difficulty: easy +0.0113 +- 0.0022, moderate -0.0038 +- 0.0031,
hard +0.0142 +- 0.0031. Categories: phone_dialing +0.047 +- 0.008, alarm_siren +0.012 +- 0.004, footsteps +0.010,
bell +0.007, click +0.007, vehicle_interior +0.004, clap +0.002 (logs/breakdown_dsp_o_o3b_002.txt).

## PSD diagnostic (oracle_eval.py --tv-lsd-grid configs/sweeps/dsp_o_psd_diag.yaml, full train)

PSD as the gain sees it (after the two-band bias, `--post-bias`) vs the O2 oracle |N|^2 averaged over +-2 frames;
100 Hz-8 kHz x all frames. logs/oracle_tvlsd_dsp_o_psd_diag_train_postbias.txt.

| variant | LSD all | bias all | LSD alarm_siren | bias alarm_siren | LSD phone_dialing |
|---|---|---|---|---|---|
| dsp-v7 | 9.83 | +6.46 | 13.33 | +8.34 | 35.93 |
| finalist (bias 1.0 / 2.0) | 10.48 | +7.46 | 13.85 | +9.34 | 36.78 |
| tonal (prom 14, beta 1.0) | 9.85 | +6.47 | 13.33 | +8.36 | 35.93 |
| tv-spp (0.9, max) | 9.97 | +6.75 | 13.56 | +8.70 | 37.11 |

The finalist moves the PSD further from the oracle (more over-estimation above 1 kHz) and still gains PESQ: the gain
is a perceptual over-subtraction trade-off, not better tracking. Neither extension moves the PSD towards the O2
oracle; the gap to O2 is dominated by cells where the noise is far below the stationary estimate (phone_dialing:
bias +34 dB, the tones are intermittent), which an upward-only tonal floor or a speech-gated tracker cannot reach.

## Finalist configs/dsp_o.yaml (hash dabbbf1) vs dsp-v7

| | dsp-v7 | dsp_o | paired d +- SE (z) |
|---|---|---|---|
| train PESQ (1296) | 1.9440 | 1.9519 | +0.0079 +- 0.0016 (4.98) |
| val PESQ (164) | 1.9884 | 1.9923 | +0.0039 +- 0.0044 (0.89) |
| val 0/5/10/15 dB | 1.331/1.749/1.959/2.356 | 1.340/1.746/1.974/2.352 | +0.008/-0.003/+0.015/-0.004 |
| val easy/moderate/hard | 2.037/2.020/1.735 | 2.041/2.016/1.751 | +0.004/-0.005/+0.017 |
| val SNR dB | 12.84 | 12.94 | |
| val_remix PESQ (328) | 1.8942 | 1.8924 | -0.0017 +- 0.0029 (-0.60) |
| val_remix 0/5/10/15 dB | 1.280/1.607/2.064/2.630 | 1.289/1.609/2.059/2.616 | +0.010/+0.002/-0.005/-0.014 |
| val_remix SNR dB | 11.20 | 11.30 | |
| level probe -30/-20/-10/0 dB (val) | 1.9840/1.9900/1.9887/1.9884 | 1.9899/1.9943/1.9927/1.9923 | all >= dsp-v7 |
| main.py wall time (val, 164 files) | 20.0 s | 18.7 s | same session |

main.py outputs of dsp_o re-scored: val PESQ 1.9923 (identical to eval.py). O1 -> O2 gap recovered on validation:
(1.9923 - 1.9884) / (2.670 - 1.9884) = 0.6 %.

Promotion rule: full-train d >= 0.003 and z >= 3 (pass: +0.0079, z 4.98); val >= 1.9884 (pass: 1.9923);
val_remix >= 1.8942 (**fail**: 1.8924); probes >= dsp-v7 - 0.02 (pass). Not promoted; val_remix was not used
for selection. The val gain (+0.0039) is within 1 SE and val_remix loses at 10/15 dB and on moderate files,
consistent with a small, noise-type-dependent effect of the stronger high-band over-subtraction.

## Files

- Code: src/nrc/methods/mmse_lsa.py (`noise_tonal`, `noise_tv_spp`, tightened classification, tone power),
  scripts/tonal_diag.py, scripts/sweep.py (`--cats`), scripts/sweep_inject_ref.py, scripts/oracle_eval.py
  (`--tv-lsd-grid`, `--post-bias`), scripts/selfcheck.py.
- Configs: configs/dsp_o.yaml; configs/sweeps/dsp_o_{o1,o1_mech,o2,o3,o3b,o3c,psd_diag}.yaml (dsp_o_o1 = the
  original O1 grid; stopped after 3 of 36 points under the brief rule, logs/sweep_dsp_o_o1.csv).
- Logs: logs/sweep_dsp_o_*_paired.txt, logs/breakdown_dsp_o_*.txt, logs/tonal_diag.txt,
  logs/oracle_tvlsd_dsp_o_psd_diag_train_postbias.txt, logs/eval_dsp_o_*.log, logs/main_time_dsp_{k,o}.log.
