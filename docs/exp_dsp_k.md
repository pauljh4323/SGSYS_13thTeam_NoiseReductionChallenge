# exp/dsp-k: SPP-weighted whole-file noise PSD, upward-only local-min correction, bias re-tune (2026-11-11) — promoted as dsp-v7

Motivation: docs/exp_oracle.md. The true mean noise PSD would give about +0.21 to +0.27 val PESQ over the dsp-v6
lowest-10%-frame estimate. Branch from main (dsp-v6) plus the exp/oracle commits (scripts/oracle_eval.py, analysis only).

Protocol: train-400 subset (seed 0), each stage from the values adopted so far; paired per-file PESQ against the stage
reference; keep only if d >= 0.002 and d >= 2*SE. Code hash b21bb64 (options + diagnostic), finalist config c330f12.
The new options are off by default. dsp_h, a quantile config and an initial config produce bit-identical outputs on
20 train files (max |diff| 0).

## Method
- `noise_est: spp_mean`: lambda_0 = the dsp-v6 initial estimate (before bias). Each iteration computes
  gamma = |Y|^2/lambda_i and the a posteriori SPP with prior 0.5 and fixed xi_H1 (Gerkmann & Hendriks 2012):
  P1 = 1/(1+(1+xi)exp(-gamma xi/(1+xi))). It then sets lambda_{i+1}(k) = sum_l (1-P1)|Y|^2 / sum_l (1-P1), falling
  back to lambda_i when the denominator is below 1e-3. The two-band bias is applied after. The whole-file iterative
  use is our own variant; its fixed point equals that of the paper's recursion lambda <- (1-P1)|Y|^2 + P1 lambda
  averaged over the file.
- `tv_noise`: P = mean of the forward and backward one-pole smoothing of |Y|^2 (tv_alpha). M = centred min of P over
  +-W frames. lambda(k,l) = max(lambda_stat(k), tv_bias M(k,l)), then the bias. This can only raise the noise
  estimate, never lower it (asserted in selfcheck).

## PSD diagnostic (train-400, oracle_eval.py --lsd-grid; pre-bias PSD vs oracle mean |N|^2, 100 Hz-8 kHz)

| estimator | LSD dB | signed bias dB | easy / moderate / hard LSD |
|---|---|---|---|
| dsp-v6 initial (lowest 10 % frames) | 8.28 | +4.56 | 7.15 / 10.29 / 9.46 |
| spp_mean 1 it / 10 dB | 6.80 | +2.92 | 5.72 / 8.38 / 8.52 |
| spp_mean 1 / 15 | 6.93 | +3.45 | 5.89 / 8.50 / 8.46 |
| spp_mean 2 / 10 | 6.16 | +1.97 | 5.10 / 7.46 / 8.21 |
| spp_mean 2 / 15 | 6.32 | +2.84 | 5.36 / 7.60 / 8.03 |
| spp_mean 3 / 10 | 5.82 | +1.37 | 4.75 / 6.99 / 8.15 |
| spp_mean 3 / 15 | 5.99 | +2.47 | 5.06 / 7.10 / 7.83 |

The dsp-v6 estimate over-estimates the mean noise PSD by about 4.6 dB, which contradicts the earlier hypothesis that
it under-estimates. The lowest-energy frames still contain speech, because speech fills most frames. SPP weighting
removes the excess step by step. In this diagnostic, LSD ranks the K1 points in the same order as PESQ.

## Stages (train-400)

| stage | grid | result (d ± SE, z) | runner-up / note |
|---|---|---|---|
| K1 | spp_iters {1,2,3} × spp_xi_h1_db {10,15}, ref dsp_h | **adopt 3 / 10 dB**: +0.0370 ± 0.0071 (z 5.20) | runner-up 2/10 +0.0359 ± 0.0061. All 6 points kept (min +0.0282). Edge: iters 3 (max) and xi 10 (min), not extended |
| K2 | tv_noise, W {8,16,32} × tv_bias {1,1.5,2}, tv_alpha 0.8, ref K1 | kept off | best W 32 / 1.0: −0.2148 ± 0.0129. Every point −0.21 to −0.50. The best point is at the large-W / small-bias corner, i.e. the corner closest to no correction |
| K3 | noise_bias_low × noise_bias_high {1,1.25,1.5}², ref K1 | kept 1.25 / 1.5 | best 1.0/1.5 +0.0022 ± 0.0020 (z 1.09). high 1.0 all −0.034 to −0.050. Edge: low 1.0, high 1.5 |

Per difficulty, train-400 paired:
- K1 adopted vs dsp_h: easy +0.0369 ± 0.0091 (240), moderate +0.0662 ± 0.0164 (100), hard −0.0114 ± 0.0116 (60).
- K2 best vs K1: easy −0.223, moderate −0.251, hard −0.124. The loss grows with SNR (0 dB −0.035, 15 dB −0.300).

The local minimum over ±0.13 to 0.5 s still lands on speech energy during sustained voicing, so the correction inflates
lambda under speech.

Finalist configs/dsp_k.yaml = dsp-v6 + noise_est spp_mean (spp_iters 3, spp_xi_h1_db 10).

## Acceptance (revised rule 2026-11-11)

| check | dsp_k | dsp-v6 (dsp_h) | criterion | pass |
|---|---|---|---|---|
| full train paired d (n=1296) | +0.0302 ± 0.0039 (z 7.70); 1.9440 vs 1.9137 | — | d >= 0.003 and z >= 3 | yes |
| val PESQ (0/5/10/15 dB) | 1.9884 (1.331/1.749/1.959/2.356); paired +0.0247 ± 0.0099 | 1.9637 (1.296/1.699/1.949/2.337) | >= 1.9637 | yes |
| val SNR | 12.84 dB | 12.70 dB | report | — |
| val_remix PESQ / SNR | 1.8942 / 11.20 dB; paired +0.0418 ± 0.0081 | 1.8524 / 11.04 dB | >= 1.8524 | yes |
| level probe −30/−20/−10/0 dB | 1.9840/1.9900/1.9887/1.9884 | 1.9589/1.9646/1.9639/1.9637 | >= v6 − 0.02 | yes |
| main.py CPU (validation, 164 files) | 13.7 s | 11.3 s (same session) | < 180 s | yes |

Full train per target SNR: 0 dB +0.053 ± 0.012, 5 dB +0.052 ± 0.008, 10 dB +0.033 ± 0.007, 15 dB +0.007 ± 0.007.

Full train per difficulty: easy +0.030 ± 0.005 (776), moderate +0.059 ± 0.008 (325), **hard −0.017 ± 0.007 (195)**.
On validation, hard is +0.006 ± 0.025 (24); on val_remix it is +0.016 ± 0.015 (48).

Train clipped samples: 2, the same as dsp_h.

Validation per category (n small, indicative only):
- Top gains: vehicle_interior (3) +0.206, destroyer_engine (5) +0.119, phone_dialing (4) +0.095, alarm_siren (23)
  +0.083, wind (26) +0.060.
- Top losses: aircraft_cockpit (14) −0.079, traffic_vehicle (9) −0.063, tooth_brushing (1) −0.059, water (5) −0.055,
  bell (3) −0.037.

Decision: all criteria pass. Merge to main, set submission.yaml = dsp_k, tag dsp-v7.

Caveats:
- The full-train loss on hard (non-stationary) noise is significant. A lower stationary floor exposes more transient
  noise. Those categories need a working time-varying tracker, which K2's local-minimum design is not.
- K1's best point sits at two grid edges, and LSD keeps falling with more iterations. A follow-up could extend
  spp_iters / lower xi_H1, with the bias re-tuned jointly.

Logs:
- logs/oracle_lsd_dsp_k_k1_train_sub400.txt, logs/sweep_dsp_k_k{1,2,3}_paired.txt, logs/perdiff_dsp_k_k{1,2_best}.txt
- logs/eval_dsp_k_{train,validation,val_remix}.log, logs/paired_dsp_k_vs_dsp_h_train.txt,
  logs/breakdown_dsp_k_vs_dsp_h.txt
- logs/probe_dsp_k_g{-30,-20,-10,0}.log, logs/time_dsp_{k,h}.log, logs/selfcheck_dsp_k.log
