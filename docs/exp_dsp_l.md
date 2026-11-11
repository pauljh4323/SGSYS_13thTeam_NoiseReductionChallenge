# exp/dsp-l: SPP grid extension, speech-presence prior, two-band bias re-tune (2026-11-11). Not promoted.

Starting point: docs/exp_dsp_k.md. K1's best point (3 iterations, xi_H1 10 dB) sat on both grid edges, and LSD kept
falling with more iterations. Branch from main (dsp-v7).

Protocol: train-400 subset (seed 0). Each stage starts from the values adopted so far and is compared paired, per
file, against the stage reference. A point is kept only if d >= 0.002 and d >= 2*SE. Code hash 499f3eb (option +
LSD key), finalist config 5b6fbf6.

## Method change
- `spp_prior` q (default 0.5): P1 = 1 / (1 + ((1-q)/q)(1+xi_H1) exp(-gamma xi_H1/(1+xi_H1))). This is the general
  a posteriori SPP form in Gerkmann & Hendriks 2012 (already in docs/references.md). With q = 0.5 the odds factor
  is exactly 1.0. Against main (dsp-v7), the dsp_k outputs on 20 train files are bit-identical, both with the
  default and with an explicit 0.5 (max |diff| 0). With q = 0.3, all 20 files differ.

## Stages (train-400; LSD / signed bias = pre-bias stationary PSD vs oracle mean |N|^2, 100 Hz-8 kHz)

L1, `spp_iters` × `spp_xi_h1_db`, reference dsp_k (pesq 1.9511):

| iters / xi_H1 dB | d ± SE (z) | LSD dB | bias dB |
|---|---|---|---|
| 3 / 10 (dsp_k) | 0 | 5.821 | +1.372 |
| 3 / 7.5 | −0.0044 ± 0.0010 (−4.3) | 5.816 | +0.998 |
| 3 / 5 | −0.0061 ± 0.0012 (−5.0) | 5.816 | +0.907 |
| 5 / 10 | −0.0040 ± 0.0021 (−2.0) | 5.521 | +0.682 |
| 5 / 7.5 | −0.0116 ± 0.0032 (−3.6) | 5.549 | +0.214 |
| 5 / 5 | −0.0137 ± 0.0034 (−4.1) | 5.559 | +0.114 |
| 8 / 10 | −0.0095 ± 0.0038 (−2.5) | 5.379 | +0.192 |
| 8 / 7.5 | −0.0172 ± 0.0048 (−3.6) | 5.442 | −0.344 |
| 8 / 5 | −0.0191 ± 0.0050 (−3.8) | 5.460 | −0.444 |

Result: kept 3 / 10 dB. The runner-up was 5 / 10 at −0.0040 ± 0.0021. 3 / 10 is the corner of the L1 grid, but K1
already covered the other side (iters 1–2, xi_H1 15), so the optimum is bracketed and not an edge result.

In K1, LSD ranked the points the same way as PESQ. That breaks down here. More iterations lower both the LSD and
the positive bias, yet PESQ drops. Once the bias is near 0 dB, the estimate is close to the true mean, but at the
fixed two-band bias 1.25 / 1.5 that leaves more residual noise. The positive bias of +1.4 dB appears to act as
useful over-subtraction.

L2, `spp_prior` {0.3, 0.5, 0.7}:

| prior | d ± SE (z) | LSD dB | bias dB |
|---|---|---|---|
| 0.5 | 0 | 5.821 | +1.372 |
| 0.3 | −0.0029 ± 0.0028 (−1.05) | 5.998 | +2.494 |
| 0.7 | −0.0267 ± 0.0040 (−6.7) | 5.905 | +0.102 |

Result: kept 0.5 (interior). The runner-up was 0.3.

L3, `noise_bias_low` {1.0, 1.25, 1.5} × `noise_bias_high` {1.25, 1.5, 1.75}:

Result: **adopt 1.0 / 1.75**, +0.0074 ± 0.0023 (z 3.17). The runner-up was 1.25 / 1.75 at +0.0058 ± 0.0011
(z 5.34). **Edge: low 1.0 is the grid minimum and high 1.75 is the grid maximum.** Every high = 1.25 point lost
(−0.0097 to −0.0235).

L3 adopted vs dsp_k, per difficulty (train-400): easy +0.0108 ± 0.0032 (240), moderate +0.0023 ± 0.0042 (100),
hard +0.0024 ± 0.0052 (60).

Finalist configs/dsp_l.yaml = dsp_k + noise_bias_low 1.0 / noise_bias_high 1.75.

## Acceptance (revised rule 2026-11-11)

| check | dsp_l | dsp-v7 (dsp_k) | criterion | pass |
|---|---|---|---|---|
| full train paired d (n=1296) | +0.0072 ± 0.0013 (z 5.46); 1.9512 vs 1.9440 | — | d >= 0.003 and z >= 3 | yes |
| val PESQ (0/5/10/15 dB) | 1.9915 (1.337/1.745/1.973/2.353); paired +0.0031 ± 0.0036 | 1.9884 (1.331/1.749/1.959/2.356) | >= 1.9884 | yes |
| val SNR | 12.94 dB | 12.84 dB | report | — |
| val_remix PESQ / SNR | **1.8935** / 11.29 dB; paired −0.0007 ± 0.0024 | 1.8942 / 11.20 dB | >= 1.8942 | **no** |
| level probe −30/−20/−10/0 dB | 1.9887/1.9934/1.9919/1.9915 | 1.9840/1.9900/1.9887/1.9884 | >= v7 − 0.02 | yes |
| main.py CPU (validation, 164 files) | 13.9 s | 14.5 s (same session) | < 180 s | yes |

Paired deltas vs dsp_k by group:

| group | full train | validation | val_remix |
|---|---|---|---|
| easy | +0.0115 ± 0.0018 (776) | +0.0057 ± 0.0048 (99) | +0.0021 ± 0.0034 (198) |
| moderate | −0.0034 ± 0.0026 (325) | −0.0059 ± 0.0079 (41) | **−0.0103 ± 0.0041 (82)** |
| hard | +0.0077 ± 0.0023 (195) | +0.0080 ± 0.0056 (24) | +0.0041 ± 0.0033 (48) |
| 0 dB | +0.0140 ± 0.0041 | +0.0055 ± 0.0067 | +0.0083 ± 0.0036 |
| 15 dB | +0.0082 ± 0.0023 | −0.0031 ± 0.0064 | −0.0078 ± 0.0051 |

Validation per category (n small, indicative only):
- Top gains: shower (1) +0.089, broadband_noise (10) +0.046, aircraft_cockpit (14) +0.040, military_vehicle (5)
  +0.036, hf_channel (4) +0.033.
- Top losses: vehicle_interior (3) −0.058, wind (26) −0.036, door_moving (1) −0.014, bell (3) −0.014,
  machine (34) −0.012.

The train clip count is 2, the same as dsp_k.

Decision: **not promoted.** val_remix misses the gate by 0.0007. The paired delta there is −0.0007 ± 0.0024, so
there is no measurable gain off the training distribution. The bias shift trades low-frequency attenuation (low
1.25 → 1.0) for more high-band subtraction (1.5 → 1.75). That helps easy noise and hurts moderate noise, which is
significant on val_remix (−0.010, z −2.5). main is untouched, and dsp-v7 remains the candidate.

Caveats / follow-ups:
- The L3 winner sits on two grid edges (low 1.0, high 1.75). An extension (low 0.75, high 2.0) would likely raise
  the train gain further. It would also probably widen the moderate / wind loss, which is what failed val_remix.
- L1 shows that a lower-biased SPP estimate is not usable at the current bias. A joint (iters, bias_high) search
  would be the principled test, but the ceiling looks small (L3 +0.007).
- The remaining gap to the oracle stationary ceiling (val 2.234) is not in the mean PSD level. The next direction
  is the gain rule or noise-type-adaptive bias, not further mean-PSD tuning.

Logs:
- logs/sweep_dsp_l_l{1,2,3}_paired.txt (+ _run.txt, .csv, _perfile.csv), logs/oracle_lsd_dsp_l_l{1,2}_train_sub400.txt,
  logs/perdiff_dsp_l_l3.txt
- logs/eval_dsp_l_{train,validation,val_remix}.log, logs/paired_dsp_l_vs_dsp_k_train.txt,
  logs/breakdown_dsp_l_vs_dsp_k.txt, logs/probe_dsp_l_g{-30,-20,-10,0}.log, logs/time_dsp_{l,k}.log
