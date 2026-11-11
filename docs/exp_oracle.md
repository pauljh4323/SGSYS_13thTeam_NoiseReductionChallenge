# exp/oracle: where dsp-v6 loses PESQ — oracle (clean-informed) diagnostics (2026-11-11)

**Analysis only.** `scripts/oracle_eval.py` reads clean audio and the true noise `n = noisy - clean` to build
oracle variants. It is not a method, is not in `src/nrc/methods` or the registry, is not imported by `main.py`,
and nothing here goes into the submission path. Results are in `logs/oracle_*` (git-ignored), not in
`results/experiments.csv`.

Setup: configs/dsp_h.yaml (dsp-v6) everywhere except the oracle part; same STFT (Hann 1024 / hop 256), same
DD -> TSNR -> HRNR LSA gain chain, same gmin/xi_min, same 100 Hz zero-phase high-pass; outputs quantized to
16-bit PCM; pesq 0.0.4 `wb`; course SNR. Script hash ef94042. Splits: train-400 (stratified, seed 0) and full
validation (n=164). `--jobs 4`, ~2.5 min / ~1.5 min.

| Variant | Definition |
|---|---|
| O0 | dsp-v6 as is (registered method) |
| O1_b1 / O1_b2 | oracle stationary noise PSD = per-bin mean of \|N(k,l)\|^2 over the file; bias 1.0 / dsp-v6 two-band 1.25-1.5 |
| O2_b1 / O2_b2 | oracle time-varying noise PSD = \|N(k,l)\|^2 averaged over +-2 frames; bias 1.0 / two-band |
| O3 | dsp-v6 gain x \|Y\| with the clean STFT phase |
| O4 | ideal Wiener mask \|S\|^2/(\|S\|^2+\|N\|^2) on the noisy STFT (noisy phase), then the high-pass |
| O5_b1 / O5_b2 | O2_b1 / O2_b2 gain with the clean STFT phase |

**O0 reproduction:** validation O0 = 1.9637 (0/5/10/15 dB 1.2960/1.6994/1.9486/2.3368), SNR 12.70 dB = the
eval.py dsp_h row (74133b6). Train-400 O0 per-file PESQ and SNR equal `outputs/dsp_h/eval_train.csv` exactly
(max |diff| 0). The script's own re-implementation of the gain chain (used for O1-O5) reproduces the registered
method's waveform bit-exactly on every file (max |diff| 0.0).

## Validation (n=164)

| Variant | PESQ | 0 dB | 5 dB | 10 dB | 15 dB | easy (99) | moderate (41) | hard (24) | SNR dB | paired d vs O0 +- SE |
|---|---|---|---|---|---|---|---|---|---|---|
| O0 | 1.9637 | 1.2960 | 1.6994 | 1.9486 | 2.3368 | 2.0100 | 1.9892 | 1.7293 | 12.70 | — |
| O1_b1 | 2.1733 | 1.4245 | 1.9372 | 2.1402 | 2.5666 | 2.1859 | 2.2392 | 2.0087 | 13.88 | +0.2096 +- 0.0171 |
| O1_b2 | 2.2342 | 1.4643 | 1.9732 | 2.2069 | 2.6442 | 2.2223 | 2.3234 | 2.1306 | 13.87 | +0.2704 +- 0.0188 |
| O2_b1 | 2.6704 | 1.7215 | 2.2625 | 2.7077 | 3.1646 | 2.4741 | 2.8500 | 3.1732 | 14.81 | +0.7066 +- 0.0397 |
| O2_b2 | 2.6099 | 1.6698 | 2.1909 | 2.6436 | 3.1130 | 2.4076 | 2.7963 | 3.1263 | 14.57 | +0.6462 +- 0.0399 |
| O3 | 2.2497 | 1.4614 | 1.9371 | 2.2444 | 2.6782 | 2.3054 | 2.2805 | 1.9675 | 14.11 | +0.2860 +- 0.0117 |
| O4 | 3.4592 | 2.8320 | 3.2112 | 3.4909 | 3.7644 | 3.3254 | 3.5346 | 3.8822 | 15.99 | +1.4955 +- 0.0443 |
| O5_b1 | 3.1044 | 2.0447 | 2.6869 | 3.1718 | 3.6054 | 2.8954 | 3.3334 | 3.5757 | 16.16 | +1.1407 +- 0.0412 |
| O5_b2 | 2.9998 | 1.9460 | 2.5655 | 3.0648 | 3.5131 | 2.7845 | 3.2352 | 3.4861 | 15.69 | +1.0361 +- 0.0414 |

## Train-400 (seed 0)

| Variant | PESQ | 0 dB (36) | 5 dB (92) | 10 dB (133) | 15 dB (139) | easy (240) | moderate (100) | hard (60) | SNR dB | paired d vs O0 +- SE |
|---|---|---|---|---|---|---|---|---|---|---|
| O0 | 1.9141 | 1.3389 | 1.6293 | 1.8880 | 2.2766 | 1.9111 | 2.0358 | 1.7231 | 11.66 | — |
| O1_b1 | 2.1576 | 1.5055 | 1.8637 | 2.1613 | 2.5175 | 2.1627 | 2.2875 | 1.9208 | 12.64 | +0.2435 +- 0.0157 |
| O1_b2 | 2.2103 | 1.5113 | 1.8969 | 2.2103 | 2.5988 | 2.1949 | 2.3576 | 2.0264 | 12.60 | +0.2962 +- 0.0160 |
| O2_b1 | 2.6152 | 1.7005 | 2.1917 | 2.6732 | 3.0770 | 2.4297 | 2.8279 | 3.0030 | 13.50 | +0.7011 +- 0.0267 |
| O2_b2 | 2.5626 | 1.6319 | 2.1213 | 2.6176 | 3.0430 | 2.3663 | 2.7838 | 2.9789 | 13.29 | +0.6485 +- 0.0269 |
| O3 | 2.1742 | 1.4926 | 1.8488 | 2.1491 | 2.5901 | 2.1708 | 2.3121 | 1.9580 | 12.88 | +0.2601 +- 0.0067 |
| O4 | 3.3905 | 2.6633 | 3.1064 | 3.4478 | 3.7119 | 3.2558 | 3.5174 | 3.7178 | 14.49 | +1.4764 +- 0.0275 |
| O5_b1 | 3.0113 | 2.0015 | 2.5902 | 3.0912 | 3.4751 | 2.8143 | 3.2551 | 3.3928 | 14.59 | +1.0972 +- 0.0274 |
| O5_b2 | 2.9189 | 1.8879 | 2.4688 | 2.9904 | 3.4155 | 2.7042 | 3.1725 | 3.3553 | 14.20 | +1.0048 +- 0.0277 |

## Attributed gaps (paired PESQ, mean +- SE)

| Gap | Validation | Train-400 |
|---|---|---|
| (a) stationary noise estimation, O1_b1 - O0 (O1_b2 - O0) | +0.210 +- 0.017 (+0.270 +- 0.019) | +0.244 +- 0.016 (+0.296 +- 0.016) |
| (b) noise non-stationarity, O2_b1 - O1_b1 (O2_b2 - O1_b2) | +0.497 +- 0.036 (+0.376 +- 0.032) | +0.458 +- 0.023 (+0.352 +- 0.022) |
| (c) phase, O3 - O0 | +0.286 +- 0.012 | +0.260 +- 0.007 |
| phase on top of the O2 magnitude, O5_b1 - O2_b1 | +0.434 +- 0.012 | +0.396 +- 0.008 |
| magnitude-only ceiling O4 (abs; d vs O0) | 3.459 (+1.496) | 3.391 (+1.476) |

## Top-5 noise categories by gain over O0 (validation; small n, indicative only)

| O1_b1 - O0 | O1_b2 - O0 | O2_b1 - O0 | O2_b2 - O0 |
|---|---|---|---|
| animal (2) +0.516 | animal (2) +0.864 | bell (3) +1.956 | bell (3) +1.919 |
| vehicle_interior (3) +0.509 | door_moving (1) +0.615 | click (1) +1.853 | click (1) +1.862 |
| door_moving (1) +0.439 | vehicle_interior (3) +0.546 | tooth_brushing (1) +1.743 | animal (2) +1.708 |
| alarm_siren (23) +0.413 | bell (3) +0.527 | animal (2) +1.734 | tooth_brushing (1) +1.704 |
| phone_dialing (4) +0.365 | alarm_siren (23) +0.464 | clap (1) +1.596 | clap (1) +1.596 |

Train-400 agrees on the pattern: O1 top = phone_dialing (11) +0.67, vehicle_interior (9) +0.60, alarm_siren (57)
+0.57; O2 top = bell (3) +1.81, click (1) +1.57, animal (14) +1.40, footsteps (3) +1.38, clap (2) +1.19.

## Interpretation

- **Non-stationarity is the largest single loss (b ~ +0.4-0.5).** The gain concentrates in impulsive/transient
  noise (bell, click, clap, footsteps, animal, tooth_brushing) and in the `hard` difficulty (O0 1.73 -> O2 3.17
  on validation), which the stationary lowest-energy-frame PSD cannot represent at all. Caveat: O2 smooths only
  +-2 frames of the true noise periodogram, so it also knows the noise realization; it is an upper bound that no
  blind tracker can reach. Realistic target: the fraction recoverable by non-causal tracking (file-level offline
  processing is allowed) — e.g. minimum-statistics / IMCRA / SPP-based tracking, possibly combined with the
  stationary floor (max or blend), and transient detection for impulsive categories.
- **Stationary PSD estimation still costs ~0.2-0.3.** Even for stationary-ish noise (alarm_siren n=23/57,
  vehicle_interior, phone_dialing) the true mean PSD beats the lowest-10%-frames estimate. The lowest-energy
  frames likely under-estimate the mean and miss tonal/sweeping components (not measured here; the PSD
  estimate itself was not compared), so a whole-file robust mean estimate
  (speech-presence-weighted average rather than a minimum-type selection) is the cheapest lever.
- **Over-subtraction compensates for estimation error, not non-stationarity.** With the true stationary PSD the
  dsp-v6 two-band bias still helps (+0.06), but with the true time-varying PSD it hurts (-0.06). Re-tune the bias
  together with any new tracker; a better tracker will likely want a smaller bias.
- **Phase (c ~ +0.26-0.29) is real but hard to harvest classically**, and it grows once the magnitude is better
  (+0.43 on top of O2). Clean phase is unavailable; classical phase reconstruction/enhancement would recover
  only part of it. Lower priority than noise tracking.
- **Ceiling.** The magnitude-only ideal Wiener mask reaches 3.46 (val), +1.50 over dsp-v6; O5 (oracle noise +
  clean phase) at 3.10 is still below it, so the gain rule itself (DD smoothing, LSA, -30 dB floor) also leaves
  ~0.35+ even with perfect noise knowledge. Gaps are not additive; each is measured against its own baseline.
- **Next classical direction (recommendation):** (1) non-causal noise tracking for non-stationary noise,
  evaluated on the hard/transient categories first, with the bias re-tuned jointly; (2) a less biased whole-file
  stationary estimate as a fallback/floor; (3) phase only after (1)-(2).
