# exp/dsp-g: staged paired sweeps on top of dsp-v4 (2026-11-11) — not promoted

Protocol: train-400 subset (seed 0), each stage from the values adopted so far; paired per-file PESQ vs the stage
reference; keep a value only if d >= 0.002 and d >= 2*SE (SE = std/sqrt(n)). Code/config hash 22ab5b0.

| stage | grid | result (d ± SE, z) | runner-up / note |
|---|---|---|---|
| G1 | highpass_hz {100,120,150,200} | kept 100 | best 150: +0.0078 ± 0.0039 (z 1.97, < 2); 150 cuts train-400 SI-SDR 11.35 -> 6.44 dB |
| G2 | hop_div {4,8} x window {hann,hamming,blackman} | kept 4 / hann | best 4/blackman −0.0003 ± 0.0020; hop_div 8 ≈ −0.02 |
| G3 | noise_freq_smooth_bins {0,1,2,4} x {mean,median} | kept 0 | best 1/median −0.0045 ± 0.0007 |
| G4 | noise_select_bands {1,2,4,8} | kept 1 | 2 bands −0.0240 ± 0.0090 |
| G5 | hrnr_nonlinearity {halfwave,fullwave} x hrnr_rho {0.3,0.5} | **adopt rho 0.3 (halfwave)**: +0.0032 ± 0.0011 (z 2.87) | runner-up = reference (rho 0.5); 0.3 is the lower edge of the grid (not extended) |
| G6 | noise_bias {1.0,1.25,1.5} x noise_fraction {0.05,0.10,0.15} | kept 1.25 / 0.10 (interior) | best 1.5/0.10 −0.0006 ± 0.0018 |

Finalist configs/dsp_g.yaml = dsp-v4 + hrnr_rho 0.3.

| check | dsp_g | dsp-v4 | criterion | pass |
|---|---|---|---|---|
| full train paired d (n=1296) | +0.0032 ± 0.0009 (z 3.67); 1.9088 vs 1.9056 | — | d >= 0.005 and >= 3·SE | **no** (d) |
| val PESQ (0/5/10/15 dB) | 1.9570 (1.290/1.694/1.939/2.332); paired +0.0072 ± 0.0017 | 1.9497 (1.283/1.683/1.934/2.325) | >= 1.9547 | yes |
| val SNR | 12.69 dB | 12.72 dB | report | — |
| val_remix PESQ / SNR | 1.8500 / 11.02 dB; paired +0.0051 ± 0.0011 | 1.8449 / 11.03 dB | >= 1.8449 | yes |
| level probe −30/−20/−10/0 dB | 1.9519/1.9578/1.9572/1.9570 | 1.9459/1.9509/1.9500/1.9497 | >= v4 − 0.02 | yes |
| main.py CPU (validation, 164 files) | 9.3 s | 9.9 s | < 180 s | yes |

Decision: acceptance failed on the full-train delta (0.0032 < 0.005); main and submission.yaml unchanged (dsp-v4).
