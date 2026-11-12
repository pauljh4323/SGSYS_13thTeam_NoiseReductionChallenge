# exp/r2-soup — uniform model soups of the gru v3b → v5b lineage (round 2)

**Decision: not adopted. r2-prep stays on gru_v5b_off (tag r2-ready unchanged).**

## Setup
- `scripts/soup.py`: uniform average (float64 accumulate, cast to float32) of all float params/buffers; non-float from the first model (none exist). It asserts the same format/model_cfg/keys/shapes/dtypes.
- Architecture: GRUMask conv2d front end with **GroupNorm**, so there is **no BatchNorm** and no running stats. BN recalibration was skipped (`bn_recalibrated: False` in the checkpoints). `feat_mean`/`feat_std` are identical across all sources.
- Sanity: a soup of {gru_v5b} alone is **bit-identical** to gru_v5b on 20 stratified validation files (max |diff| 0, PESQ diff 0).
- Source weights come from dl-main (blob hashes verified). The four soups were fixed in advance, with no greedy selection: a {v4b, v5a, v5b}, b {v5a, v5b}, c {v3b, v4b, v5a, v5b}, d {v4a, v4b, v5a, v5b}.
- Evals: `scripts/eval.py` with in-memory 16-bit output, at code hash 656a4fd (clean).

## Validation (n=164)
| config | PESQ | 0 dB | 5 dB | 10 dB | 15 dB | SNR dB | paired Δ vs v5b ± SE (z) |
|---|---|---|---|---|---|---|---|
| gru_v5b_off | 2.6962 | 1.9531 | 2.4100 | 2.7022 | 3.0834 | 16.81 | — |
| gru_soup_a_off | 2.6929 | 1.9589 | 2.4106 | 2.6963 | 3.0779 | 16.83 | −0.0032 ± 0.0014 (−2.35) |
| **gru_soup_b_off** | **2.6948** | 1.9514 | 2.4105 | 2.6998 | 3.0820 | 16.82 | −0.0013 ± 0.0004 (−3.22) |
| gru_soup_c_off | 2.6801 | 1.9489 | 2.4101 | 2.6796 | 3.0597 | 16.82 | −0.0160 ± 0.0029 (−5.57) |
| gru_soup_d_off | 2.6805 | 1.9509 | 2.4102 | 2.6792 | 3.0608 | 16.83 | −0.0156 ± 0.0028 (−5.61) |

Winner by validation: **soup_b**.

## soup_b on val_remix (n=328)
PESQ 2.5159 (0/5/10/15 dB 1.8235/2.3158/2.7212/3.2058), SNR 15.46 dB. gru_v5b scores 2.5170 / 15.45. Paired Δ = −0.0012 ± 0.0003 (z = −3.60).

## Acceptance (round-2 rule)
| gate | result |
|---|---|
| val PESQ ≥ 2.6962 | **FAIL** (2.6948) |
| val_remix paired Δ > 0, z ≥ 2 | **FAIL** (−0.0012, z −3.60) |
| probe −30/−20/−10/0 ≥ v5b − 0.02 | not measured: Claude Code stopped the job under system memory pressure; not rerun because two gates already fail |
| main.py wall time, 164 files < 180 s | not measured (same job) |

## Notes
- All soups gain slightly on output SNR (+0.01–0.02 dB) but lose PESQ, mostly at 10/15 dB. The more the soup reaches back to v3b/v4a, the larger the loss: c and d are about −0.016.
- The fine-tuning steps move the weights far enough that averaging them dilutes v5b's gains. With these checkpoints, uniform soups give no benefit.
