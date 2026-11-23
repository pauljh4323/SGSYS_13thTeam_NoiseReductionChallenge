# exp/r2-ens — output-level mask ensembles of the differentiable-PESQ models (round 2) — CLOSED

Branch exp/r2-ens from r2-prep (tag r2-ready, d30286e). No training. Code and configs at 851d5c2; all metrics from
scripts/eval.py at 851d5c2 (16-bit, --jobs 1, clean-hash rows in results/experiments.csv). Table: logs/r2_ens_compare.txt
(scripts/r2_ens_compare.py). Batch: scripts/run_r2_ens.sh, logs/run_r2_ens.out.

## Method
`gru_ens` (src/nrc/methods/gru_mask.py, `enhance_ensemble`):
- The input is scaled exactly as in level mode rms_norm: g = target_rms / max(rms(noisy), 1e-6), with target_rms = 0.079130.
- Every member computes its mask on the same |STFT(g * noisy)| (512/128 periodic Hann).
- The masks are combined with an arithmetic mean (`combine: mean`) or a geometric mean, exp(mean(log m)) (`combine: geo`).
- The combined mask is applied to the complex STFT, followed by the ISTFT. The output stays at the normalised level and
  is scaled down as a whole if its peak exceeds 0.99.
- N = 1 uses the member's mask directly. It is bit-identical to gru_mask rms_norm on 20/20 validation files
  (scripts/ens_identity.py, logs/r2_ens_identity_n1.log).
- scripts/eval.py gained `--resume`, which writes a per-file partial CSV as each file finishes and requires --jobs 1.
  A resumed run was checked to be identical to an uninterrupted one.

Candidates were fixed in advance with equal weights; nothing was tuned. E1 = {v8p2x, v8p3} with the arithmetic mean,
E1g = the same members with the geometric mean, and E2 = {v8p2x, v8p3, v8p2} with the arithmetic mean.

## Results (reference: submission gru_v8p2x + rms_norm)
| candidate | val PESQ | @0/5/10/15 dB | val SNR | val d (SE, z) | val_remix PESQ | val_remix d (SE, z) | probe -30/-20/-10/0 | main.py CPU (164 files) | rule |
|---|---|---|---|---|---|---|---|---|---|
| submission v8p2x rms_norm | **3.4867** | 2.865/3.261/3.536/3.758 | 2.24 | - | **3.3453** | - | 3.487 x4 | 40.3 s (0.25 s/file) | ref |
| E1 mean {v8p2x, v8p3} | 3.4672 | 2.849/3.260/3.511/3.731 | 2.63 | -0.0194 (0.0060, -3.26) | 3.3184 | -0.0269 (0.0056, -4.76) | 3.467 x4 | 71.9 s (0.44) | FAIL |
| E1g geo {v8p2x, v8p3} | 3.4760 | 2.852/3.264/3.520/3.744 | 2.53 | -0.0106 (0.0039, -2.73) | 3.3300 | -0.0153 (0.0037, -4.10) | 3.476 x4 | 75.7 s (0.46) | FAIL |
| E2 mean {v8p2x, v8p3, v8p2} | 3.4478 | 2.831/3.238/3.494/3.710 | 2.86 | -0.0388 (0.0075, -5.20) | 3.2937 | -0.0516 (0.0071, -7.28) | 3.448 x4 | 111.2 s (0.68) | FAIL |

- No run had a PESQ failure or a clipped sample. All four timings come from the same batch.
- Per-SNR validation deltas are negative at 0, 10 and 15 dB for every ensemble. Only E1g at 5 dB is positive
  (+0.003, z 0.35).
- On val_remix, every SNR bin is negative for every ensemble.
- The rule was val >= 3.4867, a val_remix paired delta > 0 with z >= 2, and every probe level >= 3.467. All three
  candidates fail the val and val_remix parts; E2 also fails the probe.

## Decision: closed, submission unchanged (gru_v8p2x + rms_norm; r2-prep untouched)
Mask averaging dilutes the PESQ-specific behaviour of each member. Each member's sigmoid mask is the product of its own
differentiable-PESQ fine-tune. Averaging pulls v8p2x's mask toward the weaker, less aggressive members, so the gain
shrinks in proportion to their weight (E2 < E1 < E1g < v8p2x). The geometric mean loses less, since it is dominated by
the smaller mask and so keeps more suppression. Even so, it stays significantly below v8p2x on both held-out sets. With
equal weights and members weaker than v8p2x, the ensemble cannot beat its best member. It would also cost 1.8 to 2.8 x
the CPU time. Not pursued further. Tuned weights would mean selecting on held-out data, and they approach weight 1 on
v8p2x anyway.
