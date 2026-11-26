# Round 2 generalisation suite (branch exp/r2-gen)

**Status: v2 (final), 2026-11-26.** All 8 models are scored on all 4 sets (32/32).
Tables come from `python scripts/gen/report_suite.py` (it reads the eval.py per-file CSVs; full output in
logs/gen_report_tables.md). Metrics come from scripts/eval.py (pesq 0.0.4 WB, 16 kHz) and rows are in results/experiments.csv.
No training was done. main.py and configs/submission.yaml are unchanged.

## Why
val and val_remix reuse the train noise recordings (NOISEX-92 + Hu Nonspeech), so they don't measure noise
generalisation. This suite mixes the 164 validation clean files with **external noise corpora that no model of ours has seen**,
and keeps separate dev (selection) and test splits. Only the test sets are scored here. The dev sets are reserved.

## Data and sets
| corpus | licence | files / hours | split unit | train_pool / gen_dev / gen_test files |
|---|---|---|---|---|
| DEMAND (ch01, 16 kHz) | CC BY-SA 3.0 | 17 / 1.42 h | environment | 12 / 2 (DKITCHEN, PSTATION) / 3 (TBUS, TCAR, TMETRO) |
| ESC-50 (44.1 -> 16 kHz) | CC BY-NC 3.0 | 2000 / 2.78 h | fold, freesound source | 1359 / 198 / 389 |
| MUSAN noise | CC BY 4.0 (verify) | 930 / 6.23 h | file | 629 / 92 / 167 |

Details and SHA-256 hashes: docs/external_data.md. Split lists: splits/gen/ (deterministic, and the asserts check they are disjoint).
- **Overlap check** (Haitsma–Kalker fingerprints, BER < 0.35 over ≥ 2 s): **0 matches** between the 1460 challenge noise
  estimates (train+val noisy − clean) and any corpus file. The closest match had BER 0.454.
  Corpus-to-corpus matched 149 windows (ESC-50 ↔ MUSAN, both from freesound, plus ESC-50 near-duplicates).
  All 96 files involved are excluded from every split.
  Caveat: the positive control matched only 24/40 degraded copies of a file to itself, so the check is a lower bound on overlap.
- **G_unseen_test** has 328 files: 2 per clean file, corpora 110/109/109, SNR −5/0/5/10 dB with 82 each.
  **G_multi_test** has 164 files: two corpora summed at equal power, 0 dB total.
  **G_low** has 328 files: each file's own validation noise at −5 and −10 dB. This is seen noise *type* at unseen SNR.
  Each set's max SNR error is ≤ 0.003 dB, and peak-guarded mixes are scaled jointly with their clean file.
- DEMAND gen_test is only 3 transport environments, which are low-frequency and stationary, so DEMAND is the easy corpus here.
  It is reported separately below.
- **Not unseen for the pretrained models.** MP-SENet and CMGAN were trained on VoiceBank-DEMAND, and DFN3 on DNS data,
  which includes freesound and DEMAND. Read their DEMAND/ESC-50/MUSAN numbers as in-domain-ish.
- The main.py length check passed: 1 s and 10 s inputs kept fs, channels and length.

## Results: PESQ / SNR (dB)
| model | val | G_unseen_test | G_multi_test | G_low |
|---|---|---|---|---|
| noisy input | 1.395 / 9.63 | 1.420 / 2.50 | 1.123 / 0.00 | 1.099 / −7.50 |
| dsp_v8 | 2.021 / 12.81 | 1.778 / 8.98 | 1.224 / 4.80 | 1.154 / 1.05 |
| gru_v5b_off | 2.696 / 16.81 | 1.914 / 12.34 | 1.339 / 7.18 | 1.465 / 6.99 |
| gru_v8p2z3d | 3.636 / 1.24 | 2.832 / 1.09 | 2.226 / 0.88 | 2.307 / 0.44 |
| **gru_v8p2z3e** (submitted) | **3.653** / 1.20 | **2.835** / 1.01 | **2.240** / 0.78 | **2.311** / 0.39 |
| gru_v8p2z4w | 3.646 / 1.20 | 2.820 / 1.05 | 2.218 / 0.86 | 2.311 / 0.43 |
| ext_dfn3 | 2.102 / 11.20 | 2.083 / 11.18 | 1.644 / 9.00 | 1.302 / 4.91 |
| ext_mpsenet | 1.791 / 11.02 | 1.688 / 8.49 | 1.244 / 5.00 | 1.150 / 0.87 |
| ext_cmgan | 2.046 / 14.22 | 1.814 / 11.40 | 1.388 / 8.18 | 1.229 / 3.46 |

There were 0 PESQ failures. **5 of 8 G_low evals (dsp_v8, gru_v5b_off, gru_v8p2z4w, ext_mpsenet, ext_cmgan) exited silently
after 86 rows**, at G_low file 87 (`speaker015_utt01_seg0013_gl0`, N071 wind, −5 dB), with no traceback.
`eval.py --resume` kept the 86 rows and scored the rest without error on the next run every time. The cause is not identified:
it could be a native crash in pesq or soundfile on that file.

**Resolved (2026-11-26): native pesq crash on the *noisy* reference score, not on any model output.**
The suite log shows `Segmentation fault` (Windows 0xC0000005) for all 5 exits. The repro used pesq 0.0.4 in env nrc
(Python 3.12), with fresh processes and the signals as eval.py scores them (int16-quantised).
- `pesq(16000, clean, y, 'wb')` alone, 20 fresh processes per signal plus 20 calls in one process:
  - All 8 enhanced outputs: 0 crashes, and each gives the same score every time
    (dsp_v8 1.1496, gru_v5b_off 1.6286, z3d 2.1896, z3e 2.1253, z4w 2.1154, dfn3 1.3614, mpsenet 1.6384, cmgan 1.5708).
  - `noisy`: crashed in 1 of 20 fresh processes and in the 20-calls-in-one-process run. Its score changes from run to
    run (1.0548–1.0666), which suggests pesq is reading uninitialised memory.
- eval.py `evaluate_row` on this file, 3 × 8 models in fresh processes: 7 of 24 exited with 0xC0000005 and no stderr.
  This is the original silent exit. The crashes do not follow the model (round 1: cmgan; rounds 2 and 3: dsp_v8, dfn3
  and mpsenet), and `pesq_in` varies run to run (1.055–1.074). So the crash is in
  `pesq_in = pesq_wb(clean, noisy)`. `--resume` worked because each retry is a new draw.
- Minimal trigger, testing the noisy pair with 6 processes × 5 calls each:
  - The crash is on the degraded side only. With noisy as the reference instead, it never crashed (0/6).
  - Level does not matter: scaling the noisy signal by 0.1, 0.5 or 2 still crashes, and so does adding 1-LSB dither.
  - It needs the full 3 s. Crops to the first, middle or last 2 s are deterministic.
  - It needs broadband noise dominating the speech: wind noise alone (5/6) or high-passed at 300 Hz (4/6) still
    crash. Low-passed at 300 Hz, or clean + 0.3 × noise, is deterministic.
- Scan, 4 calls per pair in a fresh process, flagging a crash or a score that changes:
  - validation: noisy, dsp_v8 and z3e (= `configs/submission.yaml`), 0/164 flagged each.
  - G_low: noisy 2/328 flagged, both from this utterance and noise (`_gl0` at −5 dB and `_gl1` at −10 dB).
    z3e 0/328.
- **Hidden-test risk: low.** The TA scores clean vs our output. In every scan, none of our outputs triggered it,
  including at −10 dB. Even the raw noisy validation inputs (0–15 dB) never did.
  It would take an output that is almost all broadband noise, and z3e removes most of the noise on exactly these files.
  **No main.py change.**
- Optional hardening for our tooling only: run eval.py's `pesq_in` (or both PESQ calls) in a child process, retry once
  on a native crash, and record any remaining failure as a PESQ failure.
  Scripts: the session scratchpad (prep/isolation/variants/scan/evalpath), not committed.

### Paired PESQ delta vs gru_v5b_off (mean ± 95 % CI, z)
| model | val | G_unseen_test | G_multi_test | G_low |
|---|---|---|---|---|
| dsp_v8 | −0.675 ± 0.092 (−14.4) | −0.136 ± 0.035 (−7.6) | −0.115 ± 0.033 (−6.8) | −0.311 ± 0.049 (−12.5) |
| gru_v8p2z3d | +0.940 ± 0.053 (34.5) | +0.918 ± 0.050 (35.7) | +0.887 ± 0.073 (23.9) | +0.842 ± 0.037 (45.1) |
| gru_v8p2z3e | +0.957 ± 0.053 (35.7) | +0.921 ± 0.050 (35.9) | +0.901 ± 0.070 (25.1) | +0.847 ± 0.036 (45.5) |
| gru_v8p2z4w | +0.949 ± 0.054 (34.7) | +0.906 ± 0.051 (34.8) | +0.880 ± 0.073 (23.6) | +0.846 ± 0.036 (45.6) |
| ext_dfn3 | −0.594 ± 0.078 (−14.9) | +0.169 ± 0.053 (6.2) | +0.305 ± 0.057 (10.4) | −0.162 ± 0.043 (−7.5) |
| ext_mpsenet | −0.905 ± 0.095 (−18.7) | −0.226 ± 0.062 (−7.1) | −0.095 ± 0.045 (−4.1) | −0.315 ± 0.058 (−10.7) |
| ext_cmgan | −0.650 ± 0.096 (−13.3) | −0.100 ± 0.069 (−2.8) | +0.050 ± 0.053 (1.8) | −0.235 ± 0.046 (−10.0) |

Among the v8p2 models, z3e is never beaten. Paired vs z3e: z3d −0.003 (z −0.7) / −0.014 (−1.6) / −0.005 (−1.0) on
unseen/multi/low, and z4w −0.015 (z −3.0) / −0.022 (−2.5) / −0.001 (−0.2).

### Per corpus (DEMAND separate)
| model | G_unseen DEMAND (110) | G_unseen ESC-50 (109) | G_unseen MUSAN (109) | G_multi D+E (55) | G_multi D+M (55) | G_multi E+M (54) |
|---|---|---|---|---|---|---|
| noisy input | 1.799 | 1.201 | 1.255 | 1.114 | 1.181 | 1.073 |
| dsp_v8 | 2.513 | 1.329 | 1.485 | 1.190 | 1.381 | 1.099 |
| gru_v5b_off | 2.626 | 1.492 | 1.618 | 1.330 | 1.507 | 1.176 |
| gru_v8p2z3d | 3.497 | 2.431 | 2.560 | 2.304 | 2.542 | 1.825 |
| gru_v8p2z3e | 3.511 | 2.431 | 2.557 | 2.304 | 2.564 | 1.845 |
| gru_v8p2z4w | 3.495 | 2.429 | 2.530 | 2.304 | 2.527 | 1.816 |
| ext_dfn3 | 2.567 | 1.737 | 1.939 | 1.641 | 1.866 | 1.421 |
| ext_mpsenet | 2.182 | 1.296 | 1.581 | 1.235 | 1.356 | 1.138 |
| ext_cmgan | 2.193 | 1.465 | 1.781 | 1.450 | 1.502 | 1.210 |

### Per input SNR
| model | G_unseen −5 | 0 | 5 | 10 dB | G_low −10 | −5 dB |
|---|---|---|---|---|---|---|
| dsp_v8 | 1.484 | 1.596 | 1.883 | 2.148 | 1.118 | 1.189 |
| gru_v5b_off | 1.530 | 1.748 | 2.061 | 2.317 | 1.350 | 1.579 |
| gru_v8p2z3d | 2.219 | 2.662 | 3.112 | 3.333 | 2.089 | 2.524 |
| gru_v8p2z3e | 2.228 | 2.659 | 3.111 | 3.341 | 2.091 | 2.532 |
| gru_v8p2z4w | 2.205 | 2.650 | 3.095 | 3.330 | 2.086 | 2.536 |
| ext_dfn3 | 1.636 | 1.869 | 2.259 | 2.565 | 1.230 | 1.375 |
| ext_mpsenet | 1.391 | 1.552 | 1.850 | 1.958 | 1.135 | 1.165 |
| ext_cmgan | 1.484 | 1.665 | 1.960 | 2.149 | 1.164 | 1.295 |

## Findings so far
1. **The PESQ-optimised v8p2 models do not collapse on unseen noise.** On G_unseen_test their margin over v5b is about the
   same as on val (+0.92 vs +0.96), and it is +0.88–0.90 on two-noise mixtures and +0.85 at −5/−10 dB. Every corpus
   (ESC-50 2.43, MUSAN 2.53–2.56) is far above every other model.
   The absolute PESQ drop from val to G_unseen_test (−0.82) is mostly the lower input SNR of the set (noisy 2.50 dB vs 9.63 dB).
2. **gru_v8p2z3e stays the best choice.** It is first or tied on every set. z4w (H 384) is significantly worse on
   G_unseen (−0.015, z −3.0), so the width doesn't buy generalisation.
3. **The pretrained zero-shot baselines are far behind.** DFN3 is the best of the three on every set. It beats v5b on external
   noise (+0.17 / +0.31) but is −0.75 PESQ below z3e on G_unseen, and is weak at very low SNR (G_low 1.30).
   MP-SENet and CMGAN (VoiceBank-DEMAND) are worse than DFN3 everywhere, including on DEMAND (2.18 / 2.19 vs 2.57),
   which is part of their training noise. This is consistent with a narrow training domain (VoiceBank-DEMAND: 28 speakers, 0–15 dB SNR; from memory).
   **Best fine-tuning backbone: DeepFilterNet3.** It generalises best, is about 30× cheaper on CPU (0.3–0.6 s vs 11–19 s per file),
   has a comparable size (2.1 M params) and is MIT/Apache-2.0. None of the three is a reason to change direction: our own
   v8p2 models are already +0.75 PESQ above the best of them on unseen noise.
4. The v8p2 output SNR stays about 1 dB on every set. This is the known PESQ-vs-SNR trade-off of the rms_norm/PESQ-loss
   models and is not new here. SNR is the auxiliary metric.
5. dsp_v8 (round 1) is tuned on the challenge noises: −0.68 vs v5b on val, but only −0.14 on external noise.

## Pretrained models: inference
Each pretrained model ran zero-shot in its own conda env (ext_dfn / ext_mpsenet / ext_cmgan, never in requirements*.txt)
via scripts/gen/ext_infer.py with 4 torch threads. An output that would clip in 16-bit is scaled to peak 0.99 for that whole
file (PESQ is level-invariant). The number of rescaled files per model is listed below.

| model | params | CPU s/file (val) | rescaled files (val / unseen / multi / low) | licence | checkpoint |
|---|---|---|---|---|---|
| DeepFilterNet3 | 2.14 M | 0.34 | 0 / 0 / 0 / 0 | MIT / Apache-2.0 | deepfilternet 0.5.6 default |
| MP-SENet | 2.26 M | 11.0 (17–19 on G sets) | 3 / 7 / 12 / 13 | MIT | g_best_vb @ 89932cf |
| CMGAN | 1.83 M | 12.7 (15–16.5 on G sets) | 0 / 2 / 5 / 5 | MIT | src/best_ckpt/ckpt @ 3994600 |

MP-SENet val was first written hard-clipped (382 samples, PESQ 1.7913). Rerunning the 3 affected files with rescaling gave 1.7914.
DFN3/CMGAN val metas predate the rescale field, but they had 0 would-clip samples, so 0 files were rescaled.

## Housekeeping notes
- MUSAN download: an old curl from the first session was still appending to the same file when the download was resumed.
  The 11 corrupt 1 MiB blocks were found by decompressing (zlib) and re-fetched by HTTP range from the official
  openslr mirror (openslr.trmal.net, byte-identical, same size). Then `gzip -t` (CRC32 + length) passed. Only `musan/noise/` was kept and the tarball was deleted.
- make_splits fix: 4 ESC-50 freesound sources have clips in two folds, so the split now uses each source's highest fold.
  This moves 4-209698-A-37 and 4-234879-A-6 to gen_test so that no source spans splits.
