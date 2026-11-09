# Speech Noise Reduction Challenge 1 — 13조

SWE2011 Signals and Systems project, **Challenge 1 ("내 목소리를 구해줘!"): traditional signal processing only**.
Single-channel speech enhancement scored by wideband PESQ (PESQ-WB, 16 kHz); SNR is the auxiliary metric.

- Team: **13조**
- Submitted algorithm: classical DSP only (no machine learning, no trained models or learned weights).
- Validation results of the submitted version: [docs/validation_results.md](docs/validation_results.md)

## Submissions

| round | date | tag | commit | method | val PESQ | val SNR (dB) |
|---|---|---|---|---|---|---|
| 1 (first) |  | dsp-v4 | db1085e | MMSE-LSA + HRNR (n_fft 1024) | 1.9497 | 12.72 |
| - (superseded, never submitted) | - | dsp-v3 | 7080f9b | MMSE-LSA + HRNR + 100 Hz high-pass, re-tuned alpha / xi_min (`configs/dsp_retune.yaml`) | 1.9352 | 12.71 |
| - (superseded, never submitted) | - | dsp-v2 | e68a073 | MMSE-LSA + HRNR + 100 Hz high-pass (`configs/dsp_tune_c.yaml`) | 1.8746 | 12.64 |
| - (superseded, never submitted) | - | dsp-v1 | af75105 | MMSE-LSA (`configs/mmse_lsa_top1.yaml`) | 1.7505 | 13.56 |

Validation = official validation split (164 files), scored on the 16-bit WAVs written by `main.py`
(`pesq==0.0.4`, wideband, 16 kHz). The commit column is the code commit of the submitted algorithm; tag `dsp-v4`
points to the commit on `main` with identical `main.py`, `src/` and `configs/`. Round history: [docs/submissions.md](docs/submissions.md).

## Method (submitted)

MMSE log-spectral amplitude estimator (Ephraim & Malah, 1985) with a two-step a priori SNR and harmonic
regeneration (HRNR, Plapous, Marro & Scalart, 2006), `src/nrc/methods/mmse_lsa.py`:

1. STFT: 1024-point Hann window, hop 256 (64 ms / 16 ms at 16 kHz).
2. Noise PSD: mean power of the 10 % lowest-energy frames of the file (`noise_fraction: 0.10`),
   smoothed over 3 frequency bins, multiplied by a noise bias of 1.25 (`noise_bias: 1.25`).
3. A posteriori SNR gamma = |Y|^2 / noise PSD; first-step a priori SNR by decision-directed smoothing
   (Ephraim & Malah, 1984; `alpha: 0.94`), floored at `xi_min_db: -25`.
4. HRNR (`xi_mode: hrnr`): TSNR second step xi = |G_dd Y|^2 / noise; the TSNR estimate is resynthesized,
   half-wave rectified to regenerate harmonics, re-analysed, and the final a priori SNR is
   xi = (0.5 |G_tsnr Y|^2 + 0.5 |S_harmo|^2) / noise.
5. LSA gain G = xi / (1 + xi) * exp(0.5 * E1(v)), v = xi * gamma / (1 + xi), clipped to [`gmin_db: -30` dB, 0 dB];
   enhanced magnitude = G * |Y| with the noisy phase, inverse STFT to the original length.
6. Output high-pass: 4th-order Butterworth at 100 Hz applied forward-backward (zero phase, `highpass_hz: 100`).

Parameters were chosen by grid-search stages on a 400-file stratified train subset only
(`configs/sweeps/dsp_tune_{a,b,c}.yaml`, then the joint re-tune `configs/sweeps/dsp_retune_d{1,2,3}.yaml` and the grid-edge extension
`configs/sweeps/dsp_edge_e{1,2}.yaml`, `scripts/sweep.py`); see `configs/dsp_edge.yaml`.
Validation PESQ 1.9497 / SNR 12.72 dB (dsp-v3 1.9352 / 12.71 dB, dsp-v2 1.8746 / 12.64 dB, dsp-v1 1.7505 / 13.56 dB, course Wiener baseline 1.5987 / 13.12 dB,
unprocessed noisy input 1.3947 / 9.63 dB). Relative to dsp-v1 the PESQ gain costs about 0.85 dB of validation SNR overall: SNR improves at
0/5 dB input SNR and drops at 10/15 dB (high-pass and noise over-estimation remove some speech energy).
References: [docs/references.md](docs/references.md).

## Install

Python 3.12, pip only (no conda needed):

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

`requirements.txt` holds only the pinned runtime dependencies of `main.py` (numpy, scipy, soundfile, pyyaml).
Evaluation and development tools (PESQ, STOI, pandas, matplotlib, ...) are in `requirements-dev.txt`;
`environment.yml` is the full local development environment (conda).

## Run

```bash
python main.py
```

Reads every `.wav` under `input/` (recursively, subfolders mirrored) and writes the enhanced file with the
**same filename** to `output/`: 16-bit PCM with the **same sample rate, channel count and number of samples**
as the input. Multi-channel input is averaged to mono for enhancement and the result is written to every channel;
non-16 kHz input is resampled to 16 kHz for enhancement and back. If a file cannot be processed it is copied
through unchanged. CPU-only, offline, ~20 s for the 164 validation files.
Paths can be given explicitly (`python main.py --input <noisy dir> --output <out dir>`); `main.py` works from any
working directory (the config is resolved against the repo folder, no absolute paths).

## Checks and evaluation (developers)

```bash
pip install -r requirements-dev.txt
python scripts/selfcheck.py
python scripts/e2e_check.py      # I/O format, fresh clone, fresh pip venv, torch-free, classical allow-list
python scripts/eval.py --config configs/submission.yaml --split validation
python scripts/report_val.py     # -> docs/validation_results.md
```

The challenge data is not in the repository (data/noise_reduction_challenge, override with `--data-root`).

## Deep-learning experiments (not part of this challenge)

Earlier BiGRU mask experiments (tags best-v3 ... best-v8) are **not eligible for Challenge 1** and are kept
only on branch [`dl-main`](../../tree/dl-main). They are not part of this submission and never merged into `main`.
