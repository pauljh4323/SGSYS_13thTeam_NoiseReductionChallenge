# Speech Noise Reduction Challenge 1 — 13조

SWE2011 Signals and Systems project, **Challenge 1 ("내 목소리를 구해줘!"): traditional signal processing only**.
Single-channel speech enhancement scored by wideband PESQ (PESQ-WB, 16 kHz); SNR is the auxiliary metric.

- Team: **13조**
- Submitted algorithm: classical DSP only (no machine learning, no trained models or learned weights).
- Validation results of the submitted version: [docs/validation_results.md](docs/validation_results.md)

## Submissions

| round | date | tag | commit | method | val PESQ | val SNR (dB) |
|---|---|---|---|---|---|---|
| 1 (first) | pending | dsp-v1 | af75105 | MMSE-LSA (`configs/submission.yaml`) | 1.7505 | 13.56 |

Validation = official validation split (164 files), scored on the 16-bit WAVs written by `main.py`
(`pesq==0.0.4`, wideband, 16 kHz). The commit column is the code commit of the submitted algorithm; tag `dsp-v1`
points to the merge on `main` with identical `main.py`, `src/` and `configs/`. Round history: [docs/submissions.md](docs/submissions.md).

## Method (submitted)

MMSE log-spectral amplitude estimator (Ephraim & Malah, 1985) with the decision-directed a priori SNR
(Ephraim & Malah, 1984), `src/nrc/methods/mmse_lsa.py`:

1. STFT: 512-point Hann window, hop 128 (32 ms / 8 ms at 16 kHz).
2. Noise PSD: mean power of the 20 % lowest-energy frames of the file (`noise_fraction: 0.20`),
   smoothed over 3 frequency bins.
3. A posteriori SNR gamma = |Y|^2 / noise PSD; a priori SNR by decision-directed smoothing
   (`alpha: 0.96`), floored at `xi_min_db: -15`.
4. LSA gain G = xi / (1 + xi) * exp(0.5 * E1(v)), v = xi * gamma / (1 + xi), clipped to [`gmin_db: -25` dB, 0 dB].
5. Enhanced magnitude = G * |Y| with the noisy phase, inverse STFT to the original length.

Parameters were chosen by grid search (`configs/sweeps/`, `scripts/sweep.py`); see `configs/mmse_lsa_top1.yaml`.
Validation PESQ 1.7505 vs 1.5987 for the course Wiener baseline and 1.3947 for the unprocessed noisy input.
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
