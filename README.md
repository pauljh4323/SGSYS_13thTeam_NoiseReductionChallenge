# Speech Noise Reduction Challenge — Team 13

SWE2011 Signals and Systems project: single-channel speech enhancement, scored by wideband PESQ (16 kHz).

## How to run (for TAs)

```bash
conda env create -f environment.yml
conda activate nrc
# put the noisy .wav files in input/
python main.py
# enhanced files (same filenames) are written to output/
```

pip alternative (Python 3.12) instead of conda: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, then `python main.py`.

`main.py` processes every `.wav` under `input/` (recursively, subfolders mirrored) and writes 16-bit PCM wavs with the same filename, sample rate and length to `output/`. Multi-channel input is averaged to mono for enhancement and the result is written to every channel (shape preserved); non-16 kHz input is resampled to 16 kHz for enhancement and back. If a file cannot be processed, it is copied through unchanged and the run continues. CPU-only, no network.

## Current method

Course Wiener filter (`configs/submission.yaml`): STFT (n_fft 512, hop 128, Hann), noise PSD from the first 20 % of frames, decision-directed a-priori SNR (smoothing 0.98), Wiener gain with floor 0.05.
Validation PESQ (wideband, 164 files, scored on main.py's 16-bit outputs): see `docs/submissions.md`.

## Setup

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate nrc
```

If conda fails with `CondaToSNonInteractiveError` (Anaconda `defaults` channels in your condarc), restrict that one command to conda-forge, e.g. in bash: `CONDA_CHANNELS=conda-forge conda env create -f environment.yml`.

pip alternative (Python 3.12):

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Reads every `.wav` in `input/` and writes the enhanced file with the same name to `output/`.
