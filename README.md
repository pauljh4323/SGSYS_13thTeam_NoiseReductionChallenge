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

Conv2d front-end + BiGRU magnitude mask (`configs/submission.yaml` = inference params of `configs/dl/gru_v2b_off.yaml`, weights `weights/gru_v2b.pt`, 9.5 MB, trained from scratch on the train split only): STFT (n_fft 512, hop 128), log1p magnitude -> 2 x Conv2d (16 ch, 3x3, freq stride 2, GroupNorm, PReLU) -> 2-layer bidirectional GRU (hidden 256, 2.37M params) -> sigmoid mask on the noisy magnitude, noisy phase, inverse STFT. Trained with U(-30,+6) dB input-gain augmentation on all samples, so no inference-time level guard is used (level norm off).
Validation PESQ (wideband, 164 files, scored on main.py's 16-bit outputs): **2.5051** (0/5/10/15 dB: 1.7975/2.2128/2.5067/2.8913); with input attenuated by -10/-20/-30 dB: 2.4921/2.4791/2.4390. Previous: gru_v1 + range_boost level guard 2.3598 (`configs/dl/gru_v1_lnrb.yaml`), gru_v1 2.3594, MMSE-LSA 1.7505, course Wiener baseline 1.5987. Details in `docs/submissions.md`.

CPU runtime: ~53 s for the 164 validation files (3 s each) on a 16-thread desktop CPU (torch limited to 4 threads); no GPU used or needed.

Fallback (classical, no neural network): `python main.py --config configs/mmse_lsa_top1.yaml` (MMSE-LSA, val PESQ 1.7505).

## Setup

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate nrc
```

The environment includes PyTorch (CPU build is enough; `main.py` always runs on CPU). With pip, `pip install -r requirements.txt` pulls the default torch wheel; on Linux you can use the smaller CPU-only wheel via `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.

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

Reads every `.wav` in `input/` and writes the enhanced file with the same name to `output/`. Paths can be given explicitly (`python main.py --input /path/to/noisy --output /path/to/out`); `main.py` can be launched from any working directory: the config and the weights are resolved against the repo folder.
