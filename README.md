# Speech Noise Reduction Challenge — Team 13

SWE2011 Signals and Systems project: single-channel speech enhancement, scored by wideband PESQ (16 kHz).

**Status: scaffold only** — no enhancement algorithm implemented yet.

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
