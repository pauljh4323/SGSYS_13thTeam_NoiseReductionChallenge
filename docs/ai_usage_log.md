# AI Usage Log

| date | goal | what AI did | verification |
|---|---|---|---|
| 2026-11-05 | Scaffold repo + reproducible conda env | Created .gitignore, environment.yml/requirements.txt, skeleton (main.py stub, src/nrc, results/experiments.csv), docs, CLAUDE.md, README; built env nrc from conda-forge only (nodefaults + process-local CONDA_CHANNELS override for ToS error) | Imports OK (pesq 0.0.4); manifests train 1296 / val 164; PESQ-wb(clean, noisy) on val row 0 = 1.120; staged files checked for data/reference/wav before push |
