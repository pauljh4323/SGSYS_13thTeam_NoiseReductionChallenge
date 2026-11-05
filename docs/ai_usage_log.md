# AI Usage Log

| date | goal | what AI did | verification |
|---|---|---|---|
| 2026-11-05 | Scaffold repo + reproducible conda env | Created .gitignore, environment.yml/requirements.txt, skeleton (main.py stub, src/nrc, results/experiments.csv), docs, CLAUDE.md, README; built env nrc from conda-forge only (nodefaults + process-local CONDA_CHANNELS override for ToS error) | Imports OK (pesq 0.0.4); manifests train 1296 / val 164; PESQ-wb(clean, noisy) on val row 0 = 1.120; staged files checked for data/reference/wav before push |
| 2026-11-05 | Port course Wiener baseline + eval pipeline; reproduce course val numbers | Extracted notebook code cells (logs/), ported io/stft/Wiener/course SNR+PESQ into src/nrc, added method registry, configs, scripts/enhance.py, eval.py (PESQ-wb, SI-SDR, STOI, SNR; --jobs, --subset, --from-outputs), selfcheck.py | selfcheck OK (recon err ~1e-16, lengths 48000/47999); full val (164): noisy PESQ 1.3947, Wiener PESQ 1.5984, Wiener SNR 13.12 dB, 0 PESQ failures, matches course CSV |
