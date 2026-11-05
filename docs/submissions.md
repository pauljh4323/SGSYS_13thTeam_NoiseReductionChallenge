# Submissions

| date | tag | method (config) | code commit | val PESQ (16-bit, main.py outputs) | float / in-mem 16-bit | e2e |
|---|---|---|---|---|---|---|
| 2026-11-05 | sub-20261105, best-v1 | course Wiener (configs/submission.yaml) | 2595b98 | 1.5987 (0/5/10/15 dB: 1.1459/1.3788/1.5610/1.9062) | 1.5984 / 1.5987 | PASS: local + fresh clone bit-identical (164 files), 0 failures, 44.1 kHz stereo + 1.7 s file preserved; main.py 164 files in ~7 s CPU |
| 2026-11-05 | best-v2 (candidate, not yet submitted) | MMSE-LSA (configs/submission.yaml = mmse_lsa_top1: noise_est initial, nf 0.20, alpha 0.96, xi_min_db -15, gmin_db -25) | f780702 | 1.7505 (0/5/10/15 dB: 1.1728/1.4687/1.6993/2.1468) | - | PASS: local + fresh clone bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; main.py 164 files in 19.8 s (clone run 10.3 s) CPU |
| 2026-11-05 | best-v3 (candidate, not yet submitted) | BiGRU mask gru_v1 (configs/submission.yaml = configs/dl/gru_v1.yaml inference params, weights/gru_v1.pt 2.4 MB, torch CPU) | 1c8a744 | 2.3594 (0/5/10/15 dB: 1.7324/2.0606/2.3624/2.7271) | - / 2.3594 | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; fresh conda env (nrc_ci from environment.yml, conda-forge only) running the clone's main.py from another cwd: bit-identical to nrc (164/164), env removed afterwards; main.py 164 files ~36 s CPU (73-111 s under parallel load) |

## Level probe (eval.py --input-gain-db, noisy scaled before enhancement, clean unchanged; logs only)

Full validation (164 files), in-memory 16-bit, PESQ-WB:

| input gain | gru_v1 | mmse_lsa_top1 |
|---|---|---|
| 0 dB | 2.3594 | 1.7505 |
| -10 dB | 2.2981 (-0.061) | 1.7505 (0.000) |
| -20 dB | 2.0695 (-0.290) | 1.7498 (-0.001) |

gru_v1 depends on input level (training remix gain range was -6..+3 dB), but it still beats MMSE-LSA by +0.32 at -20 dB. MMSE-LSA is effectively scale-invariant. A possible follow-up is input level normalization before the GRU (not done here). Logs: `logs/probe_{submission,mmse_lsa_top1}_gain{-20,-10,0}.log`.
