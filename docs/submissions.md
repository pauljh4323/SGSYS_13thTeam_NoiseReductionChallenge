# Submissions

| date | tag | method (config) | code commit | val PESQ (16-bit, main.py outputs) | float / in-mem 16-bit | e2e |
|---|---|---|---|---|---|---|
| 2026-11-05 | sub-20261105, best-v1 | course Wiener (configs/submission.yaml) | 2595b98 | 1.5987 (0/5/10/15 dB: 1.1459/1.3788/1.5610/1.9062) | 1.5984 / 1.5987 | PASS: local + fresh clone bit-identical (164 files), 0 failures, 44.1 kHz stereo + 1.7 s file preserved; main.py 164 files in ~7 s CPU |
| 2026-11-05 | best-v2 (candidate, not yet submitted) | MMSE-LSA (configs/submission.yaml = mmse_lsa_top1: noise_est initial, nf 0.20, alpha 0.96, xi_min_db -15, gmin_db -25) | f780702 | 1.7505 (0/5/10/15 dB: 1.1728/1.4687/1.6993/2.1468) | - | PASS: local + fresh clone bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; main.py 164 files in 19.8 s (clone run 10.3 s) CPU |
| 2026-11-05 | best-v3 (candidate, not yet submitted) | BiGRU mask gru_v1 (configs/submission.yaml = configs/dl/gru_v1.yaml inference params, weights/gru_v1.pt 2.4 MB, torch CPU) | 1c8a744 | 2.3594 (0/5/10/15 dB: 1.7324/2.0606/2.3624/2.7271) | - / 2.3594 | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; fresh conda env (nrc_ci from environment.yml, conda-forge only) running the clone's main.py from another cwd: bit-identical to nrc (164/164), env removed afterwards; main.py 164 files ~36 s CPU (73-111 s under parallel load) |
| 2026-11-05 | best-v4 (candidate, not yet submitted) | BiGRU mask gru_v1 + range_boost input level guard (configs/submission.yaml = configs/dl/gru_v1_lnrb.yaml inference params: active level, train p5/p95 -27.88/-15.27 dBFS, boost-below target = train median -19.60; weights/gru_v1.pt unchanged) | 4f8774c | 2.3598 (0/5/10/15 dB: 1.7341/2.0614/2.3629/2.7268) | - / 2.3598 (exp row 9d325ea) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files ~37 s CPU |

## Level probe (eval.py --input-gain-db, noisy scaled before enhancement, clean unchanged; logs only)

Full validation (164 files), in-memory 16-bit, PESQ-WB:

| input gain | gru_v1 | mmse_lsa_top1 |
|---|---|---|
| 0 dB | 2.3594 | 1.7505 |
| -10 dB | 2.2981 (-0.061) | 1.7505 (0.000) |
| -20 dB | 2.0695 (-0.290) | 1.7498 (-0.001) |

gru_v1 depends on input level (training remix gain range was -6..+3 dB), but it still beats MMSE-LSA by +0.32 at -20 dB. MMSE-LSA is effectively scale-invariant. A possible follow-up is input level normalization before the GRU (done in best-v4, see below). Logs: `logs/probe_{submission,mmse_lsa_top1}_gain{-20,-10,0}.log`.

### Level guard variants (exp/level-norm, gru_v1 weights unchanged; full validation, logs only)

| input gain | off (gru_v1) | ln_active (target) | lnr (range clamp) | lnrb (range_boost, best-v4) | lnrb files boosted / cut |
|---|---|---|---|---|---|
| 0 dB | 2.3594 | 2.3539 | 2.3598 | 2.3598 | 0% / 17.1% |
| -10 dB | 2.2981 | 2.3539 | 2.3055 | 2.3232 | 40.9% / 0% |
| -20 dB | 2.0695 | 2.3533 | 2.2871 | 2.3533 | 100% / 0% |
| -30 dB | 1.7698 | 2.3370 | 2.2707 | 2.3370 | 100% / 0% |

Acceptance (0 dB >= 2.3574, -20 dB >= 2.33, -30 dB >= 2.30) met only by lnrb. At -10 dB 59% of files land inside the train range and are left untouched, so lnrb (2.3232) is below ln_active there. Logs: `logs/probe_lnrb_{-30,-20,-10,0}.out`, `logs/probe_lnrb_fractions.txt`.
