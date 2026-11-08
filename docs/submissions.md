# Submissions

| date | tag | method (config) | code commit | val PESQ (16-bit, main.py outputs) | float / in-mem 16-bit | e2e |
|---|---|---|---|---|---|---|
| 2026-11-05 | sub-20261105, best-v1 | course Wiener (configs/submission.yaml) | 2595b98 | 1.5987 (0/5/10/15 dB: 1.1459/1.3788/1.5610/1.9062) | 1.5984 / 1.5987 | PASS: local + fresh clone bit-identical (164 files), 0 failures, 44.1 kHz stereo + 1.7 s file preserved; main.py 164 files in ~7 s CPU |
| 2026-11-05 | best-v2 (candidate, not yet submitted) | MMSE-LSA (configs/submission.yaml = mmse_lsa_top1: noise_est initial, nf 0.20, alpha 0.96, xi_min_db -15, gmin_db -25) | f780702 | 1.7505 (0/5/10/15 dB: 1.1728/1.4687/1.6993/2.1468) | - | PASS: local + fresh clone bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; main.py 164 files in 19.8 s (clone run 10.3 s) CPU |
| 2026-11-05 | best-v3 (candidate, not yet submitted) | BiGRU mask gru_v1 (configs/submission.yaml = configs/dl/gru_v1.yaml inference params, weights/gru_v1.pt 2.4 MB, torch CPU) | 1c8a744 | 2.3594 (0/5/10/15 dB: 1.7324/2.0606/2.3624/2.7271) | - / 2.3594 | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; fresh conda env (nrc_ci from environment.yml, conda-forge only) running the clone's main.py from another cwd: bit-identical to nrc (164/164), env removed afterwards; main.py 164 files ~36 s CPU (73-111 s under parallel load) |
| 2026-11-05 | best-v4 (candidate, not yet submitted) | BiGRU mask gru_v1 + range_boost input level guard (configs/submission.yaml = configs/dl/gru_v1_lnrb.yaml inference params: active level, train p5/p95 -27.88/-15.27 dBFS, boost-below target = train median -19.60; weights/gru_v1.pt unchanged) | 4f8774c | 2.3598 (0/5/10/15 dB: 1.7341/2.0614/2.3629/2.7268) | - / 2.3598 (exp row 9d325ea) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files ~37 s CPU |
| 2026-11-07 | best-v5 (candidate, not yet submitted) | conv2d front-end + BiGRU H=256 mask gru_v2b, level norm off (configs/submission.yaml = configs/dl/gru_v2b_off.yaml inference params, weights/gru_v2b.pt 9.5 MB, trained with U(-30,+6) dB gain aug, Colab) | c95c063 | 2.5051 (0/5/10/15 dB: 1.7975/2.2128/2.5067/2.8913) | - / 2.5051 (exp row 1b08a42) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files 52.6 s CPU |
| 2026-11-08 | best-v6 (candidate, not yet submitted) | gru_v3b: gru_v2b arch fine-tuned from weights/gru_v2b.pt with mixture gain aug + cmag+mrstft loss (lambda 0.5), level norm off (configs/submission.yaml = configs/dl/gru_v3b_off.yaml inference params, weights/gru_v3b.pt 9.5 MB, Colab) | eeebebc | 2.5885 (0/5/10/15 dB: 1.8833/2.3314/2.5864/2.9538) | - / 2.5885 (exp row 97f9274) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files 46.7 s CPU |
| 2026-11-08 | best-v7 (candidate, not yet submitted) | gru_v4b: gru_v3b fine-tuned from weights/gru_v3b.pt with mixture gain aug + cmag+mrstft (0.5) + SI-SDR (0.1) loss, lr 2e-4, level norm off (configs/submission.yaml = configs/dl/gru_v4b_off.yaml inference params, weights/gru_v4b.pt 9.5 MB, Colab) | 265e4a5 | 2.6796 (0/5/10/15 dB: 1.9579/2.4059/2.6800/3.0581) | - / 2.6796 (exp row 05d16ba) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files 24.0 s CPU |

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

### gru_v2 candidates (exp/dl-v2, Colab-trained; full validation, in-memory 16-bit; probe = eval.py --input-gain-db, logs only)

| candidate | PESQ | 0/5/10/15 dB | easy/moderate/hard | SI-SDR | STOI | probe -30/-20/-10/0 dB | main.py 164 files |
|---|---|---|---|---|---|---|---|
| gru_v1_lnrb (best-v4) | 2.3598 | 1.7341/2.0614/2.3629/2.7268 | 2.3037/2.3868/2.5451 | 15.119 | 0.9295 | 2.3370/2.3533/2.3232/2.3598 | 43.9 s |
| gru_v2a_off | 2.2249 | 1.6047/1.9162/2.2092/2.6159 | 2.1747/2.2441/2.3990 | 14.455 | 0.9190 | 2.0526/2.1608/2.2055/2.2249 | 45.7 s |
| gru_v2a_lnrb | 2.2246 | 1.6050/1.9144/2.2090/2.6165 | 2.1738/2.2446/2.4000 | 14.463 | 0.9191 | 2.2114/2.2258/2.2168/2.2246 | 44.2 s |
| **gru_v2b_off (best-v5)** | **2.5051** | 1.7975/2.2128/2.5067/2.8913 | 2.4315/2.5216/2.7804 | 15.768 | 0.9365 | 2.4390/2.4791/2.4921/2.5051 | 52.6 s |
| gru_v2b_lnrb | 2.5042 | 1.7977/2.2119/2.5056/2.8905 | 2.4305/2.5210/2.7795 | 15.769 | 0.9365 | 2.4779/2.4980/2.4932/2.5042 | 51.8 s |

Acceptance (0 dB >= 2.3698, -10/-20 dB >= 2.33, -30 dB >= 2.30, CPU < 180 s): v2b_off and v2b_lnrb pass, v2a fails (0 dB 2.2249). v2b off vs lnrb differ by 0.0009 at 0 dB (< 0.002) -> off preferred. Largest category changes v2b_off vs gru_v1_lnrb (all gains; n = 1-3 files each): machine_gun +0.742, clap +0.459, bell +0.334, click +0.269, animal +0.269. Train logs: v2a best epoch 94 (2.2249), v2b best epoch 148 (2.5051). Logs: `logs/compare_gru_v2_candidates.csv`, `logs/compare_gru_v1_lnrb_vs_gru_v2b_off_by_category.csv`, `logs/eval_gru_v2*_validation*.txt`, `logs/timing_gru_v2.log`, `logs/e2e_v5.out`.

### gru_v3 candidates (exp/dl-v3, Colab fine-tune from gru_v2b; full validation, in-memory 16-bit; probe = eval.py --input-gain-db, logs only)

| candidate | PESQ | 0/5/10/15 dB | easy/moderate/hard | SI-SDR | STOI | probe -30/-20/-10/0 dB | main.py 164 files |
|---|---|---|---|---|---|---|---|
| gru_v2b_off (best-v5) | 2.5051 | 1.7975/2.2128/2.5067/2.8913 | 2.4315/2.5216/2.7804 | 15.768 | 0.9365 | 2.4390/2.4791/2.4921/2.5051 | 52.6 s |
| gru_v3a_off (mixture gain, cmag) | 2.5249 | 1.8181/2.2370/2.5204/2.9140 | 2.4541/2.5222/2.8215 | 15.822 | 0.9374 | 2.4409/2.4900/2.5087/2.5249 | not run |
| **gru_v3b_off (best-v6)** (mixture gain, cmag+mrstft) | **2.5885** | 1.8833/2.3314/2.5864/2.9538 | 2.5343/2.5916/2.8066 | 16.463 | 0.9402 | 2.5284/2.5622/2.5746/2.5885 | 46.7 s |

Acceptance (0 dB >= 2.5151, -10/-20/-30 dB each >= 2.40, CPU < 180 s): v3b passes; v3a fails (0 dB 2.5249). Train logs: v3a best epoch 48 (2.5249), early-stopped at 64; v3b best epoch 76 (2.5885), early-stopped at 92; both final lr 1e-5. v3b vs v2b_off by noise category: 18/23 improved; top gains door_moving +0.258 (n=1), destroyer_operations +0.152, military_vehicle +0.150, phone_dialing +0.141, wind +0.123 (n=26); top losses clap -0.100 (n=1), tooth_brushing -0.089 (n=1), animal -0.025, shower -0.021, water -0.006. Logs: `logs/eval_gru_v3*_validation*.txt`, `logs/compare_gru_v2b_off_vs_gru_v3b_off_by_category.csv`, `logs/timing_gru_v3b_off.out`, `logs/e2e_v3b.log`.

### gru_v4 candidates (exp/dl-v4, Colab fine-tune from gru_v3b; full validation + val_remix, in-memory 16-bit; probe = eval.py --input-gain-db, logs only)

| candidate | PESQ | 0/5/10/15 dB | easy/moderate/hard | SI-SDR | STOI | probe -30/-20/-10/0 dB | val_remix PESQ (0/5/10/15 dB) | main.py 164 files |
|---|---|---|---|---|---|---|---|---|
| gru_v3b_off (best-v6) | 2.5885 | 1.8833/2.3314/2.5864/2.9538 | 2.5343/2.5916/2.8066 | 16.463 | 0.9402 | 2.5284/2.5622/2.5746/2.5885 | 2.4295 (1.7709/2.2303/2.6228/3.0968) | 46.7 s |
| gru_v4a_off (cmag+mrstft 1.0) | 2.5938 | 1.8903/2.3409/2.5867/2.9609 | 2.5383/2.5989/2.8142 | 16.515 | 0.9405 | 2.5392/2.5704/2.5814/2.5938 | 2.4378 (1.7758/2.2399/2.6324/3.1060) | 29.3 s |
| **gru_v4b_off (best-v7)** (cmag+mrstft 0.5+sisdr 0.1) | **2.6796** | 1.9579/2.4059/2.6800/3.0581 | 2.6189/2.6870/2.9173 | 16.820 | 0.9411 | 2.6165/2.6589/2.6705/2.6796 | 2.5048 (1.8146/2.3058/2.7086/3.1931) | 24.0 s |

Acceptance (official PESQ >= 2.5985, -10/-20/-30 dB each >= 2.45, val_remix >= 2.4295, CPU < 180 s): v4b passes; v4a fails (official 2.5938). Train logs: v4a best epoch 30 (2.5938), early-stopped at 46; v4b best epoch 48 (2.6796), early-stopped at 64; both final lr 1.25e-5. Note: the Colab checkpoint selection used the official validation set (training-log best == official eval), so val_remix (+0.075 over v3b) is the held-out confirmation. CPU times differ from v3b only by machine load (same architecture, 2,373,651 params). v4b vs v3b_off by noise category: 21/23 improved (146/164 files); top gains animal +0.201 (n=2), vehicle_interior +0.196 (n=3), bell +0.192 (n=3), tooth_brushing +0.178 (n=1), shower +0.166 (n=1); losses click -0.186 (n=1), phone_dialing -0.062 (n=4). Logs: `logs/eval_gru_v4*_validation*.txt`, `logs/eval_gru_v4*_val_remix.txt`, `logs/compare_gru_v3b_off_vs_gru_v4b_off_by_category.csv`, `logs/time_gru_v4*_off.log`, `logs/e2e_check_v7.log`.
