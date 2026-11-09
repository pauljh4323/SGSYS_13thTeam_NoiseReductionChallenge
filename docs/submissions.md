# Submissions

## Rule change 2026-11-09

The TA notice for **Challenge 1 ("내 목소리를 구해줘!") allows traditional signal processing only**: no machine learning or
deep learning in the submitted algorithm (grid-searched parameters and rule-based input-adaptive parameters are allowed).
Consequences for this repository:

- **best-v3 ... best-v8 (BiGRU mask models, torch) are DL and ineligible for Challenge 1.** Their code, configs and
  weights were removed from `main` and are kept on branch `dl-main` (tags unchanged) for a possible later challenge.
- **No round has been submitted yet.** Tag `sub-20261105` was created but never actually submitted; it was replaced by
  tag `baseline-wiener` (same commit 6083c83) and deleted. `sub-YYYYMMDD` tags are created only after the team confirms
  a submission.
- The first submission candidate is **dsp-v2** = MMSE-LSA with n_fft 1024, noise bias 1.25, HRNR a priori SNR and a
  100 Hz high-pass (`configs/dsp_tune_c.yaml` parameters), validation results in `docs/validation_results.md`.
  It supersedes dsp-v1 (never submitted).

| date | tag | method (config) | code commit | val PESQ / SNR (main.py outputs) | val_remix PESQ / SNR | e2e |
|---|---|---|---|---|---|---|
| 2026-11-09 | dsp-v2 (first submission candidate, not yet submitted) | MMSE-LSA + HRNR + HP (configs/submission.yaml = dsp_tune_c: n_fft 1024, hop 256, noise_est initial, nf 0.10, noise_bias 1.25, alpha 0.96, xi_min_db -15, gmin_db -25, xi_mode hrnr, highpass_hz 100) | e68a073 (eval rows e68a073) | 1.8746 / 12.64 dB (0/5/10/15 dB PESQ 1.2398/1.5725/1.8446/2.2791, SNR 6.71/11.03/12.76/15.20) | 1.7866 / 10.96 dB | PASS: classical allow-list, torch-free main.py, local + fresh clone + other cwd + fresh pip venv (requirements.txt only) all bit-identical (164 files), 44.1 kHz stereo + 1.7 s file enhanced, 0 failed / 0 clipped; main.py CPU 13.4 s (164 files) |
| 2026-11-09 | dsp-v1 (superseded by dsp-v2, never submitted) | MMSE-LSA (configs/submission.yaml = mmse_lsa_top1: noise_est initial, nf 0.20, alpha 0.96, xi_min_db -15, gmin_db -25) | af75105 (eval rows ffa094e) | 1.7505 / 13.56 dB (0/5/10/15 dB PESQ 1.1728/1.4687/1.6993/2.1468, SNR 6.02/10.36/13.83/17.50) | 1.6626 / 11.68 dB | PASS: classical allow-list, torch-free main.py, local + fresh clone + other cwd + fresh pip venv (requirements.txt only) all bit-identical (164 files), 44.1 kHz stereo + 1.7 s file enhanced, 0 failed / 0 clipped |

## History (before the rule change)

| date | tag | method (config) | code commit | val PESQ (16-bit, main.py outputs) | float / in-mem 16-bit | e2e |
|---|---|---|---|---|---|---|
| 2026-11-05 | baseline-wiener (was sub-20261105, never submitted), best-v1 | course Wiener (configs/submission.yaml) | 2595b98 | 1.5987 (0/5/10/15 dB: 1.1459/1.3788/1.5610/1.9062) | 1.5984 / 1.5987 | PASS: local + fresh clone bit-identical (164 files), 0 failures, 44.1 kHz stereo + 1.7 s file preserved; main.py 164 files in ~7 s CPU |
| 2026-11-05 | best-v2 (candidate, not yet submitted) | MMSE-LSA (configs/submission.yaml = mmse_lsa_top1: noise_est initial, nf 0.20, alpha 0.96, xi_min_db -15, gmin_db -25) | f780702 | 1.7505 (0/5/10/15 dB: 1.1728/1.4687/1.6993/2.1468) | - | PASS: local + fresh clone bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; main.py 164 files in 19.8 s (clone run 10.3 s) CPU |
| 2026-11-05 | best-v3 (DL, ineligible for Challenge 1; branch dl-main) | BiGRU mask gru_v1 (configs/submission.yaml = configs/dl/gru_v1.yaml inference params, weights/gru_v1.pt 2.4 MB, torch CPU) | 1c8a744 | 2.3594 (0/5/10/15 dB: 1.7324/2.0606/2.3624/2.7271) | - / 2.3594 | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; fresh conda env (nrc_ci from environment.yml, conda-forge only) running the clone's main.py from another cwd: bit-identical to nrc (164/164), env removed afterwards; main.py 164 files ~36 s CPU (73-111 s under parallel load) |
| 2026-11-05 | best-v4 (DL, ineligible for Challenge 1; branch dl-main) | BiGRU mask gru_v1 + range_boost input level guard (configs/submission.yaml = configs/dl/gru_v1_lnrb.yaml inference params: active level, train p5/p95 -27.88/-15.27 dBFS, boost-below target = train median -19.60; weights/gru_v1.pt unchanged) | 4f8774c | 2.3598 (0/5/10/15 dB: 1.7341/2.0614/2.3629/2.7268) | - / 2.3598 (exp row 9d325ea) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files ~37 s CPU |
| 2026-11-07 | best-v5 (DL, ineligible for Challenge 1; branch dl-main) | conv2d front-end + BiGRU H=256 mask gru_v2b, level norm off (configs/submission.yaml = configs/dl/gru_v2b_off.yaml inference params, weights/gru_v2b.pt 9.5 MB, trained with U(-30,+6) dB gain aug, Colab) | c95c063 | 2.5051 (0/5/10/15 dB: 1.7975/2.2128/2.5067/2.8913) | - / 2.5051 (exp row 1b08a42) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files 52.6 s CPU |
| 2026-11-08 | best-v6 (DL, ineligible for Challenge 1; branch dl-main) | gru_v3b: gru_v2b arch fine-tuned from weights/gru_v2b.pt with mixture gain aug + cmag+mrstft loss (lambda 0.5), level norm off (configs/submission.yaml = configs/dl/gru_v3b_off.yaml inference params, weights/gru_v3b.pt 9.5 MB, Colab) | eeebebc | 2.5885 (0/5/10/15 dB: 1.8833/2.3314/2.5864/2.9538) | - / 2.5885 (exp row 97f9274) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files 46.7 s CPU |
| 2026-11-08 | best-v7 (DL, ineligible for Challenge 1; branch dl-main) | gru_v4b: gru_v3b fine-tuned from weights/gru_v3b.pt with mixture gain aug + cmag+mrstft (0.5) + SI-SDR (0.1) loss, lr 2e-4, level norm off (configs/submission.yaml = configs/dl/gru_v4b_off.yaml inference params, weights/gru_v4b.pt 9.5 MB, Colab) | 265e4a5 | 2.6796 (0/5/10/15 dB: 1.9579/2.4059/2.6800/3.0581) | - / 2.6796 (exp row 05d16ba) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files 24.0 s CPU |
| 2026-11-09 | best-v8 (DL, ineligible for Challenge 1; branch dl-main) | gru_v5b: gru_v4b fine-tuned from weights/gru_v4b.pt with mixture gain aug + cmag+mrstft (0.5) + SI-SDR (0.3) loss, lr 2e-4, level norm off (configs/submission.yaml = configs/dl/gru_v5b_off.yaml inference params, weights/gru_v5b.pt 9.5 MB, Colab) | b705421 | 2.6962 (0/5/10/15 dB: 1.9531/2.4100/2.7022/3.0834) | - / 2.6962 (exp row ee2bced) | PASS: local + fresh clone + run from cwd=outputs/ all bit-identical (164 files), 0 failures, 0 clipped, 44.1 kHz stereo + 1.7 s file enhanced; selfcheck + selfcheck_gru OK; main.py 164 files 90.4 s CPU (slow machine day; v4b re-run 90.2 s) |

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

### gru_v5 candidates (exp/dl-v5, Colab fine-tune from gru_v4b; full validation + val_remix, in-memory 16-bit; probe = eval.py --input-gain-db, logs only)

| candidate | PESQ | 0/5/10/15 dB | easy/moderate/hard | SI-SDR | STOI | probe -30/-20/-10/0 dB | val_remix PESQ (0/5/10/15 dB) | main.py 164 files |
|---|---|---|---|---|---|---|---|---|
| gru_v4b_off (best-v7) | 2.6796 | 1.9579/2.4059/2.6800/3.0581 | 2.6189/2.6870/2.9173 | 16.820 | 0.9411 | 2.6165/2.6589/2.6705/2.6796 | 2.5048 (1.8146/2.3058/2.7086/3.1931) | 90.2 s (re-run 2026-11-09; 24.0 s on 2026-11-08) |
| gru_v5a_off (cmag+mrstft 0.5+sisdr 0.2) | 2.6927 | 1.9490/2.4099/2.6968/3.0798 | 2.6258/2.7066/2.9447 | 16.849 | 0.9406 | 2.6154/2.6685/2.6840/2.6927 | 2.5143 (1.8220/2.3138/2.7195/3.2050) | 88.5 s |
| **gru_v5b_off (best-v8)** (cmag+mrstft 0.5+sisdr 0.3) | **2.6962** | 1.9531/2.4100/2.7022/3.0834 | 2.6302/2.7075/2.9487 | 16.844 | 0.9406 | 2.6137/2.6700/2.6843/2.6962 | 2.5170 (1.8249/2.3175/2.7226/3.2060) | 87.1 / 92.0 s |

Acceptance (official PESQ >= 2.6896, -10/-20/-30 dB each >= 2.55, val_remix >= 2.5048, CPU < 180 s): both pass; v5b promoted (higher official PESQ). Train logs: both best epoch 8 (v5a 2.6927, v5b 2.6962), early-stopped at epoch 24, final lr 5e-5. Checkpoint selection used the official validation set (training-log best == official eval), so the official gain (+0.017) is selection-biased; val_remix (+0.012 over v4b) is the held-out confirmation. The margin is small. CPU: same architecture as v4b; the machine was slower on 2026-11-09 (v4b re-run 90.2 s vs 24.0 s the day before), so the three models run at the same speed. v5b vs v4b_off by noise category: 22/23 improved (115/164 files); largest changes shower +0.106 (n=1), animal +0.097 (n=2), tooth_brushing +0.053 (n=1), clap +0.051 (n=1), broadband_noise +0.042 (n=10); only loss vehicle_interior -0.037 (n=3). Logs: `logs/eval_gru_v5*_validation*.txt`, `logs/eval_gru_v5*_val_remix*`, `logs/probe_gru_v5*_off_g*.log`, `logs/compare_gru_v4b_off_vs_gru_v5b_off_by_category.csv`, `logs/time_gru_v5*_off*.log`, `logs/time_gru_v4b_off_rerun_v5.log`.
