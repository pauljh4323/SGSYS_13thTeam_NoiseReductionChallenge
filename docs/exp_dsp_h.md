# exp/dsp-h: beta-order gain, pause expander, two-band noise bias on dsp-v5 (2026-11-11) — promoted as dsp-v6

Protocol: train-400 subset (seed 0), each stage from the values adopted so far (stage reference = configs/dsp_g.yaml,
nothing adopted before H3); paired per-file PESQ vs the stage reference; keep a value only if d >= 0.002 and d >= 2*SE.
Code hash 666e86e (options), finalist config 74133b6. New options default to off; dsp-v5 outputs bit-identical on 20
train files; beta -> 0 equals LSA (|beta| < 1e-6 exact, 1e-4 within 2e-5 relative); beta = 1 matches the closed-form
MMSE-STSA gain (3e-15). Valid beta range used: beta > -2 (Gamma(beta/2+1)); the paper's recommended range is not verified.

| stage | grid | result (d ± SE, z) | runner-up / note |
|---|---|---|---|
| H1 | gain_mode beta, beta {−0.5, 0.5, 1.0} (0 = LSA reference) | kept LSA | best 0.5: −0.0124 ± 0.0028; −0.5: −0.0133 ± 0.0030; 1.0 (STSA): −0.0489 ± 0.0055. LSA is interior |
| H2 | pause_thr_db {3,6} × pause_att_db {4,8} × release {4,8} | kept off | best 6/4/4: +0.0040 ± 0.0020 (z 1.98 < 2); runner-up 6/4/8 +0.0021 ± 0.0021; att 8 all −0.014..−0.023. Best is a grid corner (thr 6 max, att 4 min, release 4 min), not extended |
| H3 | noise_bias_low {1.0,1.25,1.5} × noise_bias_high {1.0,1.25,1.5} | **adopt 1.25 / 1.5**: +0.0032 ± 0.0015 (z 2.14) | runner-up 1.0/1.5 +0.0011 ± 0.0026; high 1.0 all −0.015..−0.022. noise_bias_high 1.5 at the grid edge, not extended |

Finalist configs/dsp_h.yaml = dsp-v5 + noise_bias_low 1.25 / noise_bias_high 1.5 (split 1 kHz, 200 Hz raised-cosine cross-fade).

Train per target SNR (full train, paired vs dsp_g): 0 dB +0.0050 ± 0.0013, 5 dB +0.0069 ± 0.0033, 10 dB +0.0031 ± 0.0014,
15 dB +0.0054 ± 0.0019; SNR change ≤ 0.03 dB at every level.

| check (revised rule 2026-11-11) | dsp_h | dsp-v5 (dsp_g) | criterion | pass |
|---|---|---|---|---|
| full train paired d (n=1296) | +0.0049 ± 0.0011 (z 4.45); 1.9137 vs 1.9088 | — | d >= 0.003 and z >= 3 | yes |
| val PESQ (0/5/10/15 dB) | 1.9637 (1.296/1.699/1.949/2.337); paired +0.0067 ± 0.0020 | 1.9570 (1.290/1.694/1.939/2.332) | >= 1.9570 | yes |
| val SNR | 12.70 dB | 12.69 dB | report | — |
| val_remix PESQ / SNR | 1.8524 / 11.04 dB; paired +0.0024 ± 0.0015 (10/15 dB −0.0016/−0.0042) | 1.8500 / 11.02 dB | >= 1.8500 | yes |
| level probe −30/−20/−10/0 dB | 1.9589/1.9646/1.9639/1.9637 | 1.9519/1.9578/1.9572/1.9570 | >= v5 − 0.02 | yes |
| main.py CPU (validation, 164 files) | 11.1 s | 12.0 s (same session) | < 180 s | yes |

Decision: all criteria pass → merge to main, submission.yaml = dsp_h, tag dsp-v6. Caveat: the val_remix gain is small and
not significant on its own (z 1.6) and is negative at 10/15 dB; the rule requires only val_remix >= current.

Logs: logs/sweep_dsp_h_h{1,2,3}_paired.txt, logs/eval_dsp_h_{train,validation,val_remix}.log,
logs/paired_dsp_h_vs_dsp_g_train.txt, logs/probe_dsp_h_g{-30,-20,-10,0}.log, logs/time_dsp_h.log.
