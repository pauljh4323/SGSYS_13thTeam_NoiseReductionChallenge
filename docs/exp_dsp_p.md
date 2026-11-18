# Experiment dsp-p: a priori SNR and gain stage on dsp-v7 (branch exp/dsp-p)

Goal: improve the a priori SNR / gain stage of dsp-v7 (configs/dsp_k.yaml) with three blind options in
src/nrc/methods/mmse_lsa.py: bidirectional decision-directed estimation (`dd_direction`), a non-local-means
estimate of the ML a priori SNR term (`xi_nlm`, `nlm_*`) and low-SNR 2D gain smoothing (`gain_smooth`, `gs_*`).
All selection on the FULL train set (1296), paired keep rule d >= 0.002 and >= 2 SE (scripts/sweep.py stage verdicts).

Defaults are bit-identical to dsp-v7 (20 stratified train files vs tag dsp-v7, max |diff| 0.0, implicit and explicit
defaults; selfcheck). Stage reference per-file PESQ reused from `outputs/dsp_k/eval_train.csv` (eval.py, hash c330f12,
1296 files, mean 1.9440): src/ and scripts/eval.py are unchanged c330f12 -> main (b39cf89), inserted with
scripts/sweep_inject_ref.py under sweep.py's own key.

## Changes from the brief (recorded as required)

1. `nlm_domain` {log (default), log_bc, lin} added after the first 5 P2a points (team decision). Reason: the NLM
   weighted mean is taken over log(max(gamma - 1, xi_min)). For an exponentially distributed periodogram
   E[ln X] = ln E[X] - 0.5772, so the log-domain mean under-estimates the ML a priori SNR by ~2.5 dB (more
   suppression). The first 5 P2a points (log domain) all lost 0.035-0.071 PESQ, with the largest h the least bad.
   The same log-mean bias may explain the earlier cepstral-smoothing failure (cepstral_xi compensates only the
   periodogram bias of the ML PSD; not re-tested here). log_bc adds 0.5772 (natural-log units) before exp; lin takes
   the weighted mean of the linear ML values max(gamma - 1, 0) with the same log-domain patch-distance weights.
   The remaining 19 log-domain P2a points were not run; a mechanism check (patch 3x3, search 4x2, nlm_mix 0.5,
   nlm_domain {log_bc, lin} x nlm_h {2, 4, 8}) decides whether the P2a grid continues (any point d >= -0.002).
2. Sweeps run at --jobs 2 after the P2a run at --jobs 4 was stopped under system memory pressure.
3. P2a continued in nlm_domain log_bc only (P2m: log_bc h 8 adopted, d=+0.0053 +- 0.0009, z 6.22, at the h edge).
   Its nlm_h values {0.5, 1, 2} were replaced by {4, 8, 16}: h 2 lost 0.054 in P2m (h 0.5 / 1 smaller still), and 16 is
   the required edge extension. The two points identical to P2m points (3x3, 4x2, h 4 / 8) were copied, not re-run.
4. Sweeps run as detached processes (cmd start, logs/sweep_<name>_run.txt + done marker) because background shells
   were stopped by the host under memory pressure; a waiter checks the done marker and that the sweep process lives.

## NLM: effect of the log-domain bias correction

The Euler-constant correction (`nlm_domain: log_bc`) turned NLM from a loss into a gain at the same settings: plain
log-domain NLM lost about 0.035-0.071 PESQ (P2a first 5 points, best h 2 d = -0.035), log_bc reached d = +0.0053 +- 0.0009
(P2m, 3x3 / 4x2 / h 8, paired, z 6.22) and about +0.010 at search +-8 frames (P2a-bc, unpaired mean while running), i.e.
from about -0.07 to about +0.01. The linear-domain mean (`lin`) lost at every h (-0.053 to -0.077). Related method:
Breithaupt, Gerkmann & Martin 2008 compensate the same log-domain bias after cepstro-temporal smoothing
(docs/references.md, pending).

## Stage results (full train n=1296, paired vs the stage reference)

| stage | adopted | d +- SE (z) | runner-up / notes |
|---|---|---|---|
| P1 dd_direction | forward (= dsp-v7) | - | avg -0.0048 +- 0.0008, geo -0.0100, backward -0.0399 |
| P1b alpha (forward) | 0.94 (= dsp-v7) | - | 0.96 -0.0003 +- 0.0009, 0.92 -0.0044 |
| P2a log domain (5 of 24 pts) | stopped | - | all -0.035..-0.071 (log-mean bias, see above) |
| P2m domain check (3x3, 4x2, mix 0.5) | log_bc h 8 | +0.0053 +- 0.0009 (6.22) | log_bc h 4 -0.0011; lin -0.053..-0.077 |
| P2a-bc (log_bc, 24 pts) vs dsp_k | 3x3, search_t 8, search_f 2, h 16 | +0.0121 +- 0.0014 (8.75) | 5x3 same search/h +0.0118; both edges (search_t 8, h 16) |
| ext nlm_search_t 16 | adopted | +0.0070 +- 0.0008 (9.04) | vs previous step |
| ext nlm_h 32 (search_t 16) | not adopted | -0.0009 +- 0.0002 | h 16 interior |
| ext nlm_search_t 32 / 64 / 128 | adopted each | +0.0075 / +0.0088 / +0.0058 | stopped by the pipeline cap (4 steps per parameter) while still winning; +-128 frames (2.05 s each side) already spans the whole of a 3 s file |
| P2a total vs dsp_k | 3x3, search_t 128, search_f 2, h 16 | +0.0412 +- 0.0028 (14.76) | |
| P2b nlm_mix | 0.5 kept | - | 0.6 -0.0010 +- 0.0004, 0.3 -0.0027, 1.0 -0.0497 |
| P3 gain_smooth (12 pts) | not adopted | - | best thr -5 / bins 1 / frames 0 -0.0205 +- 0.0012; every point lost |

Adopted P2a (train, vs dsp_k): per SNR 0/5/10/15 dB +0.020/+0.038/+0.040/+0.049; easy/moderate/hard +0.047/+0.028/+0.040;
categories alarm_siren +0.042, phone_dialing +0.111, bell +0.049, click +0.033 (n 3), clap +0.011 (n 3), footsteps +0.032
(n 8), wind +0.040; no category negative (smallest factory +0.005). Logs: logs/breakdown_dsp_p_*.txt.

## Finalist configs/dsp_p.yaml (= dsp-v7 + xi_nlm log_bc, 3x3, +-128 x +-2, h 16, mix 0.5), clean hash 38c909a

| split | PESQ dsp_p / dsp_k | paired d +- SE (z) | SNR dB dsp_p / dsp_k | per SNR 0/5/10/15 dB PESQ | easy/moderate/hard |
|---|---|---|---|---|---|
| train (1296) | 1.9851 / 1.9440 | +0.0412 +- 0.0028 (14.76) | 11.52 / 11.56 | 1.458/1.735/1.942/2.328 | 2.004/2.111/1.701 |
| validation (164) | 2.0214 / 1.9884 | +0.0330 +- 0.0046 (7.16) | 12.81 / 12.84 | 1.344/1.780/2.000/2.387 | 2.069/2.052/1.772 |
| val_remix (328) | 1.9221 / 1.8942 | +0.0280 +- 0.0030 (9.34) | 11.18 / 11.20 | 1.298/1.638/2.094/2.663 | 2.043/1.888/1.481 |

Level probe (validation, input gain 0/-10/-20/-30 dB): 2.0214/2.0216/2.0220/2.0158 vs dsp-v7 1.9884/1.9887/1.9900/1.9840
(all +0.032..+0.033). O1->O2 gap fraction (val): (2.0214 - 1.9884) / (2.670 - 1.9884) = 0.048.
NLM runtime (median of 10 train files, 3 s each, measured under concurrent eval load): dsp-v7 0.10 s, P2a-bc best
(search_t 8) 0.30 s, finalist (search_t 128) 3.0 s per file. SNR is flat (-0.03 dB on val): the gain is perceptual.

Promotion rule (round 1): train d +0.0412 >= 0.003 and z 14.76 >= 3; val 2.0214 >= 1.9884; val_remix 1.9221 >= 1.8942;
each probe level >= dsp-v7 - 0.02. **PASSED -> dsp-v8.**

Run notes: the pipeline (scripts/run_pipeline_dsp_p.py) ran detached 2026-11-17 23:50 -> 2026-11-18 12:16. No stall found:
per-point times matched the NLM cost (P2b ~37 min/pt, P3 ~33 min/pt at search_t 128, --jobs 2) and the System log has no
Kernel-Power sleep events in that window. The round-1 deadline (2026-11-17 23:59 KST) passed while it ran.
