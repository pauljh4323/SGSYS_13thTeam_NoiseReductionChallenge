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
