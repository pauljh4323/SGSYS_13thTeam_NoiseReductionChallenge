# exp/dsp-m: classical phase enhancement on dsp-v7 (2026-11-11) — nothing adopted, not promoted

Motivation: docs/exp_oracle.md. Replacing the noisy phase with the clean phase under the dsp-v6 magnitude gave
+0.286 (val) / +0.260 (train-400). Noise-level tuning has run out (dsp_l failed val_remix). This is the last
classical experiment of round 1. Branch from main (dsp-v7). Code hash 9869fda.

Protocol: train-400 subset (seed 0). Each stage starts from the values adopted so far. Every point gets a paired
per-file PESQ comparison against the stage reference and is kept only if d >= 0.002 and d >= 2*SE.

## Method (`src/nrc/phase.py`, options in `src/nrc/methods/mmse_lsa.py`)
Both options act on the enhanced STFT after the magnitude gain and before the 100 Hz high-pass. The magnitude
stays |G Y|. With the options off (default), the code runs the same operations as before. Outputs are
bit-identical to dsp-v7 (dsp_k) on 20 train files (max |diff| 0).

- **P1 `phase_consistency_iters` K, `phase_gain_thr` g:** K iterations of X <- A exp(j angle(STFT(ISTFT(X)))),
  after Griffin & Lim 1984 and Le Roux & Vincent 2013. Bins with gain >= g keep their initial (noisy) phase on
  every iteration. Check: a consistent STFT is a fixed point (max |dphi| 2.7e-13).
- **P2 `phase_harmonic`:** harmonic phase reconstruction (Krawczyk & Gerkmann 2014 style). Parameters:
  `phase_beta`, `phase_bins`, `phase_reanchor`, `f0_min`/`f0_max` (70/400 Hz), `yin_threshold` 0.15,
  `harmonic_max_hz` 4000.
  - **f0 estimate:** per STFT frame, from the dsp-v7 output (after the high-pass), using a YIN-style estimator
    (de Cheveigne & Kawahara 2002) with a 512-sample integration window and parabolic interpolation.
  - **Time alignment (own addition):** a YIN estimate belongs to the time start + win/2 + tau/2, not to the
    frame centre. Each voiced run is linearly re-sampled to the frame centres. Without this, a gliding f0 is
    biased by about 0.03 Hz, and the open-loop phase propagation drifts with it. The synthetic error was
    0.288 rad before the fix and 0.099 rad after.
  - **Phase convention:** scipy frame l is centred on sample l*hop and its FFT origin is the frame start. A
    sinusoid with centre phase phi_c therefore has STFT phase phi_c - pi*k in every main-lobe bin k, at any
    frequency.
  - **Propagation and re-anchoring:** phi_c,h(l) = phi_c,h(l-1) + 2*pi*h*mean(f0(l-1), f0(l))*hop/fs
    (trapezoid). The phase starts at voicing onset from phi_noisy(k_h) + pi*k_h. A relative f0 jump > 20% also
    counts as an onset (own choice). With `phase_reanchor`, a harmonic whose bin k_h has gain >= g is reset
    from the noisy phase instead of being propagated.
  - **Bins changed:** k_h +- `phase_bins` get phi_c - pi*k. Only bins with gain < g change, as
    angle(beta e^{j phi_rec} + (1-beta) e^{j phi_noisy}).
  - **Order:** P2 runs first and P1 starts from its phase.

## Synthetic phase test (`scripts/phase_synth_test.py`, logs/phase_synth_test.txt) — PASS
Signal: 10 harmonics with amplitude 1/h. f0 glides 150 -> 165 Hz over 2 s, in two voiced segments with
silence between them. STFT is Hann 1024 / hop 256. The error is the mean |wrapped phase error| in bins k_h +- 1
of frames whose window lies fully inside voiced speech, excluding the frames where the track (re)starts.

| Case | Mean abs phase error (rad) |
|---|---|
| **Clean, YIN f0, re-anchor off (pass/fail, < 0.3)** | **0.099** (86 frames; YIN abs f0 error 0.006 Hz) |
| Same, by frames since onset 1-5 / 6-15 / 16-30 / 31+ | 0.064 / 0.074 / 0.098 / 0.133 |
| Diag (a): clean, true f0 | 0.261 (anchors at the true onset, where the window is mostly silence or ramp: constant per-harmonic offset, no drift) |
| Diag (b): 5 dB white noise, noisy phase as is | 0.204 (only 4.9% of harmonic bins have gain < 0.6) |
| (b) f0 from the noisy signal | YIN finds no voiced frames at threshold 0.15, so nothing is reconstructed (0.204) |
| (b) f0 from the dsp_k output (pipeline), re-anchor on / off | 0.152 / 0.166 (abs f0 error 0.066 Hz) |

## M1: P1 at dsp_k (logs/sweep_dsp_m_m1_paired.txt)

| K | g | PESQ | d +- SE | z |
|---|---|---|---|---|
| 1 | 0.3 | 1.9432 | -0.0079 +- 0.0006 | -12.3 |
| 3 | 0.3 | 1.9400 | -0.0111 +- 0.0009 | -12.8 |
| 1 | 0.6 | 1.9379 | -0.0132 +- 0.0009 | -13.9 |
| 1 | 1.0 | 1.9353 | -0.0158 +- 0.0012 | -13.6 |
| 3 | 0.6 | 1.9322 | -0.0188 +- 0.0012 | -15.1 |
| 3 | 1.0 | 1.9283 | -0.0227 +- 0.0015 | -15.5 |

dsp_k reference: 1.9511. **Kept dsp_k.** Every point loses, and the loss grows with K and with g, i.e. with the
number of bins the iterations change. The best point is at the low edge of both parameters, which points toward
"off". Runner-up: K=3, g=0.3. Best-point deltas, mean +- SE (n):

| Group | d +- SE (n) |
|---|---|
| 0 dB | -0.0025 +- 0.0012 (36) |
| 5 dB | -0.0078 +- 0.0012 (92) |
| 10 dB | -0.0088 +- 0.0014 (133) |
| 15 dB | -0.0084 +- 0.0009 (139) |
| easy | -0.0068 +- 0.0007 (240) |
| moderate | -0.0121 +- 0.0019 (100) |
| hard | -0.0051 +- 0.0011 (60) |

## M2: P2 at dsp_k, re-anchor on (logs/sweep_dsp_m_m2_paired.txt)

| g | beta | bins | PESQ | d +- SE | z |
|---|---|---|---|---|---|
| 0.6 | 0.5 | 2 | 1.9526 | +0.0016 +- 0.0004 | 4.0 |
| 0.6 | 1.0 | 2 | 1.9524 | +0.0013 +- 0.0007 | 1.7 |
| 0.3 | 0.5 | 2 | 1.9520 | +0.0009 +- 0.0003 | 3.4 |
| 0.6 | 0.5 | 1 | 1.9518 | +0.0008 +- 0.0003 | 3.0 |
| 0.6 | 1.0 | 1 | 1.9516 | +0.0005 +- 0.0005 | 1.0 |
| 0.3 | 0.5 | 1 | 1.9514 | +0.0004 +- 0.0002 | 2.1 |
| 0.3 | 1.0 | 2 | 1.9512 | +0.0001 +- 0.0005 | 0.3 |
| 0.3 | 1.0 | 1 | 1.9509 | -0.0002 +- 0.0003 | -0.6 |

**Kept dsp_k.** The best point is real (z 4.0) but small: +0.0016 is below the 0.002 minimum. Runner-up: g 0.6,
beta 1.0, bins 2, +0.0013 +- 0.0007. **Edge flag:** the best point is at a grid edge in all three parameters (g
upper, beta lower, bins upper). Best-point deltas, mean +- SE (n):

| Group | d +- SE (n) |
|---|---|
| 0 dB | +0.0024 +- 0.0010 (36) |
| 5 dB | +0.0020 +- 0.0008 (92) |
| 10 dB | +0.0016 +- 0.0007 (133) |
| 15 dB | +0.0010 +- 0.0006 (139) |
| easy | +0.0025 +- 0.0006 (240) |
| moderate | -0.0001 +- 0.0007 (100) |
| hard | +0.0007 +- 0.0003 (60) |

These breakdowns are in logs/dsp_m_stage_strata.txt.

## M3 (phase_reanchor true/false) — not run
M3 was planned at the M2-adopted point. M2 adopted nothing, so that point is dsp_k with P2 off, where
`phase_reanchor` has no effect. Per the stop rule, no finalist was built and there was no full-train, validation,
val_remix, probe or CPU run. main is untouched and dsp-v7 stays the submission candidate.

## Interpretation
- **P1 hurts.** The enhanced magnitudes are far from consistent: the high-HRNR gains have up to -30 dB jumps.
  Forcing consistency in low-gain bins spreads energy and smears the residual noise, instead of recovering a
  speech-like phase. The clean-phase oracle gain is therefore not reachable this way.
- **P2 helps a little, but too little.** It changes only harmonic bins below 4 kHz in voiced frames with low gain,
  a small share of the bins (4.9% of harmonic bins in the 5 dB synthetic case). The trend favours wider bins and a
  partial blend. Extending past the grid edges (bins 3, g 0.8, beta 0.25) might cross 0.002. That would be a new
  stage after the fact, and the gain is at most a few thousandths of PESQ.
