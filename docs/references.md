# References

Every algorithm source used in this repository, including methods that were tried and rejected. Entries were proposed by AI and are marked
**verification: pending (team must confirm)** until a team member has checked that the paper
exists and that the bibliographic data (title, authors, year, DOI) are correct. When verified, replace
"pending" with "verified by <name>, <date>".

On 2026-11-15 the bibliographic data (volume, issue, pages, DOI) of every entry were cross-checked against
publisher pages and indexing databases with an AI-assisted web search, and the earlier "from memory" notes
were resolved. This cross-check does not replace the team's own verification: every entry stays pending
until a team member opens the link and confirms it.

## Wiener filter with decision-directed a priori SNR (course baseline, `src/nrc/methods/wiener.py`)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Enhancement and bandwidth compression of noisy speech (Proc. IEEE 67(12), 1586-1604) | J. S. Lim, A. V. Oppenheim | 1979 | https://doi.org/10.1109/PROC.1979.11540 | Wiener-type spectral gain for speech enhancement | pending (team must confirm) |
| Speech enhancement based on a priori signal to noise estimation (Proc. IEEE ICASSP 1996, vol. 2, 629-632) | P. Scalart, J. Vieira Filho | 1996 | https://doi.org/10.1109/ICASSP.1996.543199 | Wiener gain driven by the decision-directed a priori SNR | pending (team must confirm) |
| Course notebook `noise_reduction_baselines.ipynb` (SWE2011 Signals and Systems, noise reduction challenge baseline) | course staff | 2026 | course material (not public) | Wiener baseline ported 1:1, low-energy-frame noise estimate, course SNR definition | pending (team must confirm) |

## MMSE-STSA and the decision-directed a priori SNR estimator

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator (IEEE Trans. ASSP 32(6), 1109-1121) | Y. Ephraim, D. Malah | 1984 | https://doi.org/10.1109/TASSP.1984.1164453 | decision-directed a priori SNR (alpha), MMSE amplitude estimation framework | pending (team must confirm) |

## MMSE-LSA (submitted method, `src/nrc/methods/mmse_lsa.py`, `configs/submission.yaml`)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Speech enhancement using a minimum mean-square error log-spectral amplitude estimator (IEEE Trans. ASSP 33(2), 443-445) | Y. Ephraim, D. Malah | 1985 | https://doi.org/10.1109/TASSP.1985.1164550 | LSA gain G = xi/(1+xi) * exp(0.5 * E1(v)) | pending (team must confirm) |

## MMSE-LSA extensions (`src/nrc/methods/mmse_lsa.py`, branch exp/dsp-tune)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Improved signal-to-noise ratio estimation for speech enhancement (IEEE Trans. Audio, Speech, and Language Processing 14(6), 2098-2108) | C. Plapous, C. Marro, P. Scalart | 2006 | https://doi.org/10.1109/TASL.2006.872621 | `xi_mode: tsnr` (two-step a priori SNR) and `xi_mode: hrnr` (harmonic regeneration via half-wave rectification, rho) | pending (team must confirm) |
| A novel a priori SNR estimation approach based on selective cepstro-temporal smoothing (Proc. IEEE ICASSP 2008, 4897-4900) | C. Breithaupt, T. Gerkmann, R. Martin | 2008 | https://doi.org/10.1109/ICASSP.2008.4518755 (corrected 2026-11-15: the previous DOI 10.1109/ICASSP.2008.4518655 belongs to a different ICASSP 2008 paper by Batliner et al.; the new DOI comes from an indexing database — confirm on IEEE Xplore) | `xi_cepstral_smoothing` (quefrency-dependent recursive smoothing of the ML speech PSD cepstrum, pitch-peak preservation, Euler-constant bias compensation); rejected (lowered PESQ in every stage) | pending (team must confirm) |
| On the theory of filter amplifiers (Experimental Wireless and the Wireless Engineer 7, 536-541, October 1930) | S. Butterworth | 1930 | (no DOI; journal print) | `highpass_hz` Butterworth high-pass (`scipy.signal.butter` + `sosfiltfilt`) | pending (team must confirm) |

## Noise estimation (tried, not in the submission)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Noise estimation by minima controlled recursive averaging for robust speech enhancement (IEEE Signal Processing Letters 9(1), 12-15) | I. Cohen, B. Berdugo | 2002 | https://doi.org/10.1109/97.988717 | `noise_est: mcra` option (rejected: lower PESQ than the initial-frames estimate) | pending (team must confirm) |
| All-pole modeling of degraded speech (IEEE Trans. ASSP 26(3), 197-210) | J. S. Lim, A. V. Oppenheim | 1978 | https://doi.org/10.1109/TASSP.1978.1163086 | Inspiration for `two_pass` (branch exp/dsp-twopass): the iterative idea of re-estimating from a first enhanced estimate. Our two-pass residual noise re-estimation is our own variant, not the paper's algorithm (rejected) | pending (team must confirm) |

## IMCRA and OM-LSA (`src/nrc/methods/omlsa.py`, branch exp/omlsa; rejected)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Noise spectrum estimation in adverse environments: improved minima controlled recursive averaging (IEEE Trans. Speech and Audio Processing 11(5), 466-475) | I. Cohen | 2003 | https://doi.org/10.1109/TSA.2003.811544 | `noise_est: imcra` (two-iteration minimum tracking, sub-windows, a priori absence q, bias beta; paper default constants) | pending (team must confirm) |
| Speech enhancement for non-stationary noise environments (Signal Processing 81(11), 2403-2418) | I. Cohen, B. Berdugo | 2001 | https://doi.org/10.1016/S0165-1684(01)00128-1 | `gain_mode: omlsa` (G = G_LSA^p G_min^(1-p); a priori absence from local/global/frame recursive xi, q_max) | pending (team must confirm) |

## dsp-h extensions (`src/nrc/methods/mmse_lsa.py`, branch exp/dsp-h)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| beta-order MMSE spectral amplitude estimation for speech enhancement (IEEE Trans. Speech and Audio Processing 13(4), 475-486) | C. H. You, S. N. Koh, S. Rahardja | 2005 | https://doi.org/10.1109/TSA.2005.848883 | `gain_mode: beta` gain function; valid range used here beta > -2 from the formula (Gamma(beta/2+1)), the paper's recommended range not checked (rejected) | pending (team must confirm) |
| A multi-band spectral subtraction method for enhancing speech corrupted by colored noise (Proc. IEEE ICASSP 2002, vol. 4, IV-4164) | S. Kamath, P. Loizou | 2002 | https://doi.org/10.1109/ICASSP.2002.5745591 | idea of frequency-dependent noise over-estimation (`noise_bias_low` / `noise_bias_high`, 1 kHz split; adopted in dsp-v6) | pending (team must confirm) |
| Pause expander (`pause_thr_db`, `pause_att_db`, `pause_release_frames`) | own design (this repository) | 2026 | - | frame-level extra attenuation when the 300-4000 Hz mean a posteriori SNR is low (rejected) | n/a (own design) |

## dsp-k extensions (`src/nrc/methods/mmse_lsa.py`, branch exp/dsp-k)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Unbiased MMSE-based noise power estimation with low complexity and low tracking delay (IEEE Trans. Audio, Speech, and Language Processing 20(4), 1383-1393) | T. Gerkmann, R. C. Hendriks | 2012 | https://doi.org/10.1109/TASL.2011.2180896 | a posteriori speech presence probability with fixed prior 0.5 and fixed a priori SNR xi_H1 (`noise_est: spp_mean`, `spp_xi_h1_db`); the whole-file iterative SPP-weighted mean is our own variant, not the paper's recursive tracker (adopted in dsp-v7) | pending (team must confirm) |
| Noise power spectral density estimation based on optimal smoothing and minimum statistics (IEEE Trans. Speech and Audio Processing 9(5), 504-512) | R. Martin | 2001 | https://doi.org/10.1109/89.928915 | idea of a noise floor from the minimum of the smoothed periodogram over a window (`tv_noise`); the zero-phase smoothing, centred window and upward-only max with the stationary PSD are our own design (no bias compensation from the paper; rejected) | pending (team must confirm) |

## dsp-m phase options (`src/nrc/phase.py`, `src/nrc/methods/mmse_lsa.py`, branch exp/dsp-m; rejected)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Signal estimation from modified short-time Fourier transform (IEEE Trans. ASSP 32(2), 236-243) | D. W. Griffin, J. S. Lim | 1984 | https://doi.org/10.1109/TASSP.1984.1164317 | `phase_consistency_iters`: iterations X <- A exp(j angle(STFT(ISTFT(X)))) | pending (team must confirm) |
| Consistent Wiener filtering for audio source separation (IEEE Signal Processing Letters 20(3), 217-220) | J. Le Roux, E. Vincent | 2013 | https://doi.org/10.1109/LSP.2012.2225617 | idea of enforcing STFT consistency on a masked spectrogram; our version keeps the noisy phase in high-gain bins (`phase_gain_thr`) and only iterates the phase (not the paper's consistency-penalised Wiener estimate) | pending (team must confirm) |
| STFT phase reconstruction in voiced speech for an improved single-channel speech enhancement (IEEE/ACM Trans. Audio, Speech, and Language Processing 22(12), 1931-1940) | M. Krawczyk, T. Gerkmann | 2014 | https://doi.org/10.1109/TASLP.2014.2354236 | `phase_harmonic`: phase propagation along time at harmonic frequencies, window phase offset in neighbouring bins; re-anchoring in high-gain bins, the gain-threshold blend and the f0 jump reset are our own choices | pending (team must confirm) |
| YIN, a fundamental frequency estimator for speech and music (J. Acoust. Soc. Am. 111(4), 1917-1930) | A. de Cheveigne, H. Kawahara | 2002 | https://doi.org/10.1121/1.1458024 | `yin_f0` (difference function, cumulative-mean normalisation, absolute threshold, parabolic interpolation); the time re-alignment of each estimate to the frame centre is our own addition | pending (team must confirm) |

## Software and evaluation metric

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| SciPy 1.0: fundamental algorithms for scientific computing in Python (Nature Methods 17(3), 261-272) | P. Virtanen et al. | 2020 | https://doi.org/10.1038/s41592-019-0686-2 | STFT/ISTFT (`scipy.signal`), Butterworth filter, exponential integral E1 (`scipy.special.exp1`) | pending (team must confirm) |
| ITU-T Recommendation P.862.2 (11/2007): Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs (PESQ wideband extension; first edition 11/2005, superseded) | ITU-T | 2007 | https://www.itu.int/ITU-T/recommendations/rec.aspx?rec=9275 | definition of the evaluation metric PESQ-WB (evaluation only, not used in processing) | pending (team must confirm) |
| ludlows/python-pesq v0.0.4 (PyPI package `pesq` 0.0.4, wideband PESQ wrapper of the ITU reference code) | M. Wang, C. Boeddeker, R. G. Dantas, A. Seelan | 2022 | https://doi.org/10.5281/zenodo.6549559 (code: https://github.com/ludlows/PESQ) | evaluation metric only | pending (team must confirm) |

## A priori SNR / gain stage (branch exp/dsp-p, `dd_direction`, `xi_nlm`, `gain_smooth` in `src/nrc/methods/mmse_lsa.py`)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| A non-local algorithm for image denoising (Proc. IEEE CVPR 2005, vol. 2, 60-65) | A. Buades, B. Coll, J.-M. Morel | 2005 | https://doi.org/10.1109/CVPR.2005.38 (from memory — confirm on IEEE Xplore) | `xi_nlm`: non-local-means weighted average of the log ML a priori SNR over the spectrogram (patch distance weights exp(-d^2/h^2)) | pending (team must confirm) |
| Efficient musical noise suppression for speech enhancement systems (Proc. IEEE ICASSP 2009, 4409-4412) | T. Esch, P. Vary | 2009 | https://doi.org/10.1109/ICASSP.2009.4960607 (from memory — confirm on IEEE Xplore) | `gain_smooth`: 2D moving-average smoothing of the spectral gain in low-SNR frames only (our simplified variant) | pending (team must confirm) |
| A novel a priori SNR estimation approach based on selective cepstro-temporal smoothing (Proc. IEEE ICASSP 2008, 4897-4900) | C. Breithaupt, T. Gerkmann, R. Martin | 2008 | https://doi.org/10.1109/ICASSP.2008.4518755 (see the MMSE-LSA extensions table) | related reference for `nlm_domain: log_bc`: Euler-constant bias compensation after smoothing in the log domain (E[ln X] = ln E[X] - 0.5772 for an exponentially distributed periodogram) | pending (team must confirm) |
| Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator (IEEE Trans. ASSP 32(6), 1109-1121) | Y. Ephraim, D. Malah | 1984 | https://doi.org/10.1109/TASSP.1984.1164453 | `dd_direction` backward / avg / geo: bidirectional decision-directed recursion is our own non-causal variant of the paper's (forward) DD estimator, not from the paper | pending (team must confirm) |
