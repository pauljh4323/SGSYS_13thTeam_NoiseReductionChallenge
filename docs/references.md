# References

Every algorithm source used in this repository. Entries were proposed by AI and are marked
**verification: pending (team must confirm)** until a team member has checked that the paper
exists and that the bibliographic data (title, authors, year, DOI) are correct. When verified, replace
"pending" with "verified by <name>, <date>".

## Wiener filter with decision-directed a priori SNR (course baseline, `src/nrc/methods/wiener.py`)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Enhancement and bandwidth compression of noisy speech (Proc. IEEE 67(12)) | J. S. Lim, A. V. Oppenheim | 1979 | https://doi.org/10.1109/PROC.1979.11540 | Wiener-type spectral gain for speech enhancement | pending (team must confirm) |
| Speech enhancement based on a priori signal to noise estimation (Proc. IEEE ICASSP 1996) | P. Scalart, J. Vieira Filho | 1996 | https://doi.org/10.1109/ICASSP.1996.543199 | Wiener gain driven by the decision-directed a priori SNR | pending (team must confirm) |
| Course notebook (SWE2011 Signals and Systems, noise reduction challenge baseline) | course staff | 2026 | course material (not public) | Wiener baseline ported 1:1, low-energy-frame noise estimate | pending (team must confirm) |

## MMSE-STSA and the decision-directed a priori SNR estimator

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator (IEEE Trans. ASSP 32(6)) | Y. Ephraim, D. Malah | 1984 | https://doi.org/10.1109/TASSP.1984.1164453 | decision-directed a priori SNR (alpha), MMSE amplitude estimation framework | pending (team must confirm) |

## MMSE-LSA (submitted method, `src/nrc/methods/mmse_lsa.py`, `configs/submission.yaml`)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Speech enhancement using a minimum mean-square error log-spectral amplitude estimator (IEEE Trans. ASSP 33(2)) | Y. Ephraim, D. Malah | 1985 | https://doi.org/10.1109/TASSP.1985.1164550 | LSA gain G = xi/(1+xi) * exp(0.5 * E1(v)) | pending (team must confirm) |

## MMSE-LSA extensions (`src/nrc/methods/mmse_lsa.py`, branch exp/dsp-tune)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Improved signal-to-noise ratio estimation for speech enhancement (IEEE Trans. Audio, Speech, and Language Processing 14(6)) | C. Plapous, C. Marro, P. Scalart | 2006 | https://doi.org/10.1109/TASL.2006.872621 | `xi_mode: tsnr` (two-step a priori SNR) and `xi_mode: hrnr` (harmonic regeneration via half-wave rectification, rho) | pending (team must confirm) |
| A novel a priori SNR estimation approach based on selective cepstro-temporal smoothing (Proc. IEEE ICASSP 2008) | C. Breithaupt, T. Gerkmann, R. Martin | 2008 | https://doi.org/10.1109/ICASSP.2008.4518655 | `xi_cepstral_smoothing` (quefrency-dependent recursive smoothing of the ML speech PSD cepstrum, pitch-peak preservation, Euler-constant bias compensation) | pending (team must confirm) |
| On the theory of filter amplifiers (Experimental Wireless and the Wireless Engineer 7) | S. Butterworth | 1930 | (no DOI; journal print) | `highpass_hz` Butterworth high-pass (`scipy.signal.butter` + `sosfiltfilt`) | pending (team must confirm) |

## Noise estimation (tried, not in the submission)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Noise estimation by minima controlled recursive averaging for robust speech enhancement (IEEE Signal Processing Letters 9(1)) | I. Cohen, B. Berdugo | 2002 | https://doi.org/10.1109/97.988717 | `noise_est: mcra` option (rejected: lower PESQ than the initial-frames estimate) | pending (team must confirm) |

## dsp-h extensions (`src/nrc/methods/mmse_lsa.py`, branch exp/dsp-h)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| beta-order MMSE spectral amplitude estimation for speech enhancement (IEEE Trans. Speech and Audio Processing 13(4), 475-486) | C. H. You, S. N. Koh, S. Rahardja | 2005 | https://doi.org/10.1109/TSA.2005.848883 (DOI from memory; bibliographic data confirmed via dblp) | `gain_mode: beta` gain function; valid range used here beta > -2 from the formula (Gamma(beta/2+1)), the paper's recommended range not checked | pending (team must confirm) |
| A multi-band spectral subtraction method for enhancing speech corrupted by colored noise (Proc. IEEE ICASSP 2002) | S. Kamath, P. Loizou | 2002 | https://doi.org/10.1109/ICASSP.2002.5745591 (DOI from memory) | idea of frequency-dependent noise over-estimation (`noise_bias_low` / `noise_bias_high`, 1 kHz split) | pending (team must confirm) |
| Pause expander (`pause_thr_db`, `pause_att_db`, `pause_release_frames`) | own design (this repository) | 2026 | - | frame-level extra attenuation when the 300-4000 Hz mean a posteriori SNR is low | n/a (own design) |

## dsp-k extensions (`src/nrc/methods/mmse_lsa.py`, branch exp/dsp-k)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Unbiased MMSE-based noise power estimation with low complexity and low tracking delay (IEEE Trans. Audio, Speech, and Language Processing 20(4), 1383-1393) | T. Gerkmann, R. C. Hendriks | 2012 | https://doi.org/10.1109/TASL.2011.2180896 (DOI from memory) | a posteriori speech presence probability with fixed prior 0.5 and fixed a priori SNR xi_H1 (`noise_est: spp_mean`, `spp_xi_h1_db`); the whole-file iterative SPP-weighted mean is our own variant, not the paper's recursive tracker | pending (team must confirm) |
| Noise power spectral density estimation based on optimal smoothing and minimum statistics (IEEE Trans. Speech and Audio Processing 9(5), 504-512) | R. Martin | 2001 | https://doi.org/10.1109/89.928915 (DOI from memory) | idea of a noise floor from the minimum of the smoothed periodogram over a window (`tv_noise`); the zero-phase smoothing, centred window and upward-only max with the stationary PSD are our own design (no bias compensation from the paper) | pending (team must confirm) |

## dsp-m phase options (`src/nrc/phase.py`, `src/nrc/methods/mmse_lsa.py`, branch exp/dsp-m)

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Signal estimation from modified short-time Fourier transform (IEEE Trans. Acoustics, Speech, and Signal Processing 32(2), 236-243) | D. W. Griffin, J. S. Lim | 1984 | https://doi.org/10.1109/TASSP.1984.1164317 (DOI from memory) | `phase_consistency_iters`: iterations X <- A exp(j angle(STFT(ISTFT(X)))) | pending (team must confirm) |
| Consistent Wiener filtering for audio source separation (IEEE Signal Processing Letters 20(3), 217-220) | J. Le Roux, E. Vincent | 2013 | https://doi.org/10.1109/LSP.2012.2225617 (DOI from memory) | idea of enforcing STFT consistency on a masked spectrogram; our version keeps the noisy phase in high-gain bins (`phase_gain_thr`) and only iterates the phase (not the paper's consistency-penalised Wiener estimate) | pending (team must confirm) |
| STFT phase reconstruction in voiced speech for an improved single-channel speech enhancement (IEEE/ACM Trans. Audio, Speech, and Language Processing 22(12), 1931-1940) | M. Krawczyk, T. Gerkmann | 2014 | https://doi.org/10.1109/TASLP.2014.2354236 (DOI from memory) | `phase_harmonic`: phase propagation along time at harmonic frequencies, window phase offset in neighbouring bins; re-anchoring in high-gain bins, the gain-threshold blend and the f0 jump reset are our own choices | pending (team must confirm) |
| YIN, a fundamental frequency estimator for speech and music (J. Acoust. Soc. Am. 111(4), 1917-1930) | A. de Cheveigne, H. Kawahara | 2002 | https://doi.org/10.1121/1.1458024 (DOI from memory) | `yin_f0` (difference function, cumulative-mean normalisation, absolute threshold, parabolic interpolation); the time re-alignment of each estimate to the frame centre is our own addition | pending (team must confirm) |

## Software

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| SciPy 1.0: fundamental algorithms for scientific computing in Python (Nature Methods 17) | P. Virtanen et al. | 2020 | https://doi.org/10.1038/s41592-019-0686-2 | STFT/ISTFT (`scipy.signal`), exponential integral E1 (`scipy.special.exp1`) | pending (team must confirm) |
| python-pesq 0.0.4 (ITU-T P.862.2 wideband PESQ wrapper) | L. Miao (ludlows) | 2022 | https://github.com/ludlows/PESQ | evaluation metric only | pending (team must confirm) |
