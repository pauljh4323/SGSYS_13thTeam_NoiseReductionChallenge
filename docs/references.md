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

## Round 2: BiGRU magnitude mask (`src/nrc/models/gru_mask.py`, `src/nrc/methods/gru_mask.py`, branch r2-prep / dl-main)

Added 2026-11-12 from memory (dl-main had no references.md) — every field must be verified.

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| Learning phrase representations using RNN encoder-decoder for statistical machine translation (EMNLP) | K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, Y. Bengio | 2014 | https://doi.org/10.3115/v1/D14-1179 | GRU cell | pending (team must confirm) |
| Bidirectional recurrent neural networks (IEEE Trans. Signal Processing 45(11)) | M. Schuster, K. K. Paliwal | 1997 | https://doi.org/10.1109/78.650093 | bidirectional recurrence | pending (team must confirm) |
| On training targets for supervised speech separation (IEEE/ACM TASLP 22(12)) | Y. Wang, A. Narayanan, D. Wang | 2014 | https://doi.org/10.1109/TASLP.2014.2352935 | time-frequency magnitude mask on the noisy STFT | pending (team must confirm) |
| SDR — half-baked or well done? (ICASSP) | J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey | 2019 | https://doi.org/10.1109/ICASSP.2019.8683855 | SI-SDR training loss term | pending (team must confirm) |
| Parallel WaveGAN: a fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram (ICASSP) | R. Yamamoto, E. Song, J.-M. Kim | 2020 | https://doi.org/10.1109/ICASSP40776.2020.9053795 | multi-resolution STFT loss term | pending (team must confirm) |
| PyTorch: an imperative style, high-performance deep learning library (NeurIPS) | A. Paszke et al. | 2019 | https://arxiv.org/abs/1912.01703 | STFT/ISTFT, GRU, inference runtime (`torch==2.14.1`) | pending (team must confirm) |

## Software

| title | authors | year | DOI / link | used for | verification |
|---|---|---|---|---|---|
| SciPy 1.0: fundamental algorithms for scientific computing in Python (Nature Methods 17) | P. Virtanen et al. | 2020 | https://doi.org/10.1038/s41592-019-0686-2 | STFT/ISTFT (`scipy.signal`), exponential integral E1 (`scipy.special.exp1`) | pending (team must confirm) |
| ITU-T Recommendation P.862.2: Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs (PESQ wideband extension) | ITU-T | [ ] | https://www.itu.int/rec/T-REC-P.862.2 (from memory — verify) | definition of the evaluation metric PESQ-WB (evaluation only, not used in processing) | from memory — verify; pending (team must confirm) |
| python-pesq 0.0.4 (ITU-T P.862.2 wideband PESQ wrapper) | L. Miao (ludlows) | 2022 | https://github.com/ludlows/PESQ | evaluation metric only | pending (team must confirm) |
| torch-pesq: differentiable PESQ loss for PyTorch (software, version 0.1.2, MIT license) | audiolabs (International Audio Laboratories Erlangen); individual authors: see the repository | 2023 (0.1.2; year from memory) | https://github.com/audiolabs/torch-pesq (PyPI torch-pesq 0.1.2) | gru_v8p1..3 (branch exp/r2-dpesq, scripts/train_dpesq.py, src/nrc/tpesq.py): PesqLoss(1.0, 16 kHz) as a training loss term, 0.1 d_symm + 0.0309 d_asymm, and its MOS mapping for logging; the torchaudio functions it imports are replaced by our exact pure-torch equivalents (training only, not in main.py) | pending (team must confirm) |
| The PESQetarian: On the relevance of Goodhart's law for speech enhancement (Proc. Interspeech 2024) | D. de Oliveira, S. Welker, J. Richter, T. Gerkmann | 2024 | https://doi.org/10.21437/Interspeech.2024-1981 (arXiv 2406.03460; from memory) | motivation for gru_v8p3 (PESQ loss only) and v8p1/p2: a model trained with the torch-pesq loss alone reached PESQ 3.82 on VB-DMD while other metrics did not improve; our objective is PESQ-WB only, with val_remix as the generalisation guard | pending (team must confirm) |
| TorchAudio: Building blocks for audio and speech processing (Proc. ICASSP 2022) | Y.-Y. Yang, M. Hira, Z. Ni et al. | 2022 | https://doi.org/10.1109/ICASSP43922.2022.9747236 (arXiv 2110.15018; from memory, verify) | gru_v8p2 training only: torch-pesq's torchaudio dependencies (`Spectrogram`, `lfilter`, `Resample`) were re-implemented as exact pure-torch equivalents in `src/nrc/tpesq.py` (branch exp/r2-dpesq); not used by main.py | pending (team must confirm) |
| Wavesplit: End-to-End Speech Separation by Speaker Clustering (IEEE/ACM Trans. Audio, Speech, and Language Processing, vol. 29) | N. Zeghidour, D. Grangier | 2021 | https://doi.org/10.1109/TASLP.2021.3099291 (arXiv 2002.08933; from memory, verify) | dynamic mixing for gru_v8p2z2 (branch exp/r2-dmix, src/nrc/data/remix.py p_dm): every training sample is a fresh mixture of a random train clean with the noise of a random other train file (random shift), optionally a second noise, at a random SNR | pending (team must confirm) |
| Audio augmentation for speech recognition (Proc. Interspeech 2015, pp. 3586-3589) | T. Ko, V. Peddinti, D. Povey, S. Khudanpur | 2015 | https://doi.org/10.21437/Interspeech.2015-711 (from memory, verify) | speed perturbation for gru_v8p2z3a/b (branch exp/r2-dmix, src/nrc/data/remix.py p_speed): the clean utterance is resampled by a factor U[0.9, 1.1] (tempo and pitch change, as in the paper's 0.9/1.0/1.1 speed perturbation) before mixing, to add speaker diversity | pending (team must confirm) |
| Cookbook formulae for audio EQ biquad filter coefficients (W3C Audio EQ Cookbook, Working Group Note) | R. Bristow-Johnson | 2005 (W3C note 2021) | https://www.w3.org/TR/audio-eq-cookbook/ (from memory, verify) | low-shelf / high-shelf biquad formulas (shelf slope 1) of the random EQ for gru_v8p2z3b (src/nrc/data/remix.py shelf_sos) | pending (team must confirm) |
| Net2Net: Accelerating Learning via Knowledge Transfer (Proc. ICLR 2016) | T. Chen, I. Goodfellow, J. Shlens | 2016 | https://arxiv.org/abs/1511.05641 (from memory, verify) | function-preserving widening for gru_v8p2z4w (branch exp/r2-dmix, src/nrc/models/net2net.py, scripts/widen_gru.py): gru_v8p2z3d's BiGRU is widened from H 256 to 384 with the old units copied and every weight from a new unit set to 0 (a zero-outgoing variant of Net2WiderNet rather than its unit replication + output split), so training starts from z3d's function | pending (team must confirm) |
| MUSAN: A Music, Speech, and Noise Corpus (arXiv:1510.08484) | D. Snyder, G. Chen, D. Povey | 2015 | https://arxiv.org/abs/1510.08484 (from memory, verify) | exp/r2-gen generalisation suite only: external noise (musan/noise) for G_unseen/G_multi sets; not used by main.py | pending (team must confirm) |
| The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND) (Proc. Meetings on Acoustics 19, 035081) | J. Thiemann, N. Ito, E. Vincent | 2013 | https://doi.org/10.1121/1.4799597 (data: https://zenodo.org/records/1227121; from memory, verify) | exp/r2-gen generalisation suite only: external noise; not used by main.py | pending (team must confirm) |
| ESC: Dataset for Environmental Sound Classification (Proc. ACM Multimedia 2015) | K. J. Piczak | 2015 | https://doi.org/10.1145/2733373.2806390 (from memory, verify) | exp/r2-gen generalisation suite only: external noise (ESC-50); not used by main.py | pending (team must confirm) |
| A Highly Robust Audio Fingerprinting System (Proc. ISMIR 2002) | J. Haitsma, T. Kalker | 2002 | https://ismir2002.ismir.net/proceedings/02-FP04-2.pdf (from memory, verify) | exp/r2-gen overlap check (scripts/gen/noiselib.py fingerprint): 32-bit sub-band energy-difference fingerprints, BER matching | pending (team must confirm) |
| DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement (Proc. Interspeech 2023) | H. Schröter, T. Rosenkranz, A. N. Escalante-B., A. Maier | 2023 | https://arxiv.org/abs/2305.08227 (from memory, verify) | exp/r2-gen zero-shot baseline (DeepFilterNet3, pip deepfilternet 0.5.6); reference only, not in main.py | pending (team must confirm) |
| MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra (Proc. Interspeech 2023) | Y.-X. Lu, Y. Ai, Z.-H. Ling | 2023 | https://arxiv.org/abs/2305.13686 (code github.com/yxlu-0102/MP-SENet; from memory, verify) | exp/r2-gen zero-shot baseline (g_best_vb); reference only, not in main.py | pending (team must confirm) |
| CMGAN: Conformer-based Metric GAN for Speech Enhancement (Proc. Interspeech 2022) | R. Cao, S. Abdulatif, B. Yang | 2022 | https://arxiv.org/abs/2203.15149 (code github.com/ruizhecao96/CMGAN; from memory, verify) | exp/r2-gen zero-shot baseline (best_ckpt/ckpt); reference only, not in main.py | pending (team must confirm) |
