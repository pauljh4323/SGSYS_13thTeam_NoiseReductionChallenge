# External data (exp/r2-gen generalisation suite; exp/r2-ext z6 training)

Under `data_ext/` (git-ignored), with the inventory in data_ext/inventory.json. main.py never reads these files and
no submitted model so far (up to gru_v8p2z3e) was trained on them.
DEMAND / ESC-50 / MUSAN were prepared with `python -I scripts/gen/prepare_corpora.py`.

| corpus | URL | licence | archive(s) | SHA-256 (first 16 hex) | kept | files / hours |
|---|---|---|---|---|---|---|
| DEMAND | https://zenodo.org/records/1227121 (`<ENV>_16k.zip`, 17 envs) | CC BY-SA 3.0 | 17 zips, 1,767,129,401 B | per zip in data_ext/inventory.json | ch01.wav per environment | 17 / 1.42 h |
| ESC-50 | https://github.com/karoldvl/ESC-50 (master zip) | CC BY-NC 3.0 | ESC-50-master.zip, 645,695,005 B | 805afc618aff80ef | audio/ resampled 44.1 -> 16 kHz (resample_poly 160/441) + meta csv | 2000 / 2.78 h |
| MUSAN | https://www.openslr.org/17/ (musan.tar.gz) | CC BY 4.0 (per the archive README; verify) | musan.tar.gz, 11,086,114,085 B (deleted after extraction) | 86d1061c7e15b5c9 | musan/noise/ only | 930 / 6.23 h |
| LibriSpeech train-clean-100 (Panayotov et al. 2015) | https://www.openslr.org/resources/12/train-clean-100.tar.gz (fetched from the openslr.elda.org / openslr.trmal.net mirrors, 2026-11-26) | CC BY 4.0 | train-clean-100.tar.gz, 6,387,309,499 B; md5 2a93770f6d5c6c964bc36631d331a522 (= published) | d4ddd1d5a6ab3030 (full in inventory.json) | see the keep rule below | 28,539 utts / 251 speakers / 100.59 h in the archive; kept 7,415 / 223 / 22.59 h |

Overlap exclusions: splits/gen/excluded.txt (96 files) and splits/gen/overlap_matches.csv. Splits: splits/gen/summary.csv.
Pretrained repos (data_ext/repos/): MP-SENet @ 89932cf, CMGAN @ 3994600. DeepFilterNet3 comes from pip deepfilternet 0.5.6.

## Use per split (exp/r2-ext, gru_v8p2z6a / gru_v8p2z6b)
- **Training:**
  - the LibriSpeech subset (clean speech);
  - DEMAND / ESC-50 / MUSAN **train_pool** (`splits/gen/*_train_pool.txt` minus excluded.txt) as noise, via the `ext_mix` option of `nrc.data.remix.RemixData`.
- **Model selection:** gen_unseen_dev and gen_multi_dev (built from the gen_dev noise lists), together with challenge validation.
- **Never used in training or selection:** gen_test noise and the G_unseen_test / G_multi_test / G_low sets. They are kept for the final promotion test only. The loader (`nrc.data.extpool.check_noise_rels`) and the package builder assert that no gen_dev, gen_test or excluded file is in the training noise pool.

### LibriSpeech keep rule (`python -I scripts/gen/prepare_librispeech.py`; log in logs/r2_ext_librispeech.txt)
The tarball is streamed (`tarfile` in `r|gz` mode), so nothing is extracted. FLAC files are decoded to int16; the audio is already 16 kHz.
1. Duration ≥ 3 s (27,952 utterances).
2. WADA-SNR (Kim & Stern 2008; `scripts/gen/wada.py`, Monte-Carlo G table) ≥ 30 dB, computed on the non-zero samples (15,230). Exact int16 zeros, about 1 % of samples, would saturate the estimate. The estimator reads low on our checks: speech plus white noise at a true 10 / 20 / 30 dB comes back as about 6.7 / 13.2 / 18.6 dB. That makes the 30 dB threshold conservative.
3. No Haitsma–Kalker fingerprint match (BER < 0.35) against challenge train + validation clean. There were 0 matches; the lowest BER was 0.408.
4. At most 40 utterances per speaker, chosen by highest WADA-SNR (ties broken by id). 68 speakers have fewer than 40 eligible utterances, and 28 have none.
5. Each utterance is stored as at most a 12 s window, with the start seeded by crc32 of the utterance id.

The result is 7,415 utterances, 22.59 h, 10 int16 shards totalling 2.60 GB. The median WADA-SNR is 53.3 dB for the kept utterances and 31.3 dB for the whole archive.

### Drive package `data_ext/pkg/nrc_ext_z6` (`python -I scripts/gen/build_ext_package.py`; log in logs/r2_ext_package.txt)
- `librispeech/`: the shards above, 2.602 GB.
- `noise_pool/`: train_pool noise after `noiselib.active_noise`, 0.702 GB.
  - 1,987 files: DEMAND 12 (1.00 h), ESC-50 1,355 (1.56 h), MUSAN 620 (3.53 h).
  - Status: 1,431 kept, 556 trimmed, 13 silent and dropped.
- `gen/`: gen_unseen_dev and gen_multi_dev, 0.095 GB.
- `MANIFEST.json`: the SHA-256 of all 1,002 files.

Total 3.399 GB. Upload it to `MyDrive/nrc_ext_z6`; the Colab notebook re-checks every hash.
