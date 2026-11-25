# External data (exp/r2-gen, generalisation suite only)

Under `data_ext/` (git-ignored). These files are never used by main.py or by any submitted model. Prepared with
`python -I scripts/gen/prepare_corpora.py`, with the inventory in data_ext/inventory.json.

| corpus | URL | licence | archive(s) | SHA-256 (first 16 hex) | kept | files / hours |
|---|---|---|---|---|---|---|
| DEMAND | https://zenodo.org/records/1227121 (`<ENV>_16k.zip`, 17 envs) | CC BY-SA 3.0 | 17 zips, 1,767,129,401 B | per zip in data_ext/inventory.json | ch01.wav per environment | 17 / 1.42 h |
| ESC-50 | https://github.com/karoldvl/ESC-50 (master zip) | CC BY-NC 3.0 | ESC-50-master.zip, 645,695,005 B | 805afc618aff80ef | audio/ resampled 44.1 -> 16 kHz (resample_poly 160/441) + meta csv | 2000 / 2.78 h |
| MUSAN | https://www.openslr.org/17/ (musan.tar.gz) | CC BY 4.0 (per the archive README; verify) | musan.tar.gz, 11,086,114,085 B (deleted after extraction) | 86d1061c7e15b5c9 | musan/noise/ only | 930 / 6.23 h |

Overlap exclusions: splits/gen/excluded.txt (96 files) and splits/gen/overlap_matches.csv. Splits: splits/gen/summary.csv.
Pretrained repos (data_ext/repos/): MP-SENet @ 89932cf, CMGAN @ 3994600. DeepFilterNet3 comes from pip deepfilternet 0.5.6.
