# Round-2 diagnosis: noise-type-specific processing (branch exp/r2-diag)

Analysis only (2026-11-18): no training, no model / weight / config / submission changes. Methods: gru_v5b_off
(`configs/dl/gru_v5b_off.yaml`, r2-prep) and dsp-v8 (`configs/dsp_p.yaml`, main merged in). Script
`scripts/r2_diag.py` (code at clean hash 3ca9118; stages enhance / blend / features / report), tables regenerated into
`logs/r2_diag_tables.md` and copied below; per-file CSVs in `outputs/r2_diag/` (not in git).

How the scores were produced: both methods were run on validation (164) and val_remix (328) and written as 16-bit wavs
exactly as main.py writes them; blends y = w·gru + (1-w)·dsp were formed from those 16-bit outputs, re-quantised to 16 bit
and scored with `scripts/eval.py`'s own `evaluate_row` (pred-dir path). The end points reproduce the recorded eval.py
results exactly (validation w=0 2.0214 / w=1 2.6962, val_remix w=0 1.9221 / w=1 2.5170; SNR 12.807 / 16.812 / 11.184 /
15.450), so the existing per-file CSVs (outputs/dsp_p, outputs/gru_v5b_off) match the current code. Blend rows are analysis
bounds, not candidates, and were not appended to results/experiments.csv. Noisy PESQ = pesq - delta_pesq from the same rows.

## Answers

**Are noise recordings shared across splits? Yes, entirely.** All 61 validation noise_codes also occur in train (27 codes
are train-only; 0 are validation-only); both collections (NOISEX-92, Nonspeech) and all 23 categories appear in both. Every
validation and val_remix file uses a noise_code present in train, and val_remix takes its noise from validation segments
(100 %). The manifests only give a code, so whether the same *excerpt* of a recording is reused cannot be told from metadata,
but validation is a seen-noise-type test: it does not measure generalisation to new recordings. Whether the hidden test reuses
these codes is unknown. val_remix reuses validation's noise, so it is not independent of validation for noise-dependent
choices (see routing caveat).

**Which categories does gru_v5b handle worst?** Low PESQ and low gain concentrate in two groups: (1) stationary,
broadband / low-frequency NOISEX noises: aircraft_cockpit (n 14/28; val 2.315, val_remix 1.965), factory, broadband_noise,
destroyer_*, hf_channel. Here gru's lead over dsp-v8 is smallest (+0.10..+0.35) and dsp-v8 is a competitive partner.
(2) Rare transient Nonspeech categories (footsteps, tooth_brushing, click, machine_gun, shower; n 1-2 per split), where both
methods barely improve on noisy at low SNR (val_remix tooth_brushing gain +0.14, footsteps +0.27). Group (2) is too small to
estimate reliably. gru wins most clearly on alarm_siren / bell / animal (+1.3..+2.0 over dsp). By difficulty, gru's gain is
lowest on moderate in val_remix (0.91), and by SNR it is lowest at 0 dB (val 1.953, val_remix 1.825).

**How much could per-category routing / blending add? Little: about +0.02 PESQ, an upper bound.**
(a) Per-file oracle max(gru, dsp): +0.007 (val) / +0.009 (val_remix); dsp beats gru on only 5/164 and 19/328 files.
The per-file oracle over the 5 blend weights: +0.051 / +0.055. (b) Hard per-category routing chosen on validation picks gru in
all 23 categories, so its gain is 0. (c) Per-category blend weights chosen on validation pick w = 0.75 in 9 categories (the
stationary NOISEX-type ones plus wind and phone_dialing) and 1.0 elsewhere: +0.024 on validation (in-sample) and +0.023 on
val_remix (vs in-split oracle +0.025). The reverse direction (choose on val_remix, apply to validation) gives +0.022, and 20/23
categories pick the same w. Caveat: val_remix shares validation's noise, so this transfer is optimistic. Every global
blend loses (w 0.75: -0.17), so any gain is category-specific. The mechanism (mixing in 25 % of the dsp output on stationary
noises) suggests the GRU over-suppresses or distorts there. A model-side fix (training emphasis / augmentation on those noises)
is likely worth more than an inference-time router.

**How reliably can the category be predicted from the noisy input?** k-NN (k 5) on the dsp-v8 spp_mean noise-PSD features
(32 log bands + band-energy temporal variance, 64-d) reaches top-1 0.835 / top-3 0.951 on validation and 0.732 / 0.902 on
val_remix. With the log PSD level-normalised (shape only) it reaches 0.890 / 0.976 and 0.863 / 0.957. Nearest centroid is much
weaker (0.42 / 0.63) because categories are multi-modal (several codes per category). These accuracies are inflated by the
code overlap: the k-NN mostly recognises the *same recording* seen in train, not the category. The large stationary
categories (aircraft_cockpit, broadband_noise, destroyer_*, hf_channel, wind) are reliable (≥ 0.85); machine ↔ traffic_vehicle
/ broadband_noise / factory and rare transient categories are the main confusions. Driving rule (c) with the k-NN-predicted
category instead of the label gives +0.016 on validation (rule in-sample) and **-0.001 on val_remix**: routing errors cancel
the whole bound.

**Implication for round 2:** noise-type routing or blending between gru_v5b and dsp-v8 is not worth deploying (bound
≈ +0.02, ≈ 0 with a realistic classifier). The weak spots (stationary NOISEX noises and 0 dB) point to targeted training
(oversampling or augmentation of those categories, low-SNR mixtures), and to a held-out-by-noise_code split to measure
generalisation to unseen recordings.

## 1. Split overlap

| field | train | validation | train only | validation only | both | val_remix codes in train |
|---|---|---|---|---|---|---|
| noise_code | 88 | 61 | 27 | 0 | 61 | 61/61 |
| noise_collection | 2 | 2 | 0 | 0 | 2 | 2/2 |

| noise_category | train files | validation files | val_remix files | train codes | validation codes | val codes also in train | collections |
|---|---|---|---|---|---|---|---|
| aircraft_cockpit | 111 | 14 | 28 | 3 | 3 | 3 | NOISEX-92 |
| alarm_siren | 187 | 23 | 46 | 14 | 9 | 9 | Nonspeech |
| animal | 39 | 2 | 4 | 9 | 2 | 2 | Nonspeech |
| bell | 15 | 3 | 6 | 4 | 3 | 3 | Nonspeech |
| broadband_noise | 73 | 10 | 20 | 2 | 2 | 2 | NOISEX-92 |
| clap | 3 | 1 | 2 | 1 | 1 | 1 | Nonspeech |
| click | 3 | 1 | 2 | 1 | 1 | 1 | Nonspeech |
| destroyer_engine | 36 | 5 | 10 | 1 | 1 | 1 | NOISEX-92 |
| destroyer_operations | 36 | 5 | 10 | 1 | 1 | 1 | NOISEX-92 |
| door_moving | 3 | 1 | 2 | 1 | 1 | 1 | Nonspeech |
| factory | 38 | 5 | 10 | 2 | 2 | 2 | NOISEX-92 |
| footsteps | 8 | 1 | 2 | 2 | 1 | 1 | Nonspeech |
| hf_channel | 37 | 4 | 8 | 1 | 1 | 1 | NOISEX-92 |
| machine | 264 | 34 | 68 | 12 | 12 | 12 | Nonspeech |
| machine_gun | 3 | 1 | 2 | 1 | 1 | 1 | NOISEX-92 |
| military_vehicle | 37 | 5 | 10 | 2 | 2 | 2 | NOISEX-92 |
| phone_dialing | 43 | 4 | 8 | 2 | 1 | 1 | Nonspeech |
| shower | 3 | 1 | 2 | 1 | 1 | 1 | Nonspeech |
| tooth_brushing | 3 | 1 | 2 | 1 | 1 | 1 | Nonspeech |
| traffic_vehicle | 76 | 9 | 18 | 3 | 3 | 3 | Nonspeech |
| vehicle_interior | 19 | 3 | 6 | 1 | 1 | 1 | NOISEX-92 |
| water | 57 | 5 | 10 | 14 | 5 | 5 | Nonspeech |
| wind | 202 | 26 | 52 | 9 | 6 | 6 | Nonspeech |

Validation files whose noise_code also occurs in train: 164/164; val_remix: 328/328.
val_remix noise_source_id taken from validation segments: 100%.

<details><summary>noise_code membership (train / validation)</summary>

| noise_category | noise_collection | noise_code | split |
|---|---|---|---|
| aircraft_cockpit | NOISEX-92 | buccaneer1 | both |
| aircraft_cockpit | NOISEX-92 | buccaneer2 | both |
| aircraft_cockpit | NOISEX-92 | f16 | both |
| alarm_siren | Nonspeech | N030 | train |
| alarm_siren | Nonspeech | N031 | both |
| alarm_siren | Nonspeech | N032 | train |
| alarm_siren | Nonspeech | N033 | both |
| alarm_siren | Nonspeech | N034 | train |
| alarm_siren | Nonspeech | N035 | both |
| alarm_siren | Nonspeech | N036 | both |
| alarm_siren | Nonspeech | N037 | both |
| alarm_siren | Nonspeech | N038 | both |
| alarm_siren | Nonspeech | N039 | train |
| alarm_siren | Nonspeech | N040 | both |
| alarm_siren | Nonspeech | N041 | both |
| alarm_siren | Nonspeech | N042 | both |
| alarm_siren | Nonspeech | N043 | train |
| animal | Nonspeech | N047 | train |
| animal | Nonspeech | N048 | train |
| animal | Nonspeech | N049 | both |
| animal | Nonspeech | N050 | train |
| animal | Nonspeech | N051 | train |
| animal | Nonspeech | N052 | train |
| animal | Nonspeech | N053 | train |
| animal | Nonspeech | N054 | train |
| animal | Nonspeech | N055 | both |
| bell | Nonspeech | N079 | train |
| bell | Nonspeech | N080 | both |
| bell | Nonspeech | N081 | both |
| bell | Nonspeech | N082 | both |
| broadband_noise | NOISEX-92 | pink | both |
| broadband_noise | NOISEX-92 | white | both |
| clap | Nonspeech | N086 | both |
| click | Nonspeech | N088 | both |
| destroyer_engine | NOISEX-92 | destroyerengine | both |
| destroyer_operations | NOISEX-92 | destroyerops | both |
| door_moving | Nonspeech | N098 | both |
| factory | NOISEX-92 | factory1 | both |
| factory | NOISEX-92 | factory2 | both |
| footsteps | Nonspeech | N096 | train |
| footsteps | Nonspeech | N097 | both |
| hf_channel | NOISEX-92 | hfchannel | both |
| machine | Nonspeech | N018 | both |
| machine | Nonspeech | N019 | both |
| machine | Nonspeech | N020 | both |
| machine | Nonspeech | N021 | both |
| machine | Nonspeech | N022 | both |
| machine | Nonspeech | N023 | both |
| machine | Nonspeech | N024 | both |
| machine | Nonspeech | N025 | both |
| machine | Nonspeech | N026 | both |
| machine | Nonspeech | N027 | both |
| machine | Nonspeech | N028 | both |
| machine | Nonspeech | N029 | both |
| machine_gun | NOISEX-92 | machinegun | both |
| military_vehicle | NOISEX-92 | leopard | both |
| military_vehicle | NOISEX-92 | m109 | both |
| phone_dialing | Nonspeech | N099 | both |
| phone_dialing | Nonspeech | N100 | train |
| shower | Nonspeech | N094 | both |
| tooth_brushing | Nonspeech | N095 | both |
| traffic_vehicle | Nonspeech | N044 | both |
| traffic_vehicle | Nonspeech | N045 | both |
| traffic_vehicle | Nonspeech | N046 | both |
| vehicle_interior | NOISEX-92 | volvo | both |
| water | Nonspeech | N056 | train |
| water | Nonspeech | N057 | train |
| water | Nonspeech | N058 | both |
| water | Nonspeech | N059 | both |
| water | Nonspeech | N060 | train |
| water | Nonspeech | N061 | both |
| water | Nonspeech | N062 | train |
| water | Nonspeech | N063 | both |
| water | Nonspeech | N064 | train |
| water | Nonspeech | N065 | train |
| water | Nonspeech | N066 | train |
| water | Nonspeech | N067 | train |
| water | Nonspeech | N068 | both |
| water | Nonspeech | N069 | train |
| wind | Nonspeech | N070 | both |
| wind | Nonspeech | N071 | both |
| wind | Nonspeech | N072 | train |
| wind | Nonspeech | N073 | both |
| wind | Nonspeech | N074 | train |
| wind | Nonspeech | N075 | both |
| wind | Nonspeech | N076 | train |
| wind | Nonspeech | N077 | both |
| wind | Nonspeech | N078 | both |

</details>

## 2. Per-category scores, validation

Overall n=164: noisy 1.3947, dsp_v8 2.0214 (SNR 12.81), gru_v5b 2.6962 (SNR 16.81), gru-dsp +0.6747, gru>dsp files 159/164

| noise_category | n | noisy | dsp_v8 | gru_v5b | gru-dsp | gru gain | dsp gain | gru>dsp files | SNR dsp | SNR gru |
|---|---|---|---|---|---|---|---|---|---|---|
| footsteps | 1 | 1.2168 | 1.2551 | 1.8908 | 0.6357 | 0.6740 | 0.0383 | 1/1 | 7.7948 | 13.9368 |
| hf_channel | 4 | 1.1492 | 1.5943 | 1.9070 | 0.3127 | 0.7578 | 0.4451 | 3/4 | 14.4735 | 15.4432 |
| broadband_noise | 10 | 1.1153 | 1.7865 | 2.0541 | 0.2676 | 0.9389 | 0.6712 | 10/10 | 12.4984 | 14.9464 |
| destroyer_operations | 5 | 1.2859 | 2.0907 | 2.2963 | 0.2056 | 1.0104 | 0.8048 | 5/5 | 11.5442 | 13.3668 |
| aircraft_cockpit | 14 | 1.1731 | 2.0211 | 2.3152 | 0.2940 | 1.1421 | 0.8480 | 14/14 | 12.8142 | 15.3668 |
| water | 5 | 1.2317 | 1.4871 | 2.3315 | 0.8443 | 1.0997 | 0.2554 | 5/5 | 12.0990 | 19.5397 |
| destroyer_engine | 5 | 1.2479 | 1.9888 | 2.3388 | 0.3499 | 1.0908 | 0.7409 | 5/5 | 13.9726 | 15.8194 |
| clap | 1 | 1.3776 | 1.3597 | 2.4194 | 1.0597 | 1.0418 | -0.0179 | 1/1 | 9.3834 | 20.0518 |
| factory | 5 | 1.4133 | 2.0993 | 2.4282 | 0.3288 | 1.0149 | 0.6860 | 5/5 | 13.9469 | 17.2472 |
| wind | 26 | 1.4737 | 2.1687 | 2.5291 | 0.3604 | 1.0554 | 0.6950 | 24/26 | 11.7409 | 14.0808 |
| military_vehicle | 5 | 1.4211 | 2.3093 | 2.5780 | 0.2686 | 1.1569 | 0.8882 | 5/5 | 11.0234 | 14.7882 |
| machine | 34 | 1.4156 | 1.9166 | 2.5912 | 0.6746 | 1.1756 | 0.5010 | 33/34 | 12.9264 | 16.8848 |
| traffic_vehicle | 9 | 1.5252 | 1.8804 | 2.8229 | 0.9425 | 1.2977 | 0.3552 | 8/9 | 13.6394 | 18.2222 |
| click | 1 | 2.1288 | 2.0680 | 2.9557 | 0.8877 | 0.8269 | -0.0608 | 1/1 | 14.8860 | 18.3570 |
| tooth_brushing | 1 | 1.3356 | 1.8163 | 2.9978 | 1.1816 | 1.6622 | 0.4806 | 1/1 | 16.3761 | 19.0834 |
| machine_gun | 1 | 2.0852 | 2.0777 | 3.0114 | 0.9337 | 0.9262 | -0.0075 | 1/1 | 10.5557 | 14.2504 |
| vehicle_interior | 3 | 1.8612 | 2.8175 | 3.0936 | 0.2761 | 1.2324 | 0.9563 | 3/3 | 17.6220 | 19.7609 |
| door_moving | 1 | 1.7771 | 1.8631 | 3.1602 | 1.2972 | 1.3831 | 0.0859 | 1/1 | 16.5068 | 20.1215 |
| shower | 1 | 1.5133 | 2.3631 | 3.2115 | 0.8484 | 1.6983 | 0.8499 | 1/1 | 14.7118 | 24.5416 |
| alarm_siren | 23 | 1.3106 | 2.0219 | 3.4283 | 1.4064 | 2.1177 | 0.7113 | 23/23 | 11.9667 | 19.1199 |
| animal | 2 | 1.4230 | 1.5473 | 3.5372 | 1.9900 | 2.1142 | 0.1243 | 2/2 | 13.5799 | 23.1941 |
| bell | 3 | 1.6753 | 1.7561 | 3.6909 | 1.9349 | 2.0156 | 0.0807 | 3/3 | 14.9514 | 23.1212 |
| phone_dialing | 4 | 2.0864 | 3.6255 | 4.1332 | 0.5078 | 2.0468 | 1.5391 | 4/4 | 16.8393 | 19.4345 |

| noise_difficulty | n | noisy | dsp_v8 | gru_v5b | gru-dsp | gru gain | dsp gain | gru>dsp files | SNR dsp | SNR gru |
|---|---|---|---|---|---|---|---|---|---|---|
| easy | 99 | 1.3112 | 2.0691 | 2.6302 | 0.5611 | 1.3191 | 0.7580 | 96/99 | 12.5511 | 15.6695 |
| hard | 24 | 1.5317 | 1.7720 | 2.9487 | 1.1767 | 1.4170 | 0.2403 | 24/24 | 12.6763 | 19.8411 |
| moderate | 41 | 1.5161 | 2.0523 | 2.7075 | 0.6553 | 1.1914 | 0.5361 | 39/41 | 13.5030 | 17.7992 |

| target_snr_db | n | noisy | dsp_v8 | gru_v5b | gru-dsp | gru gain | dsp gain | gru>dsp files | SNR dsp | SNR gru |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 15 | 1.0796 | 1.3441 | 1.9531 | 0.6090 | 0.8735 | 0.2645 | 15/15 | 6.6929 | 10.5085 |
| 5.0000 | 38 | 1.2181 | 1.7803 | 2.4100 | 0.6297 | 1.1919 | 0.5622 | 38/38 | 11.3745 | 14.1221 |
| 10.0000 | 55 | 1.3575 | 2.0002 | 2.7022 | 0.7019 | 1.3447 | 0.6427 | 53/55 | 12.9003 | 16.9368 |
| 15.0000 | 56 | 1.6354 | 2.3873 | 3.0834 | 0.6962 | 1.4481 | 0.7519 | 53/56 | 15.3262 | 20.2044 |

Weakest 5 for gru_v5b by PESQ: footsteps (n 1, 1.891, gain +0.674, gru-dsp +0.636), hf_channel (n 4, 1.907, gain +0.758, gru-dsp +0.313), broadband_noise (n 10, 2.054, gain +0.939, gru-dsp +0.268), destroyer_operations (n 5, 2.296, gain +1.010, gru-dsp +0.206), aircraft_cockpit (n 14, 2.315, gain +1.142, gru-dsp +0.294)

Weakest 5 for gru_v5b by gain over noisy: footsteps (n 1, 1.891, gain +0.674, gru-dsp +0.636), hf_channel (n 4, 1.907, gain +0.758, gru-dsp +0.313), click (n 1, 2.956, gain +0.827, gru-dsp +0.888), machine_gun (n 1, 3.011, gain +0.926, gru-dsp +0.934), broadband_noise (n 10, 2.054, gain +0.939, gru-dsp +0.268)

## 2. Per-category scores, val_remix

Overall n=328: noisy 1.3620, dsp_v8 1.9221 (SNR 11.18), gru_v5b 2.5170 (SNR 15.45), gru-dsp +0.5949, gru>dsp files 309/328

| noise_category | n | noisy | dsp_v8 | gru_v5b | gru-dsp | gru gain | dsp gain | gru>dsp files | SNR dsp | SNR gru |
|---|---|---|---|---|---|---|---|---|---|---|
| tooth_brushing | 2 | 1.0361 | 1.0363 | 1.1762 | 0.1399 | 0.1401 | 0.0002 | 2/2 | 1.4555 | 12.3626 |
| footsteps | 2 | 1.0731 | 1.0690 | 1.3475 | 0.2786 | 0.2744 | -0.0041 | 2/2 | 0.1943 | 10.0435 |
| aircraft_cockpit | 28 | 1.1396 | 1.6848 | 1.9651 | 0.2803 | 0.8256 | 0.5453 | 28/28 | 11.4738 | 12.9656 |
| factory | 10 | 1.3013 | 1.8586 | 2.0813 | 0.2228 | 0.7801 | 0.5573 | 10/10 | 11.8858 | 13.4893 |
| shower | 2 | 1.0761 | 1.5598 | 2.1460 | 0.5863 | 1.0699 | 0.4837 | 2/2 | 12.4932 | 18.2048 |
| water | 10 | 1.1985 | 1.3630 | 2.1592 | 0.7962 | 0.9607 | 0.1645 | 10/10 | 9.7159 | 15.3218 |
| broadband_noise | 20 | 1.1882 | 1.9745 | 2.2521 | 0.2775 | 1.0639 | 0.7864 | 19/20 | 13.9449 | 16.2551 |
| military_vehicle | 10 | 1.3695 | 2.1503 | 2.2529 | 0.1026 | 0.8834 | 0.7808 | 7/10 | 12.6664 | 13.8286 |
| machine | 68 | 1.3253 | 1.7638 | 2.2925 | 0.5287 | 0.9672 | 0.4384 | 60/68 | 9.6230 | 14.5089 |
| door_moving | 2 | 1.1688 | 1.1823 | 2.3296 | 1.1473 | 1.1608 | 0.0135 | 2/2 | 7.6606 | 15.5362 |
| traffic_vehicle | 18 | 1.4724 | 1.6161 | 2.3634 | 0.7473 | 0.8910 | 0.1437 | 16/18 | 9.0390 | 14.4146 |
| machine_gun | 2 | 1.8264 | 1.8309 | 2.3678 | 0.5369 | 0.5415 | 0.0045 | 2/2 | 9.8289 | 14.7765 |
| destroyer_operations | 10 | 1.3252 | 2.0817 | 2.3704 | 0.2888 | 1.0453 | 0.7565 | 10/10 | 12.9771 | 15.3578 |
| destroyer_engine | 10 | 1.4050 | 2.1775 | 2.3879 | 0.2104 | 0.9829 | 0.7725 | 9/10 | 13.1609 | 14.9998 |
| hf_channel | 8 | 1.2082 | 2.0888 | 2.4249 | 0.3360 | 1.2166 | 0.8806 | 8/8 | 13.6874 | 16.5933 |
| wind | 52 | 1.3957 | 2.1123 | 2.4536 | 0.3413 | 1.0579 | 0.7166 | 50/52 | 12.0345 | 14.0818 |
| click | 2 | 1.6574 | 1.5656 | 2.7305 | 1.1649 | 1.0731 | -0.0918 | 2/2 | 7.0314 | 13.9678 |
| clap | 2 | 1.1608 | 1.1602 | 2.8818 | 1.7216 | 1.7210 | -0.0006 | 2/2 | 9.5700 | 21.8439 |
| bell | 6 | 1.3614 | 1.4108 | 3.2127 | 1.8019 | 1.8513 | 0.0494 | 6/6 | 6.8811 | 19.6367 |
| animal | 4 | 1.3993 | 1.5551 | 3.2177 | 1.6625 | 1.8184 | 0.1558 | 4/4 | 8.7411 | 21.0529 |
| alarm_siren | 46 | 1.3320 | 1.9430 | 3.2946 | 1.3516 | 1.9626 | 0.6110 | 46/46 | 11.6531 | 18.6281 |
| vehicle_interior | 6 | 2.9147 | 3.6761 | 3.7213 | 0.0452 | 0.8066 | 0.7614 | 4/6 | 16.7350 | 23.7618 |
| phone_dialing | 8 | 1.9581 | 3.3724 | 3.9509 | 0.5785 | 1.9928 | 1.4143 | 8/8 | 12.8359 | 15.4299 |

| noise_difficulty | n | noisy | dsp_v8 | gru_v5b | gru-dsp | gru gain | dsp gain | gru>dsp files | SNR dsp | SNR gru |
|---|---|---|---|---|---|---|---|---|---|---|
| easy | 198 | 1.3323 | 2.0431 | 2.5909 | 0.5477 | 1.2586 | 0.7108 | 194/198 | 12.3355 | 15.2910 |
| hard | 48 | 1.3239 | 1.4806 | 2.4733 | 0.9927 | 1.1494 | 0.1567 | 48/48 | 8.0895 | 16.4012 |
| moderate | 82 | 1.4559 | 1.8884 | 2.3644 | 0.4760 | 0.9085 | 0.4325 | 67/82 | 10.2162 | 15.2770 |

| target_snr_db | n | noisy | dsp_v8 | gru_v5b | gru-dsp | gru gain | dsp gain | gru>dsp files | SNR dsp | SNR gru |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 82 | 1.1254 | 1.2976 | 1.8249 | 0.5273 | 0.6995 | 0.1722 | 80/82 | 5.7578 | 11.0497 |
| 5.0000 | 79 | 1.1876 | 1.6381 | 2.3175 | 0.6793 | 1.1299 | 0.4506 | 78/79 | 10.3966 | 13.7240 |
| 10.0000 | 88 | 1.4178 | 2.0939 | 2.7226 | 0.6288 | 1.3048 | 0.6761 | 79/88 | 13.2343 | 17.1362 |
| 15.0000 | 79 | 1.7197 | 2.6631 | 3.2060 | 0.5429 | 1.4863 | 0.9434 | 72/79 | 15.3209 | 19.8649 |

Weakest 5 for gru_v5b by PESQ: tooth_brushing (n 2, 1.176, gain +0.140, gru-dsp +0.140), footsteps (n 2, 1.348, gain +0.274, gru-dsp +0.279), aircraft_cockpit (n 28, 1.965, gain +0.826, gru-dsp +0.280), factory (n 10, 2.081, gain +0.780, gru-dsp +0.223), shower (n 2, 2.146, gain +1.070, gru-dsp +0.586)

Weakest 5 for gru_v5b by gain over noisy: tooth_brushing (n 2, 1.176, gain +0.140, gru-dsp +0.140), footsteps (n 2, 1.348, gain +0.274, gru-dsp +0.279), machine_gun (n 2, 2.368, gain +0.541, gru-dsp +0.537), factory (n 10, 2.081, gain +0.780, gru-dsp +0.223), vehicle_interior (n 6, 3.721, gain +0.807, gru-dsp +0.045)

## 3. Routing / blending bounds

| split | gru_v5b | dsp_v8 | (a) oracle max(gru,dsp) | oracle files dsp | oracle max over 5 w | global w=0 | global w=0.25 | global w=0.5 | global w=0.75 | global w=1 |
|---|---|---|---|---|---|---|---|---|---|---|
| validation | 2.6962 | 2.0214 | 2.7030 | 5 | 2.7472 | 2.0214 | 2.2113 | 2.3685 | 2.5236 | 2.6962 |
| val_remix | 2.5170 | 1.9221 | 2.5255 | 19 | 2.5717 | 1.9221 | 2.0843 | 2.2152 | 2.3425 | 2.5170 |

| split | gru_v5b | (b) route rule (from val) | (b) gain | (c) blend rule (from val) | (c) gain | (c) in-split oracle w per cat | (c) oracle gain |
|---|---|---|---|---|---|---|---|
| validation | 2.6962 | 2.6962 | 0.0000 | 2.7200 | 0.0239 | 2.7200 | 0.0239 |
| val_remix | 2.5170 | 2.5170 | 0.0000 | 2.5403 | 0.0233 | 2.5415 | 0.0245 |

| noise_category | val gru | val dsp | (b) choice | (c) w | val best-w PESQ | val_remix gru | val_remix at rule w |
|---|---|---|---|---|---|---|---|
| aircraft_cockpit | 2.3152 | 2.0211 | gru | 0.7500 | 2.3555 | 1.9651 | 1.9930 |
| alarm_siren | 3.4283 | 2.0219 | gru | 1.0000 | 3.4283 | 3.2946 | 3.2946 |
| animal | 3.5372 | 1.5473 | gru | 1.0000 | 3.5372 | 3.2177 | 3.2177 |
| bell | 3.6909 | 1.7561 | gru | 1.0000 | 3.6909 | 3.2127 | 3.2127 |
| broadband_noise | 2.0541 | 1.7865 | gru | 0.7500 | 2.1330 | 2.2521 | 2.3081 |
| clap | 2.4194 | 1.3597 | gru | 1.0000 | 2.4194 | 2.8818 | 2.8818 |
| click | 2.9557 | 2.0680 | gru | 1.0000 | 2.9557 | 2.7305 | 2.7305 |
| destroyer_engine | 2.3388 | 1.9888 | gru | 0.7500 | 2.3484 | 2.3879 | 2.4552 |
| destroyer_operations | 2.2963 | 2.0907 | gru | 0.7500 | 2.3944 | 2.3704 | 2.4340 |
| door_moving | 3.1602 | 1.8631 | gru | 1.0000 | 3.1602 | 2.3296 | 2.3296 |
| factory | 2.4282 | 2.0993 | gru | 0.7500 | 2.4514 | 2.0813 | 2.1042 |
| footsteps | 1.8908 | 1.2551 | gru | 1.0000 | 1.8908 | 1.3475 | 1.3475 |
| hf_channel | 1.9070 | 1.5943 | gru | 1.0000 | 1.9070 | 2.4249 | 2.4249 |
| machine | 2.5912 | 1.9166 | gru | 1.0000 | 2.5912 | 2.2925 | 2.2925 |
| machine_gun | 3.0114 | 2.0777 | gru | 1.0000 | 3.0114 | 2.3678 | 2.3678 |
| military_vehicle | 2.5780 | 2.3093 | gru | 0.7500 | 2.6475 | 2.2529 | 2.3961 |
| phone_dialing | 4.1332 | 3.6255 | gru | 0.7500 | 4.2126 | 3.9509 | 3.9614 |
| shower | 3.2115 | 2.3631 | gru | 1.0000 | 3.2115 | 2.1460 | 2.1460 |
| tooth_brushing | 2.9978 | 1.8163 | gru | 1.0000 | 2.9978 | 1.1762 | 1.1762 |
| traffic_vehicle | 2.8229 | 1.8804 | gru | 1.0000 | 2.8229 | 2.3634 | 2.3634 |
| vehicle_interior | 3.0936 | 2.8175 | gru | 0.7500 | 3.1332 | 3.7213 | 3.7878 |
| water | 2.3315 | 1.4871 | gru | 1.0000 | 2.3315 | 2.1592 | 2.1592 |
| wind | 2.5291 | 2.1687 | gru | 0.7500 | 2.5723 | 2.4536 | 2.4973 |

Reverse direction (w per category chosen on val_remix, applied to validation): 2.7181 (gain +0.0219); categories where the two splits pick the same w: 20/23; Spearman of per-category gru-dsp delta val vs val_remix: 0.755.

## 4. Category identifiability from noisy input

Classes: 23 noise categories (chance top-1 0.043, top-3 0.130); features z-scored with train statistics; train = 1296 noisy files.

| features | classifier | split | top-1 | top-3 |
|---|---|---|---|---|
| logPSD+var (64-d) | nearest centroid | validation | 0.421 | 0.634 |
| logPSD+var (64-d) | k-NN k=5 | validation | 0.835 | 0.951 |
| logPSD+var (64-d) | nearest centroid | val_remix | 0.421 | 0.622 |
| logPSD+var (64-d) | k-NN k=5 | val_remix | 0.732 | 0.902 |
| level-normalised logPSD+var | nearest centroid | validation | 0.476 | 0.738 |
| level-normalised logPSD+var | k-NN k=5 | validation | 0.890 | 0.976 |
| level-normalised logPSD+var | nearest centroid | val_remix | 0.503 | 0.787 |
| level-normalised logPSD+var | k-NN k=5 | val_remix | 0.863 | 0.957 |
| logPSD only (32-d) | nearest centroid | validation | 0.372 | 0.604 |
| logPSD only (32-d) | k-NN k=5 | validation | 0.817 | 0.921 |
| logPSD only (32-d) | nearest centroid | val_remix | 0.372 | 0.604 |
| logPSD only (32-d) | k-NN k=5 | val_remix | 0.756 | 0.878 |

Per category (logPSD+var features):

| category | nearest centroid / validation n | nearest centroid / validation top1 | nearest centroid / validation top3 | k-NN k=5 / validation top1 | k-NN k=5 / validation top3 | nearest centroid / val_remix top1 | nearest centroid / val_remix top3 | k-NN k=5 / val_remix top1 | k-NN k=5 / val_remix top3 |
|---|---|---|---|---|---|---|---|---|---|
| aircraft_cockpit | 14 | 0.79 | 0.93 | 1.00 | 1.00 | 0.86 | 0.96 | 0.93 | 1.00 |
| alarm_siren | 23 | 0.57 | 0.78 | 0.83 | 0.91 | 0.57 | 0.76 | 0.83 | 0.87 |
| animal | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.50 | 0.00 | 1.00 |
| bell | 3 | 0.67 | 1.00 | 0.33 | 0.67 | 0.33 | 0.67 | 0.33 | 0.50 |
| broadband_noise | 10 | 0.50 | 1.00 | 1.00 | 1.00 | 0.55 | 0.95 | 0.85 | 1.00 |
| clap | 1 | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 1.00 |
| click | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| destroyer_engine | 5 | 0.80 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| destroyer_operations | 5 | 0.80 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | 0.90 | 1.00 |
| door_moving | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| factory | 5 | 0.60 | 1.00 | 0.40 | 1.00 | 0.40 | 0.40 | 0.20 | 0.50 |
| footsteps | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| hf_channel | 4 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| machine | 34 | 0.12 | 0.35 | 0.91 | 0.97 | 0.21 | 0.37 | 0.69 | 0.96 |
| machine_gun | 1 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| military_vehicle | 5 | 0.40 | 0.80 | 0.80 | 1.00 | 0.40 | 0.80 | 0.60 | 0.90 |
| phone_dialing | 4 | 0.75 | 0.75 | 1.00 | 1.00 | 0.38 | 0.62 | 0.88 | 1.00 |
| shower | 1 | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.50 |
| tooth_brushing | 1 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| traffic_vehicle | 9 | 0.33 | 0.44 | 1.00 | 1.00 | 0.22 | 0.44 | 0.78 | 0.83 |
| vehicle_interior | 3 | 0.67 | 0.67 | 0.67 | 1.00 | 0.83 | 1.00 | 0.67 | 1.00 |
| water | 5 | 0.40 | 0.60 | 0.20 | 1.00 | 0.00 | 0.60 | 0.50 | 0.90 |
| wind | 26 | 0.19 | 0.35 | 1.00 | 1.00 | 0.33 | 0.48 | 0.87 | 0.98 |

Confusion summary (logPSD+var):

- nearest centroid, validation: top confusions (true -> pred, files): machine -> military_vehicle 6; broadband_noise -> aircraft_cockpit 5; wind -> hf_channel 5; machine -> factory 4; wind -> military_vehicle 4; alarm_siren -> water 3; machine -> aircraft_cockpit 3; machine -> broadband_noise 3
- k-NN k=5, validation: top confusions (true -> pred, files): alarm_siren -> machine 3; water -> machine 3; factory -> traffic_vehicle 2; alarm_siren -> traffic_vehicle 1; animal -> broadband_noise 1; animal -> water 1; bell -> animal 1; bell -> water 1
- nearest centroid, val_remix: top confusions (true -> pred, files): machine -> aircraft_cockpit 14; machine -> destroyer_operations 14; wind -> military_vehicle 9; broadband_noise -> aircraft_cockpit 8; traffic_vehicle -> aircraft_cockpit 7; wind -> hf_channel 7; factory -> aircraft_cockpit 6; machine -> military_vehicle 6
- k-NN k=5, val_remix: top confusions (true -> pred, files): machine -> traffic_vehicle 9; machine -> broadband_noise 6; factory -> traffic_vehicle 5; alarm_siren -> traffic_vehicle 3; factory -> broadband_noise 3; machine -> water 3; military_vehicle -> machine 3; wind -> machine 3

Blend rule (c) driven by the k-NN predicted category instead of the true label (validation row is in-sample for the rule):

| split | gru_v5b | (c) rule via k-NN predicted category | gain |
|---|---|---|---|
| validation | 2.6962 | 2.7118 | 0.0156 |
| val_remix | 2.5170 | 2.5164 | -0.0006 |
