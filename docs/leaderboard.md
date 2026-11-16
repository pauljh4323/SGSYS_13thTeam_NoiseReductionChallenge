# Leaderboard log

Living log of the course leaderboard. Update by adding a new snapshot at the top of "Snapshots" (newest first) and
refreshing "Our status". Standings live only in this file; do not copy them elsewhere.

Sheet (updated continuously by the TA):
https://docs.google.com/spreadsheets/d/1Jebd63s24CKjO6MOe14fef9j_coWdc87011rAgsibtI/edit?gid=1596638433#gid=1596638433

## Rounds and deadlines

| round | scope | dates | ranking rule |
|---|---|---|---|
| 1 | Traditional DSP only (no ML/DL in the submitted algorithm) | ends 2026-11-17 23:59 KST | at least one submission needed to be ranked |
| 2 | Deep-learning speech enhancement (decided; TA notice pending) | starts 2026-11-18 KST | rules not yet announced (see docs/round2_plan.md) |

Metric: PESQ-WB (pesq==0.0.4, 16 kHz) on the hidden test set.

## Our status (as of 2026-11-16 15:09 KST)

| item | value |
|---|---|
| submitted (round 1) | dsp-v7 (configs/dsp_k.yaml via configs/submission.yaml), posted 2026-11-16, tag sub-20261116 |
| leaderboard PESQ | 2.0334 (sheet value 2.03343789695953) |
| val PESQ / SNR | 1.9884 / 12.84 dB |
| val_remix PESQ / SNR | 1.8942 / 11.20 dB |
| submissions used (round 1) | 1 |
| round-1 rank | 5th of 15 submitting teams |
| gap to #1 | -0.108 PESQ (2.0334 vs 7조 2.1413) |

## Snapshots

### 2026-11-16 15:09 KST

Round 1

| rank | team | submissions | PESQ |
|---|---|---|---|
| 1 | 7조 | 1 | 2.1413 |
| 2 | 8조 | 1 | 2.1093 |
| 3 | 4조 | 1 | 2.0975 |
| 4 | 12조 | 1 | 2.0822 |
| 5 | 13조 (us) | 1 | 2.0334 |
| 6 | 10조 | 2 | 2.0195 |
| 7 | 1조 | 1 | 2.0006 |
| 8 | 15조 | 1 | 1.9920 |
| 9 | 2조 | 1 | 1.9645 |
| 10 | 16조 | 1 | 1.8898 |
| 11 | 6조 | 1 | 1.8823 |
| 12 | 5조 | 2 | 1.7982 |
| 13 | 3조 | 1 | 1.7821 |
| 14 | 11조 | 1 | 1.7632 |
| 15 | 9조 | 1 | 1.7133 |
| - | TA baseline (Wiener) | - | 1.5980 |
| - | 14조 | 0 | not submitted |

Round 2: board empty.

### 2026-11-11 16:02 KST

Round 1

| rank | team | submissions | PESQ |
|---|---|---|---|
| 1 | 4조 | 1 | 2.0975 |
| 2 | 6조 | 1 | 1.8823 |
| 3 | 10조 | 1 | 1.8018 |
| 4 | 5조 | 1 | 1.6553 |
| - | TA baseline (Wiener) | - | 1.5980 |
| - | 13조 (us) | 0 | not submitted |

Round 2: board empty.

## Calibration

TA Wiener baseline on the leaderboard: 1.5980. Our reproduction of the same Wiener baseline on validation: 1.5987
(16-bit outputs; results/experiments.csv). Difference -0.0007, so validation and hidden test look comparable for the
baseline; this is one point and need not hold for stronger methods.

Our own submissions (fill in after each confirmed submission):

| version | date | val PESQ | leaderboard PESQ | difference (LB - val) |
|---|---|---|---|---|
| dsp-v7 | 2026-11-16 | 1.9884 | 2.0334 | +0.045 |

The TA Wiener baseline showed no such offset (our val 1.5987 vs leaderboard 1.5980, -0.0007), so the +0.045 for dsp-v7
is method-dependent: the hidden test is not uniformly easier than validation. Treat validation as a ranking proxy, not
as a predictor of the leaderboard value.

## Notes

- 2026-11-11: a round-1 team reached 2.0975 with classical DSP, so classical headroom above our then-best (1.9497) likely existed.
- 2026-11-16: four teams are above our 2.0334 with classical DSP (best 2.1413), so further classical headroom exists.
