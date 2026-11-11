# Leaderboard log

Living log of the course leaderboard. Update by adding a new snapshot at the top of "Snapshots" (newest first) and
refreshing "Our status". Standings live only in this file; do not copy them elsewhere.

Sheet (updated continuously by the TA):
https://docs.google.com/spreadsheets/d/1Jebd63s24CKjO6MOe14fef9j_coWdc87011rAgsibtI/edit?gid=1596638433

## Rounds and deadlines

| round | scope | dates | ranking rule |
|---|---|---|---|
| 1 | Traditional DSP only (no ML/DL in the submitted algorithm) | ends 2026-11-17 23:59 KST | at least one submission needed to be ranked |
| 2 | Deep learning | starts 2026-11-18 KST | rules not yet announced (see docs/round2_plan.md) |

Metric: PESQ-WB (pesq==0.0.4, 16 kHz) on the hidden test set.

## Our status (as of 2026-11-11 16:02 KST)

| item | value |
|---|---|
| current best | dsp-v4 (configs/dsp_edge.yaml via configs/submission.yaml) |
| val PESQ / SNR | 1.9497 / 12.72 dB |
| val_remix PESQ / SNR | 1.8449 / 11.03 dB |
| submissions used (round 1) | 0 |
| implied round-1 rank | 2nd of 5 ranked teams (between 4조 2.0975 and 6조 1.8823) |
| gap to #1 | -0.1478 PESQ (1.9497 vs 2.0975) |

Caveat: the implied rank compares our **validation** score with other teams' **hidden-test** scores. The test set may
differ in difficulty; see "Calibration" for the one cross-set reference point we have.

## Snapshots

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

## Notes

- A round-1 team reached 2.0975 with classical DSP, so classical headroom above our current best (1.9497) likely exists.
