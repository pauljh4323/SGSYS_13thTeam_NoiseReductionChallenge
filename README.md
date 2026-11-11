# 2026-2 신호및시스템 음성 잡음 제거 챌린지 — 13조 (잡음처리반)

단일 채널 음성의 잡음을 전통적 신호 처리만으로 제거하는 알고리즘입니다. 평가 지표는 광대역 PESQ(PESQ-WB, 16 kHz)이며 SNR은 보조 지표입니다.

## 제출 정보

- 챌린지 회차: **1차 챌린지 ("내 목소리를 구해줘!") — 전통적 신호 처리만 허용**
- 제출 알고리즘: 전통적 신호 처리만 사용 (기계학습·딥러닝, 학습된 모델이나 가중치 없음). 파라미터는 train 데이터에서 그리드 탐색으로 정했습니다.
- 제출 버전의 validation 결과: [docs/validation_results.md](docs/validation_results.md)

| 회차 | 제출일 | 태그 | 커밋 | 방법 | Validation PESQ | Validation SNR (dB) |
|---|---|---|---|---|---|---|
| 1 (첫 제출) | [date] | dsp-v7 | cdd4f07 | MMSE-LSA + HRNR + SPP 가중 잡음 PSD | 1.9884 | 12.84 |
| - (대체됨, 미제출) | - | dsp-v6 | d9d1be6 | MMSE-LSA + HRNR + 2대역 잡음 bias (1 kHz 기준 1.25 / 1.5) (`configs/dsp_h.yaml`) | 1.9637 | 12.70 |
| - (대체됨, 미제출) | - | dsp-v5 | fb1d2dc | MMSE-LSA + HRNR, hrnr_rho 0.3 (`configs/dsp_g.yaml`) | 1.9570 | 12.69 |
| - (대체됨, 미제출) | - | dsp-v4 | db1085e | MMSE-LSA + HRNR, 그리드 경계 alpha / xi_min / gmin (`configs/dsp_edge.yaml`) | 1.9497 | 12.72 |
| - (대체됨, 미제출) | - | dsp-v3 | 7080f9b | MMSE-LSA + HRNR + 100 Hz 고역 통과, alpha / xi_min 재조정 (`configs/dsp_retune.yaml`) | 1.9352 | 12.71 |
| - (대체됨, 미제출) | - | dsp-v2 | e68a073 | MMSE-LSA + HRNR + 100 Hz 고역 통과 (`configs/dsp_tune_c.yaml`) | 1.8746 | 12.64 |
| - (대체됨, 미제출) | - | dsp-v1 | af75105 | MMSE-LSA (`configs/mmse_lsa_top1.yaml`) | 1.7505 | 13.56 |

Validation = 공식 validation 세트 (164개 파일). `main.py`가 저장한 16-bit WAV를 `pesq==0.0.4` (광대역, 16 kHz)로 채점한 값입니다.
커밋 열은 제출 알고리즘의 코드 커밋이며, 태그 `dsp-v7`은 `main.py`, `src/`, `configs/`가 동일한 `main` 브랜치 커밋을 가리킵니다.
회차별 기록: [docs/submissions.md](docs/submissions.md).

## 알고리즘 요약

최소 평균 제곱 오차 로그 스펙트럼 진폭 추정기(MMSE-LSA; Ephraim & Malah, 1985)에 2단계 a priori SNR 추정과
고조파 재생(HRNR; Plapous, Marro & Scalart, 2006)을 결합했습니다. 구현: `src/nrc/methods/mmse_lsa.py`,
설정: `configs/submission.yaml` (= `configs/dsp_k.yaml`).

처리 순서: STFT → 잡음 추정 → a priori SNR (DD → TSNR → HRNR) → MMSE-LSA 이득 → ISTFT → 100 Hz 고역 통과 필터

1. **단시간 푸리에 변환(STFT)**: 1024점 Hann 창, hop 256 (16 kHz에서 64 ms / 16 ms)
   (`n_fft: 1024`, `hop: 256`, `window: hann`).
2. **잡음 추정**: 파일에서 에너지가 가장 낮은 10 % 프레임의 평균 파워로 시작하고 (`noise_fraction: 0.10`) 주파수 방향으로
   3개 빈을 평활합니다. 이를 음성 존재 확률(SPP) 가중 파일 전체 평균으로 다시 추정합니다 (`noise_est: spp_mean`;
   Gerkmann & Hendriks, 2012 참고):
   사전 확률 0.5, 고정 a priori SNR xi_H1 = 10 dB (`spp_xi_h1_db: 10`)로 사후 SPP
   P1 = 1 / (1 + (1 + xi_H1) exp(-gamma xi_H1 / (1 + xi_H1)))를 구하고,
   lambda(k) = sum_l (1 - P1) |Y|^2 / sum_l (1 - P1)로 주파수 빈별 잡음 PSD를 갱신하는 과정을 3회 반복합니다 (`spp_iters: 3`; 파일 전체 반복 방식은 자체 변형).
   마지막으로 주파수 의존 잡음 과대추정 계수(bias)를 곱합니다: 1 kHz 아래 1.25, 위 1.5, 200 Hz 폭 raised-cosine으로 연결
   (`noise_bias_low: 1.25`, `noise_bias_high: 1.5`; Kamath & Loizou, 2002의 2대역 과대추정 아이디어).
3. **a priori SNR — 결정 지향(DD)**: 사후 SNR gamma = |Y|^2 / 잡음 PSD. 1단계 a priori SNR은 결정 지향 평활
   (Ephraim & Malah, 1984; `alpha: 0.94`)로 구하고 `xi_min_db: -25`로 하한을 둡니다.
4. **a priori SNR — TSNR → HRNR** (`xi_mode: hrnr`): 2단계(TSNR) xi = |G_dd Y|^2 / 잡음. TSNR 추정 신호를 시간 영역으로
   재합성해 반파 정류로 고조파를 재생하고 다시 분석한 뒤, 최종 a priori SNR
   xi = (rho |G_tsnr Y|^2 + (1 - rho) |S_harmo|^2) / 잡음 을 사용합니다 (`hrnr_rho: 0.3`).
   켑스트럼 평활은 사용하지 않습니다 (`xi_cepstral_smoothing: false`).
5. **MMSE-LSA 이득**: G = xi / (1 + xi) * exp(0.5 * E1(v)), v = xi * gamma / (1 + xi), [`gmin_db: -30` dB, 0 dB]로 제한.
   향상된 진폭 = G * |Y|, 위상은 잡음 섞인 입력의 위상을 그대로 사용합니다.
6. **역 STFT(ISTFT)**: 원래 길이로 복원합니다.
7. **100 Hz 고역 통과 필터**: 4차 Butterworth 100 Hz, 순방향-역방향 적용 (영위상, `highpass_hz: 100`).

`configs/dsp_k.yaml`에는 `noise_bias: 1.25`도 있으나 `noise_bias_low` / `noise_bias_high`가 지정되어 있으므로 위의 2대역 계수가 적용됩니다.

파라미터는 400개 파일의 층화 train 부분집합에서만 단계별 그리드 탐색으로 정했습니다
(`configs/sweeps/dsp_tune_{a,b,c}.yaml`, 재조정 `configs/sweeps/dsp_retune_d{1,2,3}.yaml`, 그리드 경계 확장
`configs/sweeps/dsp_edge_e{1,2}.yaml`, paired 단계 `configs/sweeps/dsp_g_g{1..6}.yaml`, `configs/sweeps/dsp_h_h{1,2,3}.yaml`,
`configs/sweeps/dsp_k_k{1,2,3}.yaml`, `scripts/sweep.py`). 시도했으나 채택하지 않은 것: beta 차수 MMSE 이득
(You, Koh & Rahardja, 2005), 휴지 구간 확장기(pause expander), 상향 전용 국소 최소값 기반 시변 잡음 보정(`tv_noise`).

dsp-v1 대비 PESQ 향상의 대가로 validation SNR이 전체적으로 약 0.85 dB 낮아졌습니다. 입력 SNR 0/5 dB에서는 SNR이 개선되고
10/15 dB에서는 낮아집니다 (고역 통과 필터와 잡음 과대추정이 음성 에너지 일부를 제거하기 때문).

## 설치 방법

Python 3.12, pip만으로 설치할 수 있습니다 (conda 불필요):

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; Linux/macOS는 `source .venv/bin/activate`
pip install -r requirements.txt
```

`requirements.txt`에는 `main.py` 실행에 필요한 고정 버전 의존성만 있습니다 (numpy, scipy, soundfile, pyyaml).
`r2-prep` 브랜치에서는 2차 챌린지용 `torch==2.14.1` (CPU)이 추가되어 있습니다 (아래 "2차 챌린지 (준비 중)" 참고).

conda를 쓰는 경우 (전체 개발 환경, 평가 도구 포함):

```bash
conda env create -f environment.yml
conda activate nrc
```

평가·개발 도구 (PESQ, STOI, pandas, matplotlib, ...)는 `requirements-dev.txt`에 있습니다.

## 실행 방법

1. 잡음이 섞인 WAV 파일을 `input/` 폴더에 넣습니다 (하위 폴더 포함, 폴더 구조는 그대로 유지).
2. 저장소 폴더에서 다음을 실행합니다.

```bash
python main.py
```

3. 처리 결과가 `output/` 폴더에 **같은 파일명**으로 저장됩니다. 출력은 16-bit PCM이며 입력과 **샘플링 주파수, 채널 수, 샘플 수가 같습니다**.

- 다채널 입력은 모노로 평균해 처리한 뒤 결과를 모든 채널에 씁니다.
- 16 kHz가 아닌 입력은 16 kHz로 리샘플링해 처리한 뒤 원래 샘플링 주파수로 되돌립니다.
- 처리할 수 없는 파일은 원본을 그대로 복사합니다.
- CPU만 사용하며 오프라인으로 동작합니다. validation 164개 파일 처리에 CPU 13.7 s 걸렸습니다 (2026-11-11 측정, [docs/submissions.md](docs/submissions.md)).
- 경로를 직접 지정할 수 있습니다: `python main.py --input <noisy dir> --output <out dir>`. 설정 파일은 저장소 폴더 기준으로
  찾으므로 어느 작업 폴더에서 실행해도 되며 절대 경로를 쓰지 않습니다.

## Validation 결과

공식 validation 세트 (164개 파일, 화자 5명), `main.py` 출력 (16-bit WAV), PESQ-WB `pesq==0.0.4` 16 kHz, SNR = 과목 정의.
전체 표와 생성 방법: [docs/validation_results.md](docs/validation_results.md).

**전체**

| 세트 | n | PESQ noisy | PESQ dsp-v7 | ΔPESQ | SNR noisy (dB) | SNR dsp-v7 (dB) | ΔSNR (dB) |
|---|---|---|---|---|---|---|---|
| validation | 164 | 1.3947 | 1.9884 | +0.5937 | 9.63 | 12.84 | +3.20 |

**목표 SNR별**

| 목표 SNR | n | PESQ noisy | PESQ dsp-v7 | ΔPESQ | SNR noisy (dB) | SNR dsp-v7 (dB) | ΔSNR (dB) |
|---|---|---|---|---|---|---|---|
| 0 dB | 15 | 1.0796 | 1.3311 | +0.2515 | 0.00 | 6.86 | +6.86 |
| 5 dB | 38 | 1.2181 | 1.7486 | +0.5305 | 5.00 | 11.45 | +6.45 |
| 10 dB | 55 | 1.3575 | 1.9591 | +0.6016 | 10.00 | 12.90 | +2.90 |
| 15 dB | 56 | 1.6354 | 2.3559 | +0.7206 | 15.00 | 15.32 | +0.32 |

**잡음 난이도별**

| 난이도 | n | PESQ noisy | PESQ dsp-v7 | ΔPESQ | SNR noisy (dB) | SNR dsp-v7 (dB) | ΔSNR (dB) |
|---|---|---|---|---|---|---|---|
| easy | 99 | 1.3112 | 2.0367 | +0.7255 | 8.23 | 12.63 | +4.40 |
| moderate | 41 | 1.5161 | 2.0202 | +0.5041 | 10.85 | 13.46 | +2.61 |
| hard | 24 | 1.5317 | 1.7348 | +0.2031 | 13.33 | 12.61 | -0.72 |

**val_remix (보조 검증)** — validation clean 음성을 다른 validation 잡음과 다시 섞은 328개 파일 (seed 1234).
파라미터 선택에는 사용하지 않는 보조 held-out 검증입니다.

| 세트 | n | PESQ noisy | PESQ dsp-v7 | ΔPESQ | SNR noisy (dB) | SNR dsp-v7 (dB) | ΔSNR (dB) |
|---|---|---|---|---|---|---|---|
| val_remix | 328 | 1.3620 | 1.8942 | +0.5322 | 7.50 | 11.20 | +3.70 |

비교 (validation PESQ / SNR): 처리 전 입력 1.3947 / 9.63 dB, 과목 Wiener 베이스라인 1.5987 / 13.12 dB.

## 2차 챌린지 (준비 중)

2차 챌린지(딥러닝, 2026-11-18 시작)용 제출본을 `r2-prep` 브랜치(태그 `r2-ready`)에 미리 준비해 두었습니다.
2차 규칙은 아직 공지되지 않았으며, 공지 후 확인할 항목은 [docs/round2_plan.md](docs/round2_plan.md)에 있습니다.
위의 1차 챌린지 내용(`dsp-v7`)은 `main` 브랜치 기준이며, 1차 종료 전까지 `main`은 변경하지 않습니다.

### 방법 요약

- 모델: `gru_v5b` — 양방향 GRU(BiGRU) 기반 크기(magnitude) 마스크 추정기, 파라미터 2,373,651개 (2.37M), 가중치 `weights/gru_v5b.pt` (9.5 MB).
- 처리 과정: 잡음 음성 → STFT (n_fft 512, hop 128, periodic Hann) → log1p 크기 + 주파수 bin별 정규화 →
  Conv2d 2층 (주파수 방향 stride 2) → BiGRU 2층 (방향당 hidden 256) → Linear + sigmoid 마스크 →
  마스크 × 잡음 STFT (잡음 위상 사용) → ISTFT (입력과 같은 길이).
- 학습: train split만 사용해 처음부터 학습 (사전학습 모델·외부 데이터 없음). 손실 = 압축 크기 손실 + 다중 해상도 STFT 손실 + SI-SDR. 학습 코드는 `dl-main` 브랜치에 있으며 `r2-prep`에는 추론 코드만 있습니다.
- 구현: `src/nrc/models/gru_mask.py`, `src/nrc/methods/gru_mask.py`. 설정: `configs/submission.yaml` (= `configs/dl/gru_v5b_off.yaml`, 입력 레벨 정규화 끔).

### 설치 방법

Python 3.12, pip만으로 설치합니다. torch는 CPU 빌드를 먼저 설치합니다:

```bash
git checkout r2-prep
python -m venv .venv
.venv/Scripts/activate        # Windows; Linux/macOS는 `source .venv/bin/activate`
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt`에도 `--extra-index-url https://download.pytorch.org/whl/cpu`와 `torch==2.14.1`이 들어 있어
`pip install -r requirements.txt`만 실행해도 됩니다. 새 venv 설치 측정값: 약 193 s, 약 801 MB (Windows, 2026-11-12).

### 실행 방법

1차와 같습니다: `input/`에 WAV 파일을 넣고 `python main.py`를 실행하면 `output/`에 같은 파일명으로 저장됩니다.
GPU 없이 CPU만 사용합니다. validation 164개 파일 처리 시간 18.5–32.1 s (2026-11-12 측정, `scripts/e2e_check.py`).

### Validation 결과

`main.py` 출력 (16-bit WAV), PESQ-WB `pesq==0.0.4` 16 kHz, SNR = 과목 정의. 코드 커밋 32a68f8.

| 세트 | n | PESQ noisy | PESQ gru_v5b | ΔPESQ | SNR noisy (dB) | SNR gru_v5b (dB) | ΔSNR (dB) |
|---|---|---|---|---|---|---|---|
| validation | 164 | 1.3947 | 2.6962 | +1.3015 | 9.63 | 16.81 | +7.18 |
| val_remix | 328 | 1.3620 | 2.5170 | +1.1551 | 7.50 | 15.45 | +7.95 |

**목표 SNR별 (validation)**

| 목표 SNR | n | PESQ noisy | PESQ gru_v5b | ΔPESQ | SNR noisy (dB) | SNR gru_v5b (dB) | ΔSNR (dB) |
|---|---|---|---|---|---|---|---|
| 0 dB | 15 | 1.0796 | 1.9531 | +0.8735 | 0.00 | 10.51 | +10.51 |
| 5 dB | 38 | 1.2181 | 2.4100 | +1.1919 | 5.00 | 14.12 | +9.12 |
| 10 dB | 55 | 1.3575 | 2.7022 | +1.3447 | 10.00 | 16.94 | +6.94 |
| 15 dB | 56 | 1.6354 | 3.0834 | +1.4480 | 15.00 | 20.20 | +5.20 |

**잡음 난이도별 (validation)**

| 난이도 | n | PESQ noisy | PESQ gru_v5b | ΔPESQ | SNR noisy (dB) | SNR gru_v5b (dB) | ΔSNR (dB) |
|---|---|---|---|---|---|---|---|
| easy | 99 | 1.3112 | 2.6302 | +1.3190 | 8.23 | 15.67 | +7.44 |
| moderate | 41 | 1.5161 | 2.7075 | +1.1914 | 10.85 | 17.80 | +6.95 |
| hard | 24 | 1.5317 | 2.9487 | +1.4170 | 13.33 | 19.84 | +6.51 |

## 저장소 구조

- `main.py` — 제출용 실행 파일 (`input/` → `output/`)
- `src/nrc/` — 신호 처리 구현 (STFT, 입출력, 지표, `methods/mmse_lsa.py` 등)
- `configs/` — 방법별 설정 파일 (`submission.yaml` = `dsp_k.yaml`), `configs/sweeps/` 그리드 탐색 설정
- `scripts/` — 평가 (`eval.py`), 그리드 탐색 (`sweep.py`), 점검 (`selfcheck.py`, `e2e_check.py`), 보고서·그림 생성
- `docs/` — validation 결과, 제출 기록, 실험 문서, 참고 문헌, AI 활용 기록, 그림
- `results/` — 실험 결과 기록 (`experiments.csv`)
- `input/`, `output/` — `main.py`의 기본 입력·출력 폴더
- `requirements.txt`, `requirements-dev.txt`, `environment.yml` — 실행용 / 개발용 의존성, conda 환경

개발자용 점검·평가 명령:

```bash
pip install -r requirements-dev.txt
python scripts/selfcheck.py
python scripts/e2e_check.py      # 입출력 형식, 새 clone, 새 pip venv, torch 미사용, 전통 기법 allow-list 점검
python scripts/eval.py --config configs/submission.yaml --split validation
python scripts/report_val.py     # -> docs/validation_results.md
```

챌린지 데이터는 저장소에 포함되어 있지 않습니다 (data/noise_reduction_challenge, `--data-root`로 변경 가능).

## 참고

- AI 활용 내역서 초안: [docs/ai_report/round1.md](docs/ai_report/round1.md) (작업별 기록: [docs/ai_usage_log.md](docs/ai_usage_log.md))
- 참고 문헌: [docs/references.md](docs/references.md)
- 딥러닝 실험 (BiGRU 마스크, 태그 best-v3 ... best-v8)은 [`dl-main`](../../tree/dl-main) 브랜치에 있으며 1차 챌린지 제출 대상이 아닙니다. `main`에는 병합하지 않습니다.
