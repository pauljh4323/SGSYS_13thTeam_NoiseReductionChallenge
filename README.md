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
`r2-prep` 브랜치에서는 2차 챌린지용 `torch==2.15.0` (CPU)이 추가되어 있습니다 (아래 "2차 챌린지" 참고).

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

## 2차 챌린지

2차 챌린지(딥러닝, 2026-11-18 시작) 제출본은 `r2-prep` 브랜치(태그 `r2-ready`)에 있습니다. 위의 1차 챌린지 내용(`dsp-v7`)은 `main` 브랜치 기준입니다.
조교 공지 체크리스트(실행 환경, 패키지, 모델·가중치 경로, 학습 설정, 추론 명령, seed, epoch, 주요 설정, 선택 가중치의 validation 점수, 절대 경로 수정 위치)를 아래 항목 순서대로 정리했습니다.

### 1. 실행 환경

- Python 3.12, CPU만 사용 (GPU 불필요), 추론 중 인터넷 다운로드 없음 (가중치는 저장소에 포함).
- 조교 환경(Python 3.12 conda, `torch 2.15.0+cpu`, `pesq 0.0.4`)과 같은 `torch==2.15.0` (CPU 빌드)로 고정했습니다.
- 확인 (2026-11-18, Windows 11, Python 3.12.16): 새 pip venv의 `torch 2.15.0+cpu`와 개발 환경 `torch 2.14.1+cpu`의 `main.py` 출력이 validation 164개 파일 전부 **바이트 단위로 동일**.

### 2. 패키지 및 설치

```bash
git checkout r2-prep
python -m venv .venv
.venv/Scripts/activate        # Windows; Linux/macOS는 `source .venv/bin/activate`
pip install -r requirements.txt
```

`requirements.txt` (`main.py` 실행용, 버전 고정): `numpy==2.5.4`, `scipy==1.18.1`, `soundfile==0.14.0`, `pyyaml==6.0.3`, `torch==2.15.0`
(`--extra-index-url https://download.pytorch.org/whl/cpu`, Windows/Linux에서는 `2.15.0+cpu` 설치, macOS는 PyPI 휠로 대체).
채점용 `pesq==0.0.4`와 평가 도구는 `requirements-dev.txt`, 학습 환경은 `dl-main`의 `environment.yml`에 있습니다 (추론에는 불필요).

### 3. 모델 및 가중치 경로

- 모델: `gru_v8p2y` — `gru_v5b`와 같은 구조(Conv2d 전단 + 양방향 GRU(BiGRU) 크기(magnitude) 마스크 추정기, 파라미터 2,373,651개 (2.37M))이며,
  `gru_v5b` 가중치를 미분 가능한 PESQ 손실로 미세조정(fine-tuning)한 `gru_v8p2` → `gru_v8p2x`를 같은 손실과 더 넓은 입력 이득 증강(U(−40, +6) dB)으로 더 학습한 가중치입니다 (아래 4절).
- 가중치: `weights/gru_v8p2y.pt` (9.5 MB). 텐서와 기본형(str/int/float/dict)만 담긴 체크포인트이며
  `torch.load(path, map_location="cpu", weights_only=True)`로 읽습니다 (`src/nrc/models/gru_mask.py`의 `load_weights`; `torch 2.15.0+cpu`에서 확인).
- 추론 설정: `configs/submission.yaml` (= `configs/dl/gru_v8p2y_rmsn.yaml`). 가중치 경로는 저장소 폴더 기준 상대 경로입니다.
- **입력 레벨 모드 `rms_norm`** (`level_norm: true`, `level_mode: rms_norm`, `target_rms: 0.079130`):
  파일 전체 RMS로 이득 g = `target_rms` / max(RMS(noisy), 1e-6)를 구하고, g × noisy를 모델에 넣습니다.
  출력은 정규화된 레벨 그대로 저장합니다 (역이득 없음). 출력 피크가 0.99를 넘으면 파일 전체를 피크 0.99로 줄입니다.
  `target_rms` 0.079130 (−22.03 dBFS)은 train noisy 1296개 파일의 파일 전체 RMS 중앙값입니다 (`scripts/rms_target.py`, 한 번 계산해 설정에 저장).
  PESQ는 레벨에 무관하고 입출력 규칙은 샘플링 주파수·채널 수·샘플 수만 고정하므로, 출력 레벨은 자유롭게 정할 수 있습니다.
  이 모드에서는 모델 입력이 입력 레벨과 무관해집니다. 그래서 입력 이득 −30/−20/−10/0 dB에서 모두 같은 PESQ (3.508)가 나옵니다.
  모드를 끈 `gru_v8p2y`는 −30 dB에서 2.607로 떨어졌습니다 (넓은 이득 증강만으로는 모델 자체가 레벨에 견고해지지 않음).
- 예비(fallback): 직전 제출 `gru_v8p2x` + `rms_norm` (`configs/dl/gru_v8p2x_rmsn.yaml`, `weights/gru_v8p2x.pt`, validation PESQ 3.4867 / SNR 2.24 dB),
  `gru_v8p2` (`configs/dl/gru_v8p2.yaml`, `weights/gru_v8p2.pt`, 레벨 정규화 끔, validation PESQ 3.3983 / SNR 4.48 dB),
  `gru_v5b` (`configs/dl/gru_v5b_off.yaml`, `weights/gru_v5b.pt`, validation PESQ 2.6962 / SNR 16.81 dB).
  되돌릴 때는 해당 설정 파일의 `method`/`params`를 `configs/submission.yaml`에 그대로 복사합니다.
- 추론 코드는 `rms_norm` 모드 추가 외에는 `gru_v5b`와 같습니다. 레벨 정규화를 끈 경우의 출력은 바이트 단위로 동일함을 확인했습니다.
- 구현: `src/nrc/models/gru_mask.py` (모델), `src/nrc/methods/gru_mask.py` (추론), `main.py` (입출력).
- 처리 과정: 잡음 음성 → STFT (n_fft 512, hop 128, periodic Hann) → log1p 크기 + 주파수 bin별 정규화 →
  Conv2d 2층 (16채널, 3x3, 주파수 방향 stride 2, GroupNorm) → BiGRU 2층 (방향당 hidden 256) → Linear + sigmoid 마스크 →
  마스크 × 잡음 STFT (잡음 위상 사용) → ISTFT (입력과 같은 길이).

### 4. 학습 설정 (seed, epoch, 주요 설정)

학습과 추론은 분리되어 있습니다. 학습 코드(`scripts/train.py`, `src/nrc/losses.py`, `src/nrc/data/`, `notebooks/colab_train.ipynb`)는 `dl-main` 브랜치에 있고,
`r2-prep`에는 추론 코드와 학습된 가중치만 있습니다. 학습 데이터는 챌린지 train split만 사용했습니다 (사전학습 모델·외부 데이터 없음,
validation은 체크포인트 선택에만 사용, clean 음성은 학습 목표·평가에만 사용). 학습 환경: Google Colab T4 GPU.

제출 가중치 `gru_v8p2y`는 7단계로 학습했습니다.
1–4단계는 `gru_v5b` 학습(`dl-main`)이고, 5–7단계는 미분 가능한 PESQ 미세조정(`exp/r2-dpesq`)입니다.
각 단계는 이전 단계의 best 가중치에서 시작합니다 (가중치 + 정규화 버퍼, optimizer/scheduler는 새로 생성).

| 단계 | 설정 파일 (`dl-main`) | 시작 가중치 | 손실 | lr | 최대 epoch | 실제 학습 epoch | 선택 epoch (val PESQ) |
|---|---|---|---|---|---|---|---|
| 1 | `configs/dl/gru_v2b.yaml` | 무작위 초기화 | 압축 크기 손실 (cmag) | 1e-3 | 150 | 150 | 148 (2.5051) |
| 2 | `configs/dl/gru_v3b.yaml` | `gru_v2b.pt` | cmag + 0.5 × MR-STFT | 3e-4 | 100 | 92 (조기 종료) | 76 (2.5885) |
| 3 | `configs/dl/gru_v4b.yaml` | `gru_v3b.pt` | cmag + 0.5 × MR-STFT + 0.1 × (−SI-SDR) | 2e-4 | 100 | 64 (조기 종료) | 48 (2.6796) |
| 4 | `configs/dl/gru_v5b.yaml` | `gru_v4b.pt` | cmag + 0.5 × MR-STFT + 0.3 × (−SI-SDR) | 2e-4 | 100 | 24 (조기 종료) | 8 (2.6962) |
| 5 | `configs/dl/gru_v8p2.yaml` (`exp/r2-dpesq`) | `gru_v5b.pt` | 0.2 × cmag + 0.32 × PESQ 손실 (torch-pesq) | 5e-5 (고정) | 100 | 100 | 99 (3.3983) |
| 6 | `configs/dl/gru_v8p2x.yaml` (`exp/r2-dpesq`) | `gru_v8p2.pt` | 5단계와 같음 | 5e-5 (고정) | 200 | 200 | 194 (3.4875) |
| 7 | `configs/dl/gru_v8p2y.yaml` (`exp/r2-dpesq`) | `gru_v8p2x.pt` | 5단계와 같음 (입력 이득 증강만 변경) | 5e-5 (고정) | 200 | 183 (조기 종료) | **153 (3.5081)** |

공통 설정: seed 0 (모든 단계, `numpy.random.default_rng(0)` + `torch.manual_seed(0)`), batch size 16 (3 s 구간), Adam, grad clip 5.0,
크기 압축 지수 0.3, MR-STFT 해상도 256/512/1024 (hop 64/128/256), 2 epoch마다 validation 전체 PESQ로 평가,
ReduceLROnPlateau (factor 0.5, patience 3회, 최소 lr 1e-5), 조기 종료 = 8회 연속(16 epoch) validation PESQ 향상 없음 (2–4단계).
데이터 증강 (train 쌍 재조합): 확률 0.5로 원본 쌍, 나머지는 임의의 train clean + 임의의 train 잡음(noisy − clean, 무작위 순환 이동·부호)을 SNR U(−5, 20) dB로 혼합;
입력 이득 U(−30, +6) dB (1단계, 모든 샘플) / 2–4단계는 확률 0.5로 U(−30, +6) dB, 아니면 U(−6, +3) dB; clean 목표에 같은 이득, 피크 0.99 제한.
5–6단계도 4단계와 같은 데이터 증강(같은 재조합 배치 흐름)을 씁니다. 7단계는 입력 이득만 모든 샘플(원본 쌍 포함)에 U(−40, +6) dB 하나로 바꿨습니다 (clean·잡음 같은 이득, 피크 0.99 제한).
선택 가중치까지의 누적 학습량: 148 + 76 + 48 + 8 + 99 + 194 + 153 = 726 epoch. GPU 학습은 cuDNN 비결정성 때문에 같은 seed로도 비트 단위 재현은 보장되지 않습니다.

**5–7단계: 미분 가능한 PESQ 미세조정 (`gru_v8p2`, `gru_v8p2x`, `gru_v8p2y`)**

- 방법: 평가 지표 PESQ(ITU-T P.862)의 청각 모델(Bark 대역 변환, 음량 압축, 대칭·비대칭 disturbance)을 미분 가능하게 구현한
  torch-pesq (audiolabs, 0.1.2, MIT 라이선스)의 `PesqLoss`를 학습 손실로 사용했습니다. torch-pesq가 쓰는 torchaudio 함수
  (`Spectrogram`, `lfilter`, `Resample`)는 `torch 2.15.0`용 torchaudio 빌드가 없어 같은 결과를 내는 순수 torch 구현(`src/nrc/tpesq.py`, `exp/r2-dpesq`)으로 바꿨습니다.
  학습 전에 torch-pesq MOS와 `pesq 0.0.4` 점수의 일치를 확인했습니다 (validation + val_remix 492개 파일, 파일별 Pearson 0.997, 개선량 Spearman 0.996).
- 손실: 0.2 × cmag + 0.32 × (0.1 × d_symm + 0.0309 × d_asymm). 0.32는 `gru_v5b` 초기 가중치에서 PESQ 항과 기존 회귀 손실의
  파라미터 기울기 크기가 같아지도록 정한 값입니다 (8개 배치 평균 비율 0.3186).
- 설정: 시작 가중치 `weights/gru_v5b.pt` (optimizer 새로 생성), Adam lr 5e-5 (고정, scheduler 없음), grad clip 5.0, batch 16, seed 0,
  최대 100 epoch, 매 epoch validation 전체 PESQ(`pesq 0.0.4`, 16-bit 출력)로 best 선택, 조기 종료 20 epoch (발동하지 않음, 100 epoch 완료, best = 99 epoch).
  Colab T4 GPU에서 epoch당 약 9 s + 검증 약 19 s.
- **SNR과 PESQ의 교환**: PESQ 손실은 출력의 절대 레벨과 위상을 제약하지 않습니다. 그래서 PESQ는 크게 오르지만 과목 정의 SNR은
  `gru_v5b` 16.81 dB에서 4.48 dB로 떨어집니다 (입력 9.63 dB보다도 낮음). 대회 목표가 PESQ-WB이므로 의도적으로 SNR을 PESQ와 바꿨습니다.
- 일반화 확인 (5단계): 체크포인트 선택에 쓰지 않은 val_remix에서도 `gru_v5b` 대비 +0.721 (z 35.1)을 얻었습니다.
- 6단계 (`gru_v8p2x`): `gru_v8p2`에서 시작해(optimizer 새로 생성) 같은 손실·설정으로 최대 200 epoch, 조기 종료 30 epoch로 학습했습니다.
  조기 종료는 발동하지 않았고 best는 194 epoch입니다.
  - 5단계 대비: val_remix +0.107 (z 12.1), validation +0.089.
  - SNR은 더 떨어졌습니다 (레벨 정규화 끔 2.91 dB, `rms_norm` 2.24 dB).
  - 레벨 정규화를 끄면 −30 dB 입력에서 2.640으로 떨어집니다. 그래서 3절의 `rms_norm` 모드와 함께 사용합니다.
- 7단계 (`gru_v8p2y`): `gru_v8p2x`에서 시작해(optimizer 새로 생성) 같은 손실로 최대 200 epoch, 조기 종료 30 epoch로 학습했습니다.
  입력 이득 증강만 U(−40, +6) dB (모든 샘플)로 넓혔습니다. 183 epoch에서 조기 종료했고 best는 153 epoch입니다 (train64 4.0666, 차이 0.56으로 6단계와 같음).
  - 6단계 + `rms_norm` 대비 (둘 다 `rms_norm`): validation +0.0210 (z 5.6), val_remix +0.0180 (z 5.7).
  - SNR은 조금 더 떨어졌습니다 (`rms_norm` 1.88 dB). 레벨 정규화를 끄면 −30 dB에서 2.607이므로 계속 `rms_norm`과 함께 사용합니다.
- 학습 재현 (`exp/r2-dpesq` 브랜치): `pip install --no-deps -r requirements-dpesq.txt` (torch-pesq 0.1.2, torchtyping 0.1.5, typeguard 2.13.3; 학습 전용,
  `requirements.txt`에는 없음) 후 `python scripts/train_dpesq.py --config configs/dl/gru_v8p2.yaml --data-root <데이터 폴더>`,
  `cp outputs/ckpt/gru_v8p2/best.pt weights/gru_v8p2.pt`. 6단계는 이어서 `--config configs/dl/gru_v8p2x.yaml`, `cp outputs/ckpt/gru_v8p2x/best.pt weights/gru_v8p2x.pt`, 7단계는 `--config configs/dl/gru_v8p2y.yaml`, `cp outputs/ckpt/gru_v8p2y/best.pt weights/gru_v8p2y.pt`입니다.

학습 재현 (`dl-main` 브랜치, 단계 순서대로, 각 단계 best 가중치를 `weights/<이름>.pt`로 복사):

```bash
git checkout dl-main
conda env create -f environment.yml && conda activate nrc   # Colab: pip install pesq==0.0.4 pystoi soundfile pyyaml
python scripts/train.py --config configs/dl/gru_v2b.yaml --data-root <데이터 폴더>
cp outputs/ckpt/gru_v2b/best.pt weights/gru_v2b.pt     # 이어서 gru_v3b, gru_v4b, gru_v5b
```

Colab 노트북 실행 순서 (`notebooks/colab_train.ipynb`): 런타임을 T4 GPU로 변경 → 셀 1 (Drive 마운트) → 셀 2 (저장소 clone, 시작 가중치 확인) →
셀 3 (pip 설치) → 셀 4 (데이터 복사) → 셀 5 (학습, 끊기면 다시 실행하면 `last.pt`에서 이어서 학습) → 셀 6 (best epoch / val PESQ 출력).
5단계는 `exp/r2-dpesq` 브랜치의 같은 노트북을 같은 순서로 실행합니다 (셀 3에서 torch-pesq 설치·동작 확인, 셀 5의 `CONFIGS`에서 학습할 설정 선택, 셀 6 학습, 셀 7 best epoch / val PESQ 출력).

### 5. 추론 명령

```bash
python main.py                                   # input/ -> output/
python main.py --input <noisy dir> --output <out dir>
```

- `input/`의 WAV 파일을 처리해 `output/` 바로 아래에 **같은 파일명**으로 저장합니다 (16 kHz 모노 입력 → 16 kHz 모노 16-bit PCM, 샘플 수 동일).
  `output/`에는 입력 WAV와 같은 이름의 파일만 생기며 로그 등 다른 파일은 쓰지 않습니다 (진행 정보는 화면에만 출력).
- 16 kHz가 아닌 입력·다채널 입력도 처리하며 입력과 같은 샘플링 주파수, 채널 수, 샘플 수로 저장합니다. 처리에 실패한 파일은 원본을 복사합니다.
- 처리 시간: validation 164개 파일 CPU 18.3 s (새 pip venv, `torch 2.15.0+cpu`), 개발 환경 27.9 s; val_remix 328개 56.8 s (2026-11-23, Windows 11).

### 6. 선택 가중치의 validation 점수

`main.py` 출력 (16-bit WAV), PESQ-WB `pesq==0.0.4` 16 kHz, SNR = 과목 정의. 코드 커밋 915a234 (`torch 2.15.0` 새 venv 출력도 바이트 단위로 동일).
val_remix = validation 쌍만으로 만든 보조 검증 세트 (잡음 재조합, 328개, `scripts/make_val_remix.py`); 체크포인트 선택에는 쓰지 않았습니다.
**SNR은 PESQ와 의도적으로 교환했습니다** (4절): 목표가 PESQ-WB이므로 SNR이 입력보다 낮아도 PESQ가 가장 높은 가중치를 골랐습니다.
`rms_norm`은 출력을 입력 레벨로 되돌리지 않으므로 SNR이 더 낮아집니다.

| 세트 | n | PESQ noisy | PESQ gru_v5b (예비) | PESQ gru_v8p2 (예비) | PESQ gru_v8p2x + rms_norm (예비) | PESQ gru_v8p2y + rms_norm | ΔPESQ (vs noisy) | SNR noisy (dB) | SNR gru_v8p2x + rms_norm (dB) | SNR gru_v8p2y + rms_norm (dB) |
|---|---|---|---|---|---|---|---|---|---|---|
| validation | 164 | 1.3947 | 2.6962 | 3.3983 | 3.4867 | **3.5077** | +2.1130 | 9.63 | 2.24 | **1.88** |
| val_remix | 328 | 1.3620 | 2.5170 | 3.2378 | 3.3453 | 3.3633 | +2.0013 | 7.50 | 2.09 | 1.79 |

**목표 SNR별 (validation)**

| 목표 SNR | n | PESQ noisy | PESQ gru_v8p2y + rms_norm | ΔPESQ | SNR noisy (dB) | SNR gru_v8p2y + rms_norm (dB) | ΔSNR (dB) |
|---|---|---|---|---|---|---|---|
| 0 dB | 15 | 1.0796 | 2.9036 | +1.8240 | 0.00 | 1.42 | +1.42 |
| 5 dB | 38 | 1.2181 | 3.2814 | +2.0633 | 5.00 | 2.05 | −2.95 |
| 10 dB | 55 | 1.3575 | 3.5552 | +2.1977 | 10.00 | 1.91 | −8.09 |
| 15 dB | 56 | 1.6354 | 3.7763 | +2.1409 | 15.00 | 1.86 | −13.14 |

**잡음 난이도별 (validation)**

| 난이도 | n | PESQ noisy | PESQ gru_v8p2y + rms_norm | ΔPESQ | SNR noisy (dB) | SNR gru_v8p2y + rms_norm (dB) | ΔSNR (dB) |
|---|---|---|---|---|---|---|---|
| easy | 99 | 1.3112 | 3.4173 | +2.1061 | 8.23 | 1.95 | −6.28 |
| moderate | 41 | 1.5161 | 3.6046 | +2.0885 | 10.85 | 1.55 | −9.30 |
| hard | 24 | 1.5317 | 3.7147 | +2.1830 | 13.33 | 2.19 | −11.14 |

입력 레벨 견고성 (validation, 입력 이득 −30/−20/−10/0 dB):
- gru_v8p2y + rms_norm: 3.508/3.508/3.508/3.508
- gru_v8p2y, 레벨 정규화 끔: 2.607/3.181/3.449/3.508
- gru_v8p2x + rms_norm: 3.487/3.487/3.487/3.487
- gru_v8p2x, 레벨 정규화 끔: 2.640/3.194/3.437/3.488
- gru_v8p2: 2.736/3.203/3.378/3.398
- gru_v5b: 2.614/2.670/2.684/2.696

선택 규칙: 다음 세 조건을 모두 만족하는 (모델, 레벨 모드) 조합 중 validation PESQ가 가장 높은 것을 골랐습니다.
- validation PESQ ≥ 직전 제출 (`gru_v8p2x` + rms_norm 3.4867)
- val_remix에서 직전 제출 대비 대응 차이 > 0, z ≥ 2 (`gru_v8p2y` + rms_norm: +0.0180, z 5.7)
- 모든 입력 레벨에서 직전 제출 − 0.02 이상

### 7. 절대 경로 수정 위치

- 추론 (`main.py`): 절대 경로를 쓰지 않습니다. 입출력 폴더는 `--input` / `--output` (기본값 `input/`, `output/`, 작업 폴더 기준)으로 바꾸고,
  가중치 경로는 `configs/submission.yaml`의 `params.weights` (저장소 폴더 기준 상대 경로)에서 바꿉니다.
- 학습 (`dl-main`): 데이터 폴더는 `scripts/train.py --data-root` 또는 환경 변수 `NRC_DATA_ROOT` (기본값 `data/noise_reduction_challenge`),
  체크포인트 폴더는 `--ckpt-dir` (기본값 `outputs/ckpt/<설정 이름>`).
  Colab 노트북의 절대 경로는 셀 2 (`/content/repo`), 셀 4 (`/content/drive/MyDrive/noise_reduction_challenge`, `/content/data`),
  셀 5 (`CKPT_ROOT = /content/drive/MyDrive/nrc_ckpt`, `--data-root`)에서 바꿉니다.

## 저장소 구조

- `main.py` — 제출용 실행 파일 (`input/` → `output/`)
- `src/nrc/` — 신호 처리 구현 (STFT, 입출력, 지표, `methods/mmse_lsa.py` 등)
- `configs/` — 방법별 설정 파일 (`main`: `submission.yaml` = `dsp_k.yaml`; `r2-prep`: `submission.yaml` = `dl/gru_v8p2y_rmsn.yaml`, 예비 `dl/gru_v8p2x_rmsn.yaml`, `dl/gru_v8p2.yaml`, `dl/gru_v5b_off.yaml`), `configs/sweeps/` 그리드 탐색 설정
- `weights/` — 2차 챌린지 학습 가중치 (`r2-prep`: 제출 `gru_v8p2y.pt`, 예비 `gru_v8p2x.pt`, `gru_v8p2.pt`, `gru_v5b.pt`)
- `scripts/` — 평가 (`eval.py`), 그리드 탐색 (`sweep.py`), 점검 (`selfcheck.py`, `e2e_check.py`), 보고서·그림 생성
- `docs/` — validation 결과, 제출 기록, 실험 문서, 참고 문헌, AI 활용 기록, 그림
- `results/` — 실험 결과 기록 (`experiments.csv`)
- `input/`, `output/` — `main.py`의 기본 입력·출력 폴더
- `requirements.txt`, `requirements-dev.txt`, `environment.yml` — 실행용 / 개발용 의존성, conda 환경

개발자용 점검·평가 명령:

```bash
pip install -r requirements-dev.txt
python scripts/selfcheck.py
python scripts/e2e_check.py      # 입출력 형식, output/ 파일 목록 = 입력 WAV 이름, 새 clone, 새 pip venv, 방법 allow-list (1차: torch 미사용)
python scripts/eval.py --config configs/submission.yaml --split validation
python scripts/report_val.py     # -> docs/validation_results.md
```

챌린지 데이터는 저장소에 포함되어 있지 않습니다 (data/noise_reduction_challenge, `--data-root`로 변경 가능).

## 참고

- AI 활용 내역서 초안: [docs/ai_report/round1.md](docs/ai_report/round1.md) (작업별 기록: [docs/ai_usage_log.md](docs/ai_usage_log.md))
- 참고 문헌: [docs/references.md](docs/references.md)
- 딥러닝 실험 (BiGRU 마스크, 태그 best-v3 ... best-v8)은 [`dl-main`](../../tree/dl-main) 브랜치에 있으며 1차 챌린지 제출 대상이 아닙니다. `main`에는 병합하지 않습니다.
