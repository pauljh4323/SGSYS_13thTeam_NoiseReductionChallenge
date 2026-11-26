# AI 활용 보고서 — 2차 챌린지 첫 제출 (gru_v8p2z3e + rms_norm)

- 소속: 연세대학교 미래캠퍼스 [ ]학과
- 조: 13조 (잡음처리반)
- 조원: [ ], [ ], [ ]
- 프로젝트명: Speech Noise Reduction Challenge 2 — 딥러닝 기반 단일 채널 음성 잡음 제거
- 제출 버전: tag `[태그]`, commit `[커밋]` (현재 후보 = 태그 `r2-ready`, 브랜치 `r2-prep`), 모델 `gru_v8p2z3e` + 입력 레벨 모드 `rms_norm`
  (`configs/submission.yaml` = `configs/dl/gru_v8p2z3e.yaml`, `weights/gru_v8p2z3e.pt`)
- 제출일: [ ]

---

## Part 1. AI 활용 과정

| 항목 | 내용 |
|---|---|
| 사용 AI 모델 | Claude (Anthropic) — 채팅(전략·실험 설계·결과 분석), Claude Code CLI(코드 작성·학습 설정·평가·커밋) |
| 활용 목적 | [ ] (예: 1차 학습 코드 이식, 미분 가능한 PESQ 손실 도입, torchaudio 대체 구현, 동적 혼합·증강 설계, 승격 규칙·일반화 세트 설계, 제출 규정 점검) |
| 작업 기록 | `docs/ai_usage_log.md` (작업마다 1행: 날짜, 목표, 핵심 프롬프트, AI가 한 일, 검증, 참고문헌) |

## Part 2. 이 모델과 학습 방법을 고른 이유

- 모델: [ ] (예: 양방향 GRU(BiGRU) 크기 마스크 추정기 2.37M 파라미터 — CPU 추론 33 s / 164개 파일, 조교 환경 `torch 2.15.0+cpu`에서 바로 실행, 가중치 9.5 MB)
- 학습 방법: [ ] (예: 회귀 손실로 학습한 `gru_v5b`에서 시작해 평가 지표인 PESQ를 직접 손실로 쓰는 미세조정 → 학습 시 `rms_norm` + 동적 혼합 → clean 음성 쪽 증강(속도 변환, 무작위 EQ))
- 학습 계보: `gru_v5b` → `gru_v8p2` → `gru_v8p2x` → `gru_v8p2y` → `gru_v8p2z2` → `gru_v8p2z3b` → `gru_v8p2z3d` → `gru_v8p2z3e` (README 2차 챌린지 4절)
- 검토했으나 채택하지 않은 방법: [ ] (예: MetricGAN 판별자, 모델 앙상블·가중치 평균(soup), 후처리, hidden 384 확장 `gru_v8p2z4w`; 근거 문서는 각 `exp/r2-*` 브랜치의 `docs/exp_r2_*.md`)

## Part 3. 주요 활용 사례 (질문과 답변)

### 사례 1

- 나의 질문: [ ]
- AI 답변 요약: [ ]
- 나의 검증 및 수정 과정: [ ]

### 사례 2

- 나의 질문: [ ]
- AI 답변 요약: [ ]
- 나의 검증 및 수정 과정: [ ]

### 사례 3

- 나의 질문: [ ]
- AI 답변 요약: [ ]
- 나의 검증 및 수정 과정: [ ]

## Part 4. 원리

- BiGRU 크기 마스크: [ ] (STFT 512/128 → log1p 크기 → Conv2d 2층 → BiGRU 2층 → sigmoid 마스크 × 잡음 STFT, 잡음 위상 사용)
- 미분 가능한 PESQ 손실: [ ] (torch-pesq `PesqLoss`, 0.2 × cmag + 0.32 × PESQ 항; 대칭·비대칭 교란 항의 의미)
- `rms_norm`: [ ] (파일 전체 RMS로 입력을 −22.03 dBFS에 맞춤 → 입력 레벨과 무관한 출력, PESQ는 레벨에 무관)
- PESQ–SNR 교환: [ ] (PESQ 손실은 절대 레벨·위상을 제약하지 않으므로 SNR이 낮아짐)

## Part 5. 우리가 확인하거나 바로잡은 것

- [ ] torch-pesq MOS와 `pesq 0.0.4` 점수의 일치 (validation + val_remix 492개 파일, 파일별 Pearson 0.997, 개선량 Spearman 0.996)
- [ ] torchaudio 함수(`Spectrogram`, `lfilter`, `Resample`)의 순수 torch 대체 구현이 같은 결과를 내는지 (`src/nrc/tpesq.py`, `exp/r2-dpesq`)
- [ ] SNR 하락이 오류가 아니라 PESQ 손실의 예상된 결과인지 (README 4절)
- [ ] 일반화 세트 평가 중 `G_low` 87번 파일에서의 무음 종료: 모델 출력이 아니라 잡음 입력(noisy) 기준 점수에서 `pesq 0.0.4`가 비정상 종료(0xC0000005)한 것으로 확인 (`exp/r2-gen`의 `docs/r2_gen_suite.md`)
- [ ] 제출 형식: `output/`에 입력 WAV 이름만 생기는지, 16 kHz 모노 48000 샘플, 새 clone + 새 pip venv에서 같은 출력 (`scripts/e2e_check.py` 8항목 체크리스트)
- [ ] 그 외 AI 답변 중 틀려서 바로잡은 내용: [ ]

## Part 6. 결과 (validation, 164개 파일, PESQ-WB `pesq==0.0.4`, SNR = 과목 정의)

| 단계 | 모델 | Validation PESQ | Validation SNR (dB) |
|---|---|---|---|
| 조교 예시 모델 | [ ] | 1.6922 | 13.06 |
| 이전 (2차 시작점) | `gru_v5b` | 2.6962 | 16.81 |
| 이후 (제출) | `gru_v8p2z3e` + rms_norm | **3.6534** | 1.21 |

- 보조 검증 val_remix (328개): `gru_v5b` 2.5170 → `gru_v8p2z3e` + rms_norm 3.4961.
- 일반화 세트 (`exp/r2-gen`, 평가 전용): G_unseen_test 2.835, G_multi_test 2.240, G_low 2.311 (`gru_v5b` 1.914 / 1.339 / 1.465).
- 해석: [ ]

## Part 7. 팀이 확인한 출처

| 출처 | 용도 | 확인 (이름 / 날짜) |
|---|---|---|
| torch-pesq (audiolabs, 0.1.2, MIT) | 학습 손실 | [ ] |
| ITU-T P.862 / P.862.2 (PESQ, 광대역) | 평가 지표, 손실의 원리 | [ ] |
| `pesq` 0.0.4 (PyPI) | 평가 | [ ] |
| [ ] (BiGRU 마스크·학습 기법 관련 논문) | [ ] | [ ] |

전체 목록: `docs/references.md` (팀 확인 전에는 pending 표시).
