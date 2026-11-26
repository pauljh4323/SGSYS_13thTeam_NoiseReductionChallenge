# 최종 보고서 개요 (2026-12-14 제출)

형식: A4 약 10쪽, 맑은 고딕 또는 나눔고딕 11 pt. 팀: 13조 잡음처리반.
각 절 아래의 문서·그림을 근거로 씁니다. 브랜치 이름이 없으면 `r2-prep`에 있는 파일입니다.
숫자는 `results/experiments.csv`와 각 문서의 값을 그대로 옮기고, 순위는 `docs/leaderboard.md`에서만 인용합니다.

## 1. 개요 (0.5쪽)
- 과제, 평가 지표 (PESQ-WB 순위, SNR 보조), 숨겨진 테스트 483개 파일 (16 kHz 모노 48000 샘플, train/validation에 없는 잡음 포함).
- 근거: `README.md` 첫 부분, `CLAUDE.md` Rounds.

## 2. 회차별 진행 과정 (실패한 실험 포함, 2쪽)
- 1차 (전통 신호 처리): Wiener → MMSE-LSA → HRNR → 2대역 bias → SPP 가중 잡음 PSD (dsp-v1 … dsp-v7).
  근거: `docs/submissions.md`, `docs/exp_dsp_g.md`, `docs/exp_dsp_h.md`, `docs/exp_dsp_k.md`, `docs/exp_oracle.md`, `docs/exp_dsp_p.md` (`main`),
  `docs/ai_report/round1.md` "시도했으나 채택하지 않은 방법".
- 2차 (딥러닝): `gru_v1` … `gru_v5b` (`dl-main`) → 미분 가능한 PESQ (`exp/r2-dpesq`: `docs/exp_r2_dpesq.md`)
  → `rms_norm` + 동적 혼합 + clean 쪽 증강 (`exp/r2-dmix`: `docs/exp_r2_dmix.md`) → `gru_v8p2z3e`.
- 실패·미채택 실험: `docs/exp_r2_v6.md`, `docs/exp_r2_v7.md`, `docs/exp_r2_mgan.md`, `docs/r2_diag.md` (`exp/r2-v6`, `exp/r2-v7`, `exp/r2-mgan`),
  `docs/exp_r2_ens.md` (`exp/r2-ens`), `docs/exp_r2_soup.md` (`exp/r2-soup`), `docs/exp_r2_mpost.md` (`exp/r2-mpost`),
  `gru_v8p2z4w` (`docs/exp_r2_dmix.md`), 잡음 분석 `docs/r2_noise_analysis.md` (`exp/r2-noise-analysis`).
- 표: 단계별 validation PESQ / SNR (README 2차 챌린지 4·6절의 표).

## 3. 알고리즘 (흐름도·구조도 포함, 1.5쪽)
- 1차: MMSE-LSA + HRNR 처리 순서 (README "알고리즘 요약", `src/nrc/methods/mmse_lsa.py`).
- 2차: STFT → log1p 크기 → Conv2d 2층 → BiGRU 2층 → sigmoid 마스크 → ISTFT, 입력 `rms_norm` (README 2차 3절, `src/nrc/models/gru_mask.py`).
- 그림 (새로 그림): `docs/figures/r2_flow.png` (추론 흐름), `docs/figures/r2_arch.png` (모델 구조, 텐서 크기 표시).

## 4. 데이터와 학습 설정 (외부 데이터 범위 포함, 1.5쪽)
- 데이터: 챌린지 train split만 학습에 사용, validation은 체크포인트 선택, val_remix는 보조 검증 (선택에 사용하지 않음).
- 학습 설정: 11단계 표 (seed, lr, epoch, 손실, 증강) — README 2차 4절, `configs/dl/*.yaml` (`dl-main`, `exp/r2-dpesq`, `exp/r2-dmix`).
- 외부 자료 범위: torch-pesq (학습 손실만), 외부 잡음 코퍼스 DEMAND / ESC-50 / MUSAN은 일반화 평가에만 사용
  (`exp/r2-gen`의 `docs/external_data.md`: 출처, 버전, 라이선스, 분할, train/validation 사용 여부). 제출 모델에는 외부 데이터·사전학습 가중치 없음.

## 5. 결과 (1.5쪽)
- 표 1 (validation, PESQ-WB + SNR): 조교 예시 모델 1.6922 / 13.06 dB, 1차 제출 dsp-v7 1.9884 / 12.84 dB, `gru_v5b` 2.6962 / 16.81 dB,
  주요 2차 실험 (`gru_v8p2`, `gru_v8p2y`, `gru_v8p2z2`, `gru_v8p2z3d`), 제출 `gru_v8p2z3e` + rms_norm 3.6534 / 1.21 dB.
- 표 2: 목표 SNR별·난이도별 (README 2차 6절, `docs/validation_results.md`).
- 표 3: val_remix와 일반화 세트 (G_unseen_test, G_multi_test, G_low; `exp/r2-gen`의 `docs/r2_gen_suite.md`), 공개 사전학습 모델 비교 포함.
- 이전 제출과 비교: `docs/submissions.md`.

## 6. 스펙트로그램과 청취 분석 (1쪽)
- 1차 그림: `docs/figures/dsp_k_speaker007_utt01_seg00{01,05,06,12}.png`, `docs/figures/wiener_baseline_*.png`.
- 2차 그림 (새로 생성): `scripts/plot_spectrograms.py`로 같은 4개 파일에 대해 noisy / clean / `gru_v5b` / `gru_v8p2z3e`.
- 청취 분석: 잔여 잡음, 음성 왜곡, 레벨 변화 (`rms_norm`), 저 SNR·외부 잡음 사례. 근거: `docs/ai_report/round1.md` "스펙트로그램·청취 분석" 형식.

## 7. 한계 (0.5쪽)
- SNR 하락 (PESQ 손실이 레벨·위상을 제약하지 않음), 외부 잡음에서 validation 대비 PESQ 하락 (3.65 → 2.84),
  잡음 위상 사용, 단일 seed, GPU 학습의 비트 단위 비재현성, 일반화 세트 overlap 검사의 하한 성격.

## 8. 최종 커밋·태그 (0.25쪽)
- 1차: tag `dsp-v7` / 제출 태그 `sub-20261116`. 2차: `[태그]` / `[커밋]` (제출 후 `sub-YYYYMMDD` 태그와 함께 기입).

## 9. 참고 문헌 (0.75쪽)
- `docs/references.md` (팀 확인 완료 항목만), torch-pesq, 외부 데이터 코퍼스 (`docs/external_data.md`).
- AI 활용 보고서: `docs/ai_report/round1.md`, `docs/ai_report/round2.md`.
