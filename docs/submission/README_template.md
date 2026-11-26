# 13조 잡음처리반 — 음성 잡음 제거 챌린지 (2차)

| 항목 | 내용 |
|---|---|
| 팀 / 회차 | 13조 잡음처리반 / 2차 챌린지 |
| 제출 버전 | 태그 `[태그]` / 커밋 `[커밋]` |
| 모델 | `gru_v8p2z3e` (BiGRU 마스크) — 가중치 `weights/gru_v8p2z3e.pt`, 설정 `configs/submission.yaml` |
| Validation | PESQ-WB 3.6534 / SNR 1.21 dB (164개 파일, `pesq==0.0.4`) |

## 설치

Python 3.12

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

`input/`의 WAV 파일을 처리해 `output/`에 같은 파일명으로 저장합니다 (16 kHz 모노, 입력과 같은 샘플 수, CPU만 사용, 절대 경로·다운로드 없음). 폴더 지정: `python main.py --input <입력 폴더> --output <출력 폴더>`.

## 학습 요약

- 데이터: 챌린지 train split만 학습, validation은 가중치 선택에만 사용, 외부 데이터·사전학습 가중치 없음.
- 손실: 0.2 × cmag (압축 크기 손실) + 0.32 × torch-pesq (미분 가능한 PESQ), 학습에만 사용.
- 증강: 동적 혼합 (SNR −5 ~ 20 dB), 속도 변환, 무작위 EQ.
- 설정: Adam lr 5e-5, batch 16, seed 3, 최대 200 epoch, 조기 종료 30 epoch, Google Colab T4 GPU.
- 선택: validation PESQ가 가장 높은 가중치, 직전 후보보다 val_remix와 외부 잡음 평가 세트에서도 높음.
- 학습 코드·실험 기록: https://github.com/pauljh4323/SGSYS_13thTeam_nrc
