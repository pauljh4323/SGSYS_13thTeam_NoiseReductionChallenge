# 2026-2 신호및시스템 음성 잡음 제거 챌린지 — 13조(잡음처리반)

단일 채널 음성의 잡음을 전통적 신호 처리만으로 제거하는 알고리즘입니다. 평가 지표는 광대역 PESQ(PESQ-WB, 16 kHz)이며 SNR은 보조 지표입니다.

## 설치 방법

Python 3.12, pip만으로 설치할 수 있습니다 (conda 불필요):

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; Linux/macOS는 `source .venv/bin/activate`
pip install -r requirements.txt
```

`requirements.txt`에는 `main.py` 실행에 필요한 고정 버전 의존성만 있습니다 (numpy, scipy, soundfile, pyyaml).

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


## 참고

- AI 참고 문헌: [docs/references.md](docs/references.md)
