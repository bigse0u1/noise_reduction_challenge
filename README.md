# 신호 및 시스템 1차 챌린지 - 전통적 신호 처리 기반 음성 잡음 제거

| 항목 | 내용 |
| :--- | :--- |
| 조 / 팀명 | 15조 / 조교님짱 |
| 팀원 | 이태경, 김영은, 김윤수 |
| 제출 회차 | 1회차 (최초 제출) |
| 제출 버전 | submission-1 |

딥러닝과 머신러닝을 사용하지 않고, 푸리에 변환 기반의 신호 처리만으로 잡음이 섞인 음성에서 잡음을 줄입니다.

## 설치

Python 3.12 이상을 권장합니다. (개발 환경: Python 3.14)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

`input/` 폴더에 테스트 WAV 파일을 넣고 실행합니다. 입력 파일마다 같은 파일명의 결과가 `output/` 에 저장됩니다.

```bash
python denoise.py
```

폴더를 직접 지정할 수도 있습니다.

```bash
python denoise.py --input <입력 폴더> --output <출력 폴더>
```

- `output/` 폴더가 없으면 자동으로 만듭니다.
- 출력은 입력과 같은 샘플링 주파수, 채널 수, 샘플 수를 유지합니다. (FLOAT WAV)
- 잡음 제거에는 Noisy 입력만 사용합니다.

## 알고리즘

log-MMSE 이득 + decision-directed 사전 SNR 추정

1. 16 kHz 음성을 STFT로 변환합니다. (Hann 창 512 samples, hop 128 samples)
2. 에너지가 가장 낮은 프레임 10%의 평균 파워로 잡음 PSD를 추정하고, 3-bin 평활화합니다.
3. 프레임마다 decision-directed 방식으로 사전 SNR ξ를 추정합니다. (이전 프레임의 처리 결과와 현재 사후 SNR의 가중 평균)
4. log-MMSE 이득 `G = ξ/(1+ξ) · exp(½ E1(v))`, `v = ξ/(1+ξ) · γ` 를 구하고 이득 하한 0.15를 적용합니다.
5. 입력 위상을 유지한 채 역 STFT로 파형을 복원합니다.

| 파라미터 | 값 |
| :--- | ---: |
| `N_FFT` | 512 |
| `HOP` | 128 |
| `NOISE_FRACTION` | 0.10 |
| `SNR_SMOOTHING` | 0.97 |
| `GAIN_FLOOR` | 0.15 |

파라미터는 [denoise.py](denoise.py) 상단에 있으며, Train 데이터에서 튜닝하고 Validation으로 확인했습니다.

## Validation 평가 결과

Validation 164개 파일의 평균입니다. (PESQ: 16 kHz wideband)

| 항목 | Noisy | 제출 버전 |
| :--- | ---: | ---: |
| PESQ-WB | 1.3947 | 1.7431 |
| SNR (dB) | 9.6341 | 13.4270 |

파일별 결과는 [validation_results.csv](validation_results.csv) 에 있습니다.

### 결과 재현

공개 데이터의 `train/`, `validation/` 폴더(각각 `samples.csv`, `clean/`, `noisy/` 포함)를 저장소 루트에 두고 실행합니다. 이 폴더들은 저장소에 포함되어 있지 않습니다.

```bash
python evaluate.py
```

## 폴더 구성

| 경로 | 내용 |
| :--- | :--- |
| `denoise.py` | 제출 코드 (`input/` → `output/`) |
| `evaluate.py` | Validation PESQ/SNR 평가 |
| `requirements.txt` | 필요한 패키지와 버전 |
| `validation_results.csv` | 제출 버전의 Validation 결과 |
| `dataset_exploration.ipynb`, `noise_reduction_baselines.ipynb`, `figures/` | 제공된 참고 자료 |
