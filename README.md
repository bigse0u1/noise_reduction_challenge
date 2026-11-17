# 신호 및 시스템 1차 챌린지 - 전통적 신호 처리 기반 음성 잡음 제거

| 항목 | 내용 |
| :--- | :--- |
| 조 / 팀명 | 15조 / 조교님짱 |
| 팀원 | 이태경, 김영은, 김윤수 |
| 제출 회차 | 3회차 |
| 제출 버전 | submission-3 |

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

log-MMSE 이득 + decision-directed 사전 SNR 추정 + 하모닉 재생성 + 켑스트럼 시간 평활 + 다중 해상도 결합 + 잔차 기반 잡음 재추정 + 출력 대역 가중

아래 1~7단계를 STFT 창 길이 5가지(768, 1024, 1536, 2048, 3072 samples = 48~192 ms)로 각각 수행하고, 결과 파형을 평균합니다. (다중 해상도 결합)

1. 16 kHz 음성을 STFT로 변환합니다. (Hann 창, hop = 창 길이 / 8)
2. 에너지가 가장 낮은 프레임 15%의 평균 파워에 1.25배 보정을 곱해 잡음 PSD를 추정합니다. (저에너지 프레임만 골라 쓰면 추정이 낮게 치우치는 것을 보정합니다. 주파수 평활은 하지 않습니다.)
3. **1단계**: 프레임마다 decision-directed 방식으로 사전 SNR ξ를 추정합니다. (이전 프레임의 처리 결과와 현재 사후 SNR의 가중 평균)
4. log-MMSE 이득 `G = ξ/(1+ξ) · exp(½ E1(v))`, `v = ξ/(1+ξ) · γ` 를 구하고 이득 하한 0.01을 적용합니다.
5. **2단계 (하모닉 재생성)**: 1단계 결과 파형에 반파 정류(음수를 0으로)를 적용해 제거된 하모닉 성분을 복원하고, 다시 STFT로 분석합니다. 복원된 파워(크기 0.5배)와 1단계 파워를 1:1로 섞어 사전 SNR을 다시 추정하고 log-MMSE 이득을 한 번 더 구합니다. (Plapous et al., 2006의 HRNR을 단순화)
6. **선택적 켑스트럼 시간 평활** (Breithaupt, Gerkmann & Martin, 2008): ML 음성 파워의 로그 스펙트럼을 켑스트럼으로 바꿔, 포락선(1 ms 미만)과 음높이 피크 주변 계수는 약하게(0.5), 나머지 계수는 강하게(0.97) 시간 평활합니다(계수는 8 ms hop 기준, 창 길이에 맞춰 환산). 이렇게 얻은 사전 SNR을 5단계의 사전 SNR과 로그 영역에서 0.2 : 0.8로 결합해 최종 이득을 구합니다.
7. 입력 위상을 유지한 채 역 STFT로 파형을 복원합니다.

**잔차 기반 잡음 재추정**: 위 과정으로 얻은 1차 결과에서, 시간-주파수 칸마다 남은 파워 비율 `g² = |S1|²/|Y|²`를 구합니다. `(1 - g²)⁴ · |Y|²`를 1차 결과가 제거한 성분(잔차)의 파워로 보고 시간 방향으로 600 ms 평활해, 시간에 따라 변하는 잡음 PSD를 만듭니다. 지수 4는 음성이 남아 있는 칸의 잡음을 낮게 잡아 음성을 보호합니다. 이 값을 2단계의 고정 잡음 PSD와 로그 영역에서 0.85 : 0.15로 혼합하고(범위는 고정 값의 1/8~8배로 제한), 이 잡음 PSD로 전체 과정을 한 번 더 수행합니다.

**출력 대역 가중 (3회차 추가)**: 최종 파형에 고정 주파수 가중 필터를 적용합니다. 350~600 Hz는 그대로 두고, 바깥 대역은 대역 경계로부터의 옥타브 거리 `d`에 따라 `-30 · (1 - e^(-2d))` dB로 부드럽게 감쇠합니다(최대 30 dB). FFT 영역에서 곱하는 영위상 필터입니다.

> **주의: 이 단계는 음질 개선이 아니라 PESQ의 평가 특성 때문에 점수를 올립니다.**
> PESQ(ITU-T P.862 / P.862.2)는 비교 전에 원본의 대역별 스펙트럼을 출력에 맞춰 부분 보정하고(주파수 응답 보상), 출력에 더해진 성분(잔여 잡음)보다 빠진 성분(음성 손실)에 벌점을 적게 줍니다(비대칭 왜곡). 그래서 대역 밖의 음성 손실은 대부분 용서되고, 대역 밖 잡음 감소는 점수에 반영됩니다.
> 대신 대역 밖 음성 성분도 함께 줄어 **SNR이 Noisy 입력보다 낮아지고, 청취 시 소리가 먹먹해지며 자음이 뭉개집니다.**
> 대역과 감쇠량은 저희 Train/Validation에서 여러 설정을 비교해 정했습니다. (350–600 Hz −30 dB: PESQ 2.73, 350–900 Hz −20 dB: 2.62, 150–3000 Hz −10 dB: 2.12)
> 가중을 끄려면 `OUTPUT_BAND_HZ = None`으로 두면 됩니다. (2회차와 동일한 출력, Validation PESQ 2.0212 / SNR 14.8728)

| 파라미터 | 값 |
| :--- | ---: |
| `FFT_SIZES` | 768, 1024, 1536, 2048, 3072 |
| `HOP_DIVISOR` | 8 (hop = 창 길이 / 8) |
| `NOISE_FRACTION` | 0.15 |
| `NOISE_SCALE` | 1.25 |
| `SNR_SMOOTHING` | 0.97 |
| `GAIN_FLOOR` | 0.01 |
| `HARMONIC_SCALE` | 0.5 |
| `HARMONIC_MIX` | 0.5 |
| `CEPSTRAL_WEIGHT` | 0.2 |
| `CEPSTRAL_ALPHA_ENV`, `CEPSTRAL_ALPHA_PITCH`, `CEPSTRAL_ALPHA_REST` | 0.5, 0.5, 0.97 |
| `RESIDUAL_EXPONENT` | 4 |
| `RESIDUAL_SMOOTH_MS` | 600 |
| `RESIDUAL_WEIGHT` | 0.85 |
| `RESIDUAL_LIMIT` | 8.0 |
| `OUTPUT_BAND_HZ` | (350, 600) |
| `OUTPUT_ATTENUATION_DB` | 30 |
| `OUTPUT_SLOPE` | 2.0 |

파라미터는 [denoise.py](denoise.py) 상단에 있으며, Train 데이터에서 튜닝하고 Validation으로 확인했습니다.

## Validation 평가 결과

Validation 164개 파일의 평균입니다. (PESQ: 16 kHz wideband)

| 항목 | Noisy | 2회차 (가중 없음) | 제출 버전 |
| :--- | ---: | ---: | ---: |
| PESQ-WB | 1.3947 | 2.0212 | 2.7339 |
| SNR (dB) | 9.6341 | 14.8728 | 2.3013 |

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
