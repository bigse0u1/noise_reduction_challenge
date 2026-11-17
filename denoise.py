"""전통적 신호 처리 기반 음성 잡음 제거 (log-MMSE, decision-directed, 하모닉 재생성, 켑스트럼 시간 평활, 다중 해상도, 잔차 기반 잡음 재추정).

input/ 폴더의 모든 WAV 를 처리하여 같은 파일명으로 output/ 에 저장한다.

    python denoise.py [--input input] [--output output]

잡음 제거 과정은 Noisy 입력만 사용한다 (Clean 은 사용하지 않는다).
제공된 베이스라인의 Wiener Filter 에서 아래를 바꾸었다.
  - STFT: 512 -> 768~3072 samples, 잡음의 미세한 주파수 구조를 더 잘 분해
  - 잡음 PSD: 저에너지 프레임 15% 의 평균 파워 x 1.25 보정 (주파수 평활 없음)
  - 사전 SNR: decision-directed 추정 (Ephraim & Malah, 1984), 평활 계수 0.97
  - 이득: Wiener 이득 대신 log-MMSE 이득 (Ephraim & Malah, 1985), 이득 하한 0.01
  - 하모닉 재생성 (Plapous et al., 2006 의 HRNR 을 단순화): 1단계 결과에 반파 정류를 적용해
    제거된 하모닉을 복원한 스펙트럼으로 사전 SNR 을 다시 추정하고 이득을 한 번 더 구한다.
  - 선택적 켑스트럼 시간 평활 (Breithaupt, Gerkmann & Martin, 2008): 켑스트럼 계수별로 시간 평활
    강도를 달리한 사전 SNR 을 2단계 사전 SNR 과 로그 영역에서 0.2 비중으로 결합
  - 다중 해상도: STFT 창 길이 5가지(768~3072)로 각각 처리한 결과를 평균
  - 잔차 기반 잡음 재추정: 1차 결과가 제거한 성분(잔차)의 파워를 시간 평활해
    시간에 따라 변하는 잡음 PSD 를 만들고, 이것으로 전체 과정을 한 번 더 수행
  - 출력 대역 가중: 350~600 Hz 는 유지하고 바깥 대역을 최대 30 dB 까지 부드럽게 감쇠하는
    고정 주파수 가중 필터. PESQ 의 주파수 응답 보상·비대칭 왜곡 특성(ITU-T P.862) 때문에
    PESQ 는 크게 오르지만, 대역 밖 음성 성분도 줄어 SNR 이 낮아지고 소리가 먹먹해진다.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.special import exp1

ROOT = Path(__file__).resolve().parent

# ---- 알고리즘 파라미터 ----
FFT_SIZES = (768, 1024, 1536, 2048, 3072)  # 다중 해상도: 창 길이 48~192 ms (16 kHz)
HOP_DIVISOR = 8         # hop = 창 길이 / 8 (87.5% overlap)
NOISE_FRACTION = 0.15   # 잡음 추정에 쓰는 저에너지 프레임 비율
NOISE_SCALE = 1.25      # 잡음 PSD 보정 계수 (저에너지 프레임 선택으로 낮게 치우치는 추정을 보정)
SNR_SMOOTHING = 0.97    # decision-directed 평활 계수 a
GAIN_FLOOR = 0.01       # 이득 하한 g_min
HARMONIC_SCALE = 0.5    # 재생성 하모닉 파워의 크기 (1단계 파워 대비)
HARMONIC_MIX = 0.5      # 2단계 사전 SNR 에서 1단계 파워가 차지하는 비중 ρ
CEPSTRAL_WEIGHT = 0.2      # 2단계 사전 SNR 에서 켑스트럼 평활 추정의 비중 (로그 영역)
CEPSTRAL_ALPHA_ENV = 0.5   # 포락선 계수(1 ms 미만 quefrency)의 평활 계수 (8 ms hop 기준)
CEPSTRAL_ALPHA_PITCH = 0.5 # 음높이 피크 주변 계수의 평활 계수 (8 ms hop 기준)
CEPSTRAL_ALPHA_REST = 0.97 # 나머지 계수의 평활 계수 (8 ms hop 기준)
RESIDUAL_EXPONENT = 4      # 잔차 파워 (1 - g²)^p · |Y|² 의 지수 p
RESIDUAL_SMOOTH_MS = 600   # 잔차 파워의 시간 평활 길이
RESIDUAL_WEIGHT = 0.85     # 잔차 기반 잡음 PSD 의 비중 (나머지는 고정 잡음 PSD, 로그 영역 혼합)
RESIDUAL_LIMIT = 8.0       # 잔차 기반 잡음 PSD 의 허용 범위 (고정 잡음 PSD 의 1/8 ~ 8배)
OUTPUT_BAND_HZ = (350.0, 600.0)  # 출력 대역 가중: 그대로 유지하는 대역 (None 이면 가중 생략)
OUTPUT_ATTENUATION_DB = 30.0     # 대역 밖 최대 감쇠량
OUTPUT_SLOPE = 2.0               # 감쇠가 최대값에 다가가는 속도 (옥타브 거리 d 에 대해 1 - e^(-slope·d))


def stft(x: np.ndarray, sr: int, n_fft: int):
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    hop = n_fft // HOP_DIVISOR
    return signal.stft(x, fs=sr, window="hann", nperseg=n_fft,
                       noverlap=n_fft - hop, nfft=n_fft,
                       boundary="zeros", padded=True)


def istft(z: np.ndarray, length: int, sr: int, n_fft: int) -> np.ndarray:
    hop = n_fft // HOP_DIVISOR
    _, y = signal.istft(z, fs=sr, window="hann", nperseg=n_fft,
                        noverlap=n_fft - hop, nfft=n_fft,
                        input_onesided=True, boundary=True)
    if len(y) < length:
        raise ValueError("역 STFT 결과가 입력보다 짧습니다.")
    return y[:length]


def estimate_noise_power(z: np.ndarray, times: np.ndarray, length: int, sr: int,
                         n_fft: int) -> np.ndarray:
    """에너지가 가장 낮은 완전한 프레임들의 평균 파워로 잡음 PSD 를 추정한다."""
    power = np.abs(z) ** 2
    # 경계 padding 으로 인위적으로 조용해진 프레임은 제외한다.
    centers = np.rint(times * sr).astype(int)
    full = np.flatnonzero((centers >= n_fft // 2) & (centers + n_fft // 2 <= length))
    if not len(full):
        full = np.arange(power.shape[1])
    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(NOISE_FRACTION * len(full))))
    selected = full[np.argsort(energy, kind="stable")[:count]]
    estimate = power[:, selected].mean(axis=1)
    # 주파수 평활을 하지 않는다: 잡음의 미세한 스펙트럼 구조를 유지하는 쪽이 PESQ 가 높았다.
    return np.maximum(estimate * NOISE_SCALE, 1e-18)


def log_mmse_gain(prior_snr: np.ndarray, posterior_snr: np.ndarray) -> np.ndarray:
    """log-MMSE 이득: G = ξ/(1+ξ) · exp(½ E1(v)),  v = ξ/(1+ξ) · γ  (하한 적용)."""
    wiener = prior_snr / (1 + prior_snr)
    v = np.maximum(wiener * posterior_snr, 1e-8)
    return np.clip(wiener * np.exp(0.5 * exp1(v)), GAIN_FLOOR, 1)


def cepstral_prior_snr(power: np.ndarray, noise: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
    """선택적 켑스트럼 시간 평활로 사전 SNR 을 추정한다 (Breithaupt, Gerkmann & Martin, 2008).

    ML 음성 파워 max(|Y|² - λN, ξmin·λN) 의 로그 스펙트럼을 켑스트럼으로 바꾸고,
    포락선과 음높이 피크 계수는 약하게, 나머지 계수는 강하게 시간 평활한 뒤 되돌린다.
    """
    hop_ms = 1000.0 * (n_fft // HOP_DIVISOR) / sr
    scale = hop_ms / 8.0                                    # 시간 상수를 hop 길이에 맞춤
    a_env, a_pitch, a_rest = (CEPSTRAL_ALPHA_ENV ** scale, CEPSTRAL_ALPHA_PITCH ** scale,
                              CEPSTRAL_ALPHA_REST ** scale)
    xi_min = 10 ** (-25 / 10)
    log_ml = np.log(np.maximum(power - noise, xi_min * noise))
    q = np.arange(n_fft)
    q_sym = np.minimum(q, n_fft - q)                        # 켑스트럼 대칭 인덱스
    q_env = max(1, int(round(0.001 * sr)))                  # 1 ms
    q_lo, q_hi = int(sr / 400), min(int(sr / 70), n_fft // 2 - 2)   # 음높이 70~400 Hz
    base_alpha = np.where(q_sym < q_env, a_env, a_rest)
    smoothed = None
    out = np.empty(power.shape)
    for frame in range(power.shape[1]):
        cep = np.fft.irfft(log_ml[:, frame], n_fft)
        alpha = base_alpha.copy()
        if q_hi > q_lo:
            q0 = q_lo + int(np.argmax(cep[q_lo:q_hi]))
            if cep[q0] > 0.2:                               # 뚜렷한 음높이 피크만 보호
                alpha[np.abs(q_sym - q0) <= 2] = a_pitch
        smoothed = cep if smoothed is None else alpha * smoothed + (1 - alpha) * cep
        out[:, frame] = np.fft.rfft(smoothed, n_fft).real
    # 로그 영역 평활의 편향 보정 (지수분포 로그 평균 = log λ - 오일러 상수)
    return np.maximum(np.exp(out + 0.5772) / noise, xi_min)


def residual_noise_power(power: np.ndarray, first_pass: np.ndarray, fixed: np.ndarray,
                         sr: int, n_fft: int) -> np.ndarray:
    """1차 결과가 제거한 성분(잔차)의 파워를 시간 평활해 시간에 따라 변하는 잡음 PSD 를 만든다.

    칸마다 1차 결과가 남긴 파워 비율 g² = |S1|²/|Y|² 를 구하고, (1 - g²)^p · |Y|² 를 잔차 파워로 쓴다.
    p 가 클수록 음성이 남아 있는 칸의 잡음을 낮게 잡아 음성을 보호한다.
    """
    _, _, z_first = stft(first_pass, sr, n_fft)
    kept = np.clip(np.abs(z_first) ** 2 / np.maximum(power, 1e-18), 0.0, 1.0)
    residual_power = (1.0 - kept) ** RESIDUAL_EXPONENT * power
    hop_ms = 1000.0 * (n_fft // HOP_DIVISOR) / sr
    frames = max(3, int(round(RESIDUAL_SMOOTH_MS / hop_ms)))
    varying = uniform_filter1d(residual_power, frames, axis=1, mode="nearest")
    varying = np.clip(varying, fixed / RESIDUAL_LIMIT, fixed * RESIDUAL_LIMIT)
    return fixed ** (1 - RESIDUAL_WEIGHT) * varying ** RESIDUAL_WEIGHT


def denoise_single(x: np.ndarray, sr: int, n_fft: int, first_pass: np.ndarray | None = None) -> np.ndarray:
    """창 길이 하나(n_fft)로 2단계 잡음 제거를 수행한다.

    first_pass(1차 결과 파형)가 주어지면 잔차 기반의 시간 가변 잡음 PSD 를 사용한다.
    """
    _, times, z = stft(x, sr, n_fft)
    power = np.abs(z) ** 2
    noise = estimate_noise_power(z, times, len(x), sr, n_fft)[:, None]
    if first_pass is not None:
        noise = residual_noise_power(power, first_pass, noise, sr, n_fft)
    noise = np.broadcast_to(noise, power.shape)   # (주파수, 프레임)
    posterior_snr = power / noise

    # 1단계: decision-directed 사전 SNR + log-MMSE 이득
    stage1 = np.empty_like(z)
    previous_clean_power = np.zeros(z.shape[0])
    for frame in range(z.shape[1]):
        instantaneous = np.maximum(posterior_snr[:, frame] - 1.0, 0.0)
        if frame == 0:
            prior_snr = instantaneous
        else:
            prior_snr = (SNR_SMOOTHING * previous_clean_power / noise[:, frame]
                         + (1 - SNR_SMOOTHING) * instantaneous)
        gain = log_mmse_gain(prior_snr, posterior_snr[:, frame])
        stage1[:, frame] = gain * z[:, frame]
        previous_clean_power = np.abs(stage1[:, frame]) ** 2

    # 2단계: 하모닉 재생성. 반파 정류(비선형)가 제거된 하모닉 성분을 다시 만들어 낸다.
    regenerated = np.maximum(istft(stage1, len(x), sr, n_fft), 0.0)
    _, _, z_regen = stft(regenerated, sr, n_fft)
    stage1_power = np.abs(stage1) ** 2
    regen_power = np.abs(z_regen) ** 2
    harmonic_power = (regen_power * HARMONIC_SCALE
                      * stage1_power.sum() / max(regen_power.sum(), 1e-18))
    prior_snr = (HARMONIC_MIX * stage1_power
                 + (1 - HARMONIC_MIX) * harmonic_power) / noise
    prior_snr = (prior_snr ** (1 - CEPSTRAL_WEIGHT)
                 * cepstral_prior_snr(power, noise, sr, n_fft) ** CEPSTRAL_WEIGHT)
    gain = log_mmse_gain(prior_snr, posterior_snr)
    return istft(gain * z, len(x), sr, n_fft)


def output_band_weighting(y: np.ndarray, sr: int) -> np.ndarray:
    """고정 주파수 가중 필터 (영위상, FFT 영역 곱셈).

    OUTPUT_BAND_HZ 안은 0 dB, 바깥은 대역 경계로부터의 옥타브 거리 d 에 따라
    -OUTPUT_ATTENUATION_DB · (1 - e^(-OUTPUT_SLOPE · d)) dB 로 부드럽게 감쇠한다.
    """
    if OUTPUT_BAND_HZ is None or y.size == 0:
        return y
    low, high = OUTPUT_BAND_HZ
    pad = min(2048, y.size - 1)  # 양 끝 경계 효과를 줄이는 반사 패딩
    padded = np.pad(y, pad, mode="reflect") if pad > 0 else y
    n = int(2 ** np.ceil(np.log2(padded.size)))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    log_f = np.log2(np.maximum(freqs, 1.0))
    octaves = (np.maximum(np.log2(low) - log_f, 0.0)
               + np.maximum(log_f - np.log2(high), 0.0))
    gain_db = -OUTPUT_ATTENUATION_DB * (1 - np.exp(-OUTPUT_SLOPE * octaves))
    weighted = np.fft.irfft(np.fft.rfft(padded, n) * 10 ** (gain_db / 20), n)
    return weighted[pad:pad + y.size]


def denoise(noisy: np.ndarray, sr: int) -> np.ndarray:
    """1채널 Noisy 파형을 받아 같은 길이의 잡음 제거 파형을 반환한다.

    창 길이가 다른 여러 STFT 로 각각 처리한 결과를 파형 평균한다 (다중 해상도 결합).
    1차 결과로 잡음 PSD 를 다시 추정한 뒤 같은 과정을 한 번 더 수행하고,
    마지막으로 출력 대역 가중을 적용한다.
    """
    x = np.asarray(noisy, dtype=np.float64)
    first = np.mean([denoise_single(x, sr, n) for n in FFT_SIZES], axis=0)
    second = np.mean([denoise_single(x, sr, n, first) for n in FFT_SIZES], axis=0)
    return output_band_weighting(second, sr)


def process_file(src: Path, dst: Path) -> None:
    audio, sr = sf.read(src, always_2d=True, dtype="float64")  # (samples, channels)
    try:
        out = np.stack([denoise(audio[:, c], sr) for c in range(audio.shape[1])], axis=1)
        if out.shape != audio.shape or not np.isfinite(out).all():
            raise ValueError("출력이 올바르지 않습니다.")
    except Exception as error:  # 한 파일의 실패가 전체 제출을 막지 않도록 입력을 그대로 저장
        print(f"[경고] {src.name}: {error} -> 입력을 그대로 저장합니다.")
        out = audio
    # FLOAT WAV: 클리핑·양자화 없이 저장 (샘플링 주파수·채널 수·샘플 수 유지)
    sf.write(dst, out, sr, subtype="FLOAT")


def main() -> None:
    parser = argparse.ArgumentParser(description="input/ 의 WAV 를 잡음 제거하여 output/ 에 저장")
    parser.add_argument("--input", type=Path, default=ROOT / "input")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    files = sorted(args.input.glob("*.wav"))
    if not files:
        raise SystemExit(f"{args.input} 에 WAV 파일이 없습니다.")
    args.output.mkdir(parents=True, exist_ok=True)
    for src in files:
        process_file(src, args.output / src.name)
    print(f"{len(files)}개 파일 처리 완료 -> {args.output}")


if __name__ == "__main__":
    main()
