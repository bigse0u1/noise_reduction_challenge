"""전통적 신호 처리 기반 음성 잡음 제거 (log-MMSE, decision-directed, 하모닉 재생성).

input/ 폴더의 모든 WAV 를 처리하여 같은 파일명으로 output/ 에 저장한다.

    python denoise.py [--input input] [--output output]

잡음 제거 과정은 Noisy 입력만 사용한다 (Clean 은 사용하지 않는다).
제공된 베이스라인의 Wiener Filter 에서 아래를 바꾸었다.
  - STFT: 512 -> 1024 samples (64 ms), 잡음의 미세한 주파수 구조를 더 잘 분해
  - 잡음 PSD: 저에너지 프레임 15% 의 평균 파워 (주파수 평활 없음)
  - 사전 SNR: decision-directed 추정 (Ephraim & Malah, 1984), 평활 계수 0.97
  - 이득: Wiener 이득 대신 log-MMSE 이득 (Ephraim & Malah, 1985), 이득 하한 0.01
  - 하모닉 재생성 (Plapous et al., 2006 의 HRNR 을 단순화): 1단계 결과에 반파 정류를 적용해
    제거된 하모닉을 복원한 스펙트럼으로 사전 SNR 을 다시 추정하고 이득을 한 번 더 구한다.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.special import exp1

ROOT = Path(__file__).resolve().parent

# ---- 알고리즘 파라미터 ----
N_FFT = 1024            # 16 kHz 에서 64 ms
HOP = 128               # 8 ms (87.5% overlap)
NOISE_FRACTION = 0.15   # 잡음 추정에 쓰는 저에너지 프레임 비율
SNR_SMOOTHING = 0.97    # decision-directed 평활 계수 a
GAIN_FLOOR = 0.01       # 이득 하한 g_min
HARMONIC_SCALE = 0.5    # 재생성 하모닉 파워의 크기 (1단계 파워 대비)
HARMONIC_MIX = 0.5      # 2단계 사전 SNR 에서 1단계 파워가 차지하는 비중 ρ


def stft(x: np.ndarray, sr: int):
    if x.size < N_FFT:
        x = np.pad(x, (0, N_FFT - x.size))
    return signal.stft(x, fs=sr, window="hann", nperseg=N_FFT,
                       noverlap=N_FFT - HOP, nfft=N_FFT,
                       boundary="zeros", padded=True)


def istft(z: np.ndarray, length: int, sr: int) -> np.ndarray:
    _, y = signal.istft(z, fs=sr, window="hann", nperseg=N_FFT,
                        noverlap=N_FFT - HOP, nfft=N_FFT,
                        input_onesided=True, boundary=True)
    if len(y) < length:
        raise ValueError("역 STFT 결과가 입력보다 짧습니다.")
    return y[:length]


def estimate_noise_power(z: np.ndarray, times: np.ndarray, length: int, sr: int) -> np.ndarray:
    """에너지가 가장 낮은 완전한 프레임들의 평균 파워로 잡음 PSD 를 추정한다."""
    power = np.abs(z) ** 2
    # 경계 padding 으로 인위적으로 조용해진 프레임은 제외한다.
    centers = np.rint(times * sr).astype(int)
    full = np.flatnonzero((centers >= N_FFT // 2) & (centers + N_FFT // 2 <= length))
    if not len(full):
        full = np.arange(power.shape[1])
    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(NOISE_FRACTION * len(full))))
    selected = full[np.argsort(energy, kind="stable")[:count]]
    estimate = power[:, selected].mean(axis=1)
    # 주파수 평활을 하지 않는다: 잡음의 미세한 스펙트럼 구조를 유지하는 쪽이 PESQ 가 높았다.
    return np.maximum(estimate, 1e-18)


def log_mmse_gain(prior_snr: np.ndarray, posterior_snr: np.ndarray) -> np.ndarray:
    """log-MMSE 이득: G = ξ/(1+ξ) · exp(½ E1(v)),  v = ξ/(1+ξ) · γ  (하한 적용)."""
    wiener = prior_snr / (1 + prior_snr)
    v = np.maximum(wiener * posterior_snr, 1e-8)
    return np.clip(wiener * np.exp(0.5 * exp1(v)), GAIN_FLOOR, 1)


def denoise(noisy: np.ndarray, sr: int) -> np.ndarray:
    """1채널 Noisy 파형을 받아 같은 길이의 잡음 제거 파형을 반환한다."""
    x = np.asarray(noisy, dtype=np.float64)
    _, times, z = stft(x, sr)
    power = np.abs(z) ** 2
    noise = estimate_noise_power(z, times, len(x), sr)
    posterior_snr = power / noise[:, None]

    # 1단계: decision-directed 사전 SNR + log-MMSE 이득
    stage1 = np.empty_like(z)
    previous_clean_power = np.zeros_like(noise)
    for frame in range(z.shape[1]):
        instantaneous = np.maximum(posterior_snr[:, frame] - 1.0, 0.0)
        if frame == 0:
            prior_snr = instantaneous
        else:
            prior_snr = (SNR_SMOOTHING * previous_clean_power / noise
                         + (1 - SNR_SMOOTHING) * instantaneous)
        gain = log_mmse_gain(prior_snr, posterior_snr[:, frame])
        stage1[:, frame] = gain * z[:, frame]
        previous_clean_power = np.abs(stage1[:, frame]) ** 2

    # 2단계: 하모닉 재생성. 반파 정류(비선형)가 제거된 하모닉 성분을 다시 만들어 낸다.
    regenerated = np.maximum(istft(stage1, len(x), sr), 0.0)
    _, _, z_regen = stft(regenerated, sr)
    stage1_power = np.abs(stage1) ** 2
    regen_power = np.abs(z_regen) ** 2
    harmonic_power = (regen_power * HARMONIC_SCALE
                      * stage1_power.sum() / max(regen_power.sum(), 1e-18))
    prior_snr = (HARMONIC_MIX * stage1_power
                 + (1 - HARMONIC_MIX) * harmonic_power) / noise[:, None]
    gain = log_mmse_gain(prior_snr, posterior_snr)
    return istft(gain * z, len(x), sr)


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
