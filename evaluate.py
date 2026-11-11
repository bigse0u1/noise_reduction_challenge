"""Validation 데이터에 denoise() 를 적용하고 PESQ-WB / SNR 을 계산한다.

    python evaluate.py [--split validation] [--out validation_results.csv]

samples.csv 와 Clean 파일이 필요하므로 train/, validation/ 폴더가 저장소 루트에 있어야 한다.
Clean 은 점수 계산에만 사용하며, denoise() 에는 Noisy 만 전달한다.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq
from tqdm import tqdm

from denoise import denoise

ROOT = Path(__file__).resolve().parent


def snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    error = np.sum((clean - est) ** 2)
    return float(np.inf if error == 0 else 10 * np.log10(np.sum(clean ** 2) / error))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--out", type=Path, default=ROOT / "validation_results.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(ROOT / args.split / "samples.csv", encoding="utf-8-sig")
    rows = []
    for r in tqdm(manifest.itertuples(), total=len(manifest)):
        clean, sr = sf.read(ROOT / r.clean_path, dtype="float64")
        noisy, _ = sf.read(ROOT / r.noisy_path, dtype="float64")
        est = denoise(noisy, sr)
        rows.append({
            "segment_id": r.segment_id,
            "noise_category": r.noise_category,
            "noise_difficulty": r.noise_difficulty,
            "target_snr_db": r.target_snr_db,
            "snr_noisy": snr_db(clean, noisy),
            "snr_denoised": snr_db(clean, est),
            "pesq_noisy": pesq(sr, clean, noisy, "wb"),
            "pesq_denoised": pesq(sr, clean, est, "wb"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n{args.split} {len(df)}개 파일 평균")
    print(df[["snr_noisy", "snr_denoised", "pesq_noisy", "pesq_denoised"]].mean().round(4))
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
