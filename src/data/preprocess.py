#!/usr/bin/env python3
"""
Preprocessing pipeline: FMA medium MP3s -> log-mel spectrogram tensors.

Outputs (in --out_dir):
  X.npy          float32  (N, 1, 128, 1292)
  y.npy          int64    (N,)   integer labels 0-15
  label_map.json          genre name -> integer index

Usage (from project root):
  python src/data/preprocess.py
  python src/data/preprocess.py --workers 8
"""

import os
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

SR = 22050
DURATION = 30.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
# exact frame count for a 30s clip at the settings above
N_FRAMES = int(np.ceil(DURATION * SR / HOP_LENGTH))  # 1292


def _track_path(track_id: int, audio_dir: str) -> str:
    tid_str = f"{track_id:06d}"
    return os.path.join(audio_dir, tid_str[:3], f"{tid_str}.mp3")


def _process_track(args: tuple):
    """Worker function: load one MP3 and return its log-mel array or None."""
    tid, audio_dir = args
    path = _track_path(tid, audio_dir)
    if not os.path.exists(path):
        return None
    try:
        y, _ = librosa.load(path, sr=SR, duration=DURATION, mono=True)
        target_samples = int(SR * DURATION)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
        # enforce fixed time axis
        if log_mel.shape[1] > N_FRAMES:
            log_mel = log_mel[:, :N_FRAMES]
        elif log_mel.shape[1] < N_FRAMES:
            pad = N_FRAMES - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)))
        # per-sample z-score normalisation
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        return tid, log_mel[np.newaxis].astype(np.float32)  # (1, n_mels, T)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert FMA medium MP3s to log-mel spectrogram .npy tensors"
    )
    parser.add_argument("--audio_dir", default="data/fma_medium",
                        help="Root folder containing the FMA medium MP3 subdirectories")
    parser.add_argument("--tracks_csv", default="data/fma_metadata/tracks.csv",
                        help="Path to tracks.csv from the FMA metadata zip")
    parser.add_argument("--out_dir", default="data",
                        help="Directory where X.npy, y.npy and label_map.json are written")
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help=f"Number of parallel worker processes (default: all {cpu_count()} logical cores)")
    args = parser.parse_args()

    print("Loading metadata...")
    # tracks.csv has a two-level header row
    tracks = pd.read_csv(args.tracks_csv, index_col=0, header=[0, 1])
    medium = tracks[tracks[("set", "subset")] == "medium"]
    genre_series = medium[("track", "genre_top")].dropna()

    genres = sorted(genre_series.unique())
    print(f"Genres ({len(genres)}): {genres}")
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    os.makedirs(args.out_dir, exist_ok=True)
    label_map_path = os.path.join(args.out_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(genre_to_idx, f, indent=2)
    print(f"Label map saved to {label_map_path}")

    track_ids = genre_series.index.tolist()
    tasks = [(tid, args.audio_dir) for tid in track_ids]

    mel_cache: dict = {}
    skipped = 0

    print(f"Processing {len(tasks)} tracks with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_track, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tracks"):
            tid = futures[future]
            result = future.result()
            if result is None:
                skipped += 1
            else:
                _, log_mel = result
                mel_cache[tid] = log_mel

    print(f"Processed {len(mel_cache)} tracks, skipped {skipped}")

    # build arrays preserving the original genre_series order
    valid_ids = [tid for tid in track_ids if tid in mel_cache]
    X = np.stack([mel_cache[tid] for tid in valid_ids])           # (N, 1, n_mels, T)
    y = np.array(
        [genre_to_idx[genre_series[tid]] for tid in valid_ids],
        dtype=np.int64,
    )

    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)

    print(f"\nSaved X.npy  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved y.npy  shape={y.shape}  dtype={y.dtype}")
    unique, counts = np.unique(y, return_counts=True)
    dist = {genres[i]: int(c) for i, c in zip(unique, counts)}
    print(f"Class distribution: {dist}")


if __name__ == "__main__":
    main()
