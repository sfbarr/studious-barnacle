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
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

# worker lives in a separate module so it can be imported (not re-run) by
# spawned processes on Windows — avoids the __main__ pickling problem
from mel_worker import process_track


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

    # resolve to absolute so relative paths survive the process boundary
    audio_dir = os.path.abspath(args.audio_dir)

    print("Loading metadata...")
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
    tasks = [(tid, audio_dir) for tid in track_ids]

    mel_cache: dict = {}
    skipped = 0

    print(f"Processing {len(tasks)} tracks with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_track, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tracks"):
            tid = futures[future]
            result = future.result()
            if result is None:
                skipped += 1
            else:
                _, log_mel = result
                mel_cache[tid] = log_mel

    print(f"Processed {len(mel_cache)} tracks, skipped {skipped}")

    valid_ids = [tid for tid in track_ids if tid in mel_cache]
    X = np.stack([mel_cache[tid] for tid in valid_ids])
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
    # ensure src/data is on the path so workers can import mel_worker
    sys.path.insert(0, os.path.dirname(__file__))
    main()
