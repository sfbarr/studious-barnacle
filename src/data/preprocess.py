#!/usr/bin/env python3
"""
Preprocessing pipeline: FMA medium MP3s -> log-mel spectrogram tensors.

Outputs (in --out_dir):
  X.npy                float32  (N, 1, 128, 1292)
  y.npy                int64    (N,)   integer labels 0-15
  label_map.json                genre name -> integer index
  preprocessing_log.json        timing, memory, and class distribution stats

Usage (from project root):
  python src/data/preprocess.py
  python src/data/preprocess.py --workers 8
"""

import os
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

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
                        help="Directory where outputs are written")
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help=f"Number of parallel worker processes (default: all {cpu_count()} logical cores)")
    args = parser.parse_args()

    pipeline_start = time.perf_counter()
    started_at = datetime.now().isoformat()
    process = psutil.Process(os.getpid())

    audio_dir = os.path.abspath(args.audio_dir)

    print(f"Started: {started_at}")
    print(f"Workers: {args.workers} / {cpu_count()} logical cores")
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
    skipped_ids: list = []
    track_times: list = []
    peak_mem_mb = 0.0

    print(f"Processing {len(tasks)} tracks with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_track, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tracks"):
            tid = futures[future]
            result = future.result()
            if result is None:
                skipped_ids.append(tid)
            else:
                _, log_mel, elapsed = result
                mel_cache[tid] = log_mel
                track_times.append(elapsed)

            mem_mb = process.memory_info().rss / 1024 / 1024
            peak_mem_mb = max(peak_mem_mb, mem_mb)

    n_processed = len(mel_cache)
    n_skipped = len(skipped_ids)
    print(f"Processed {n_processed} tracks, skipped {n_skipped}")

    valid_ids = [tid for tid in track_ids if tid in mel_cache]
    X = np.stack([mel_cache[tid] for tid in valid_ids])
    y = np.array(
        [genre_to_idx[genre_series[tid]] for tid in valid_ids],
        dtype=np.int64,
    )

    np.save(os.path.join(args.out_dir, "X.npy"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)

    total_time = time.perf_counter() - pipeline_start
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {genres[i]: int(c) for i, c in zip(unique, counts)}

    print(f"\nSaved X.npy  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved y.npy  shape={y.shape}  dtype={y.dtype}")
    print(f"Class distribution: {class_dist}")
    print(f"\nTotal time:       {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"Peak memory:      {peak_mem_mb:.0f} MB")
    if track_times:
        print(f"Per-track time:   avg={np.mean(track_times):.3f}s  "
              f"min={np.min(track_times):.3f}s  max={np.max(track_times):.3f}s")

    log = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "total_time_sec": round(total_time, 2),
        "total_time_min": round(total_time / 60, 2),
        "workers": args.workers,
        "logical_cores": cpu_count(),
        "tracks_submitted": len(tasks),
        "tracks_processed": n_processed,
        "tracks_skipped": n_skipped,
        "skipped_ids": skipped_ids,
        "peak_memory_mb": round(peak_mem_mb, 1),
        "per_track_time_sec": {
            "mean": round(float(np.mean(track_times)), 4),
            "min":  round(float(np.min(track_times)), 4),
            "max":  round(float(np.max(track_times)), 4),
            "std":  round(float(np.std(track_times)), 4),
        } if track_times else {},
        "output_shape": {"X": list(X.shape), "y": list(y.shape)},
        "class_distribution": class_dist,
        "audio_dir": audio_dir,
        "out_dir": os.path.abspath(args.out_dir),
    }
    log_path = os.path.join(args.out_dir, "preprocessing_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Preprocessing log saved to {log_path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
