"""Run the trained CRNN on a held-out FMA Large test set.

Reads MP3s from data/fma_large_test/, joins their genre labels from
data/fma_large_test_sample.csv, computes log-mel spectrograms with the same
parameters used in training, loads a saved best_model.pt, and writes metrics
in the same format as the other runs in results/.

Usage (from project root):
    python src/eval_test_set.py
    python src/eval_test_set.py --checkpoint results/optimized_weighted_20260507_172654/best_model.pt
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from data.mel_worker import process_track
from models.crnn import CRNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir",  default="data/fma_large_test")
    p.add_argument("--sample_csv", default="data/fma_large_test_sample.csv")
    p.add_argument("--label_map",  default="data/label_map.json")
    p.add_argument("--checkpoint", default="results/optimized_no_weights_20260506_214806/best_model.pt")
    p.add_argument("--rnn_hidden", type=int, default=512)
    p.add_argument("--n_classes",  type=int, default=16)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers",    type=int, default=8)
    p.add_argument("--out_dir",    default=None,
                   help="Defaults to results/test_fma_large_<timestamp>/")
    return p.parse_args()


def preprocess_tracks(track_ids, audio_dir, workers):
    """Compute log-mel for each track ID in parallel. Returns dict {tid: array}."""
    args_list = [(int(tid), audio_dir) for tid in track_ids]
    out = {}
    failed = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_track, a): a[0] for a in args_list}
        for i, fut in enumerate(as_completed(futures), 1):
            tid = futures[fut]
            result = fut.result()
            if result is None:
                failed.append(tid)
                continue
            _, log_mel, _ = result
            out[tid] = log_mel
            if i % 50 == 0 or i == len(futures):
                print(f"  preprocessed {i}/{len(futures)} ({len(failed)} failed)")
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s")
    return out, failed


def main():
    args = parse_args()

    audio_dir = Path(args.audio_dir)
    sample_csv = Path(args.sample_csv)
    label_map_path = Path(args.label_map)
    ckpt_path = Path(args.checkpoint)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"results/test_fma_large_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────────
    with open(label_map_path) as f:
        label_map = json.load(f)        # genre name -> int
    inv_label_map = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)

    sample = pd.read_csv(sample_csv)
    print(f"Sample manifest: {len(sample)} tracks  (csv={sample_csv})")

    # ── Preprocess MP3s in parallel ──────────────────────────────────────────
    print(f"\nPreprocessing audio in {audio_dir}/  (workers={args.workers})...")
    feats, failed = preprocess_tracks(sample["track_id"].tolist(), str(audio_dir), args.workers)

    # ── Assemble tensors, drop failures, drop labels not in label_map ────────
    X_list, y_list, kept_ids, dropped = [], [], [], []
    for _, row in sample.iterrows():
        tid = int(row["track_id"])
        genre = row["genre"]
        if tid not in feats:
            dropped.append((tid, genre, "preprocessing_failed"))
            continue
        if genre not in label_map:
            dropped.append((tid, genre, "genre_not_in_label_map"))
            continue
        X_list.append(feats[tid])
        y_list.append(label_map[genre])
        kept_ids.append(tid)

    if not X_list:
        print("ERROR: no usable tracks after preprocessing. Aborting.")
        sys.exit(1)

    X = np.stack(X_list, axis=0)           # (N, 1, 128, 1292)
    y = np.array(y_list, dtype=np.int64)   # (N,)
    print(f"\nUsable tracks: {len(X)}   dropped: {len(dropped)}")
    if dropped:
        for tid, genre, reason in dropped[:10]:
            print(f"  dropped tid={tid} genre={genre} reason={reason}")
        if len(dropped) > 10:
            print(f"  ...and {len(dropped) - 10} more")

    # ── Load model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading checkpoint {ckpt_path}  (device={device})")
    model = CRNN(n_classes=args.n_classes, rnn_hidden=args.rnn_hidden).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ── Inference (batched) ──────────────────────────────────────────────────
    all_true, all_pred = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(X), args.batch_size):
            batch = torch.from_numpy(X[i:i + args.batch_size]).to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.extend(preds.tolist())
            all_true.extend(y[i:i + args.batch_size].tolist())
    infer_sec = time.perf_counter() - t0
    print(f"Inference on {len(X)} tracks: {infer_sec:.2f}s")

    # ── Metrics ──────────────────────────────────────────────────────────────
    accuracy = float(np.mean(np.array(all_pred) == np.array(all_true)))
    macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_true, all_pred, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(all_true, all_pred))

    val_metrics = {
        "macro_f1":          round(macro_f1, 6),
        "weighted_f1":       round(weighted_f1, 6),
        "balanced_accuracy": round(balanced_acc, 6),
        "per_class":         classification_report(
            all_true, all_pred, output_dict=True, zero_division=0,
        ),
    }

    # Confusion matrix CSV (rows=true, cols=pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t][p] += 1
    cm_df = pd.DataFrame(cm,
                         index=[f"true_{i}" for i in range(n_classes)],
                         columns=[str(i) for i in range(n_classes)])
    cm_df.index.name = "true\\pred"

    # ── Persist ──────────────────────────────────────────────────────────────
    run_summary = {
        "checkpoint":       str(ckpt_path),
        "audio_dir":        str(audio_dir),
        "sample_csv":       str(sample_csv),
        "n_tracks_manifest": int(len(sample)),
        "n_tracks_evaluated": int(len(X)),
        "n_tracks_dropped":  int(len(dropped)),
        "accuracy":          round(accuracy, 6),
        "inference_time_sec": round(infer_sec, 3),
        "device":            str(device),
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    # Per-track predictions for debugging
    pred_df = pd.DataFrame({
        "track_id":  kept_ids,
        "true_idx":  all_true,
        "pred_idx":  all_pred,
        "true":      [inv_label_map[t] for t in all_true],
        "pred":      [inv_label_map[p] for p in all_pred],
        "correct":   [t == p for t, p in zip(all_true, all_pred)],
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Results written to {out_dir}/")
    print(f"  accuracy:           {accuracy:.4f}")
    print(f"  macro-F1:           {macro_f1:.4f}")
    print(f"  weighted-F1:        {weighted_f1:.4f}")
    print(f"  balanced accuracy:  {balanced_acc:.4f}")
    print(f"  tracks evaluated:   {len(X)} / {len(sample)} (dropped {len(dropped)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
