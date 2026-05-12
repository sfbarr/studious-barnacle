"""Download FMA Large MP3s for the sampled test set via the Kaggle CLI/API.

Reads track IDs from data/fma_large_test_sample.csv (produced by
sample_fma_large_only.py), derives each file's path inside the
`brunosette/fma-large` Kaggle dataset, and downloads the MP3 into
data/fma_large_test/NNN/NNNNNN.mp3 -- same directory layout as
data/fma_medium/, so the existing preprocessing pipeline drops in unchanged.
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

SAMPLE_CSV = Path("data/fma_large_test_sample.csv")
OUT_DIR    = Path("data/fma_large_test")
DATASET    = "brunosette/fma-large"

def main():
    if not SAMPLE_CSV.exists():
        print(f"Missing {SAMPLE_CSV}. Run sample_fma_large_only.py first.")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    sample = pd.read_csv(SAMPLE_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(sample)} tracks from {DATASET} -> {OUT_DIR}/\n")

    ok, fail, skip = 0, 0, 0
    for i, row in sample.iterrows():
        tid    = int(row["track_id"])
        padded = f"{tid:06d}"
        folder = padded[:3]
        fname  = f"{padded}.mp3"
        rel    = f"{folder}/{fname}"

        dest_dir  = OUT_DIR / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / fname

        prefix = f"  [{i+1:>3}/{len(sample)}] {rel}"

        if dest_file.exists() and dest_file.stat().st_size > 0:
            print(f"{prefix} skip (exists, {dest_file.stat().st_size:,} bytes)")
            skip += 1
            continue

        try:
            api.dataset_download_file(DATASET, rel, path=str(dest_dir))

            # Kaggle may wrap single-file downloads in a .zip; unwrap if so.
            zip_path = dest_dir / f"{fname}.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(dest_dir)
                zip_path.unlink()

            if dest_file.exists() and dest_file.stat().st_size > 0:
                print(f"{prefix} ok ({dest_file.stat().st_size:,} bytes)")
                ok += 1
            else:
                print(f"{prefix} FAIL (no file landed)")
                fail += 1
        except Exception as e:
            print(f"{prefix} FAIL: {e}")
            fail += 1

    print(f"\nDone. ok={ok}  skip={skip}  fail={fail}")

if __name__ == "__main__":
    main()
