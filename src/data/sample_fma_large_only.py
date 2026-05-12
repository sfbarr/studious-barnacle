"""Pick FMA Large-only tracks (excluding Small/Medium) for an unseen test set.

A track's `(set, subset)` field is the SMALLEST subset it belongs to, so
filtering for "large" gives the Large-minus-Medium-minus-Small slice.
"""

from pathlib import Path
import pandas as pd

METADATA_PATH = Path("data/fma_metadata/tracks.csv")
OUTPUT_CSV    = Path("data/fma_large_test_sample.csv")
N_PER_GENRE   = 100        # capped per-genre by min(N, available)
SEED          = 671

TARGET_GENRES = [
    "Blues", "Classical", "Country", "Easy Listening",
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Jazz", "Old-Time / Historic",
    "Pop", "Rock", "Soul-RnB", "Spoken",
]

tracks = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])

large_only = tracks[tracks[("set", "subset")] == "large"].copy()
large_16   = large_only[large_only[("track", "genre_top")].isin(TARGET_GENRES)]

# Flatten into a plain single-level-column DataFrame before sampling.
flat = pd.DataFrame({
    "genre":  large_16[("track", "genre_top")].values,
    "title":  large_16[("track", "title")].values,
    "artist": large_16[("artist", "name")].values,
}, index=large_16.index)

parts = []
for genre, group in flat.groupby("genre"):
    parts.append(group.sample(n=min(N_PER_GENRE, len(group)), random_state=SEED))
sample = pd.concat(parts).sort_index()

print(f"Large-only tracks total:                  {len(large_only):>7,}")
print(f"...with a top-level genre in our 16:      {len(flat):>7,}")
print(f"Sampled (up to {N_PER_GENRE} per genre):              {len(sample):>7,}")
print()

available = flat["genre"].value_counts().rename("available")
sampled   = sample["genre"].value_counts().rename("sampled")
counts = pd.concat([available, sampled], axis=1).fillna(0).astype(int).sort_values("sampled", ascending=False)
print("Per-genre availability and sample count:")
print(counts.to_string())
print()

out = pd.DataFrame({
    "track_id": sample.index,
    "genre":    sample["genre"].values,
    "title":    sample["title"].values,
    "artist":   sample["artist"].values,
})
out.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {OUTPUT_CSV} ({len(out)} rows)")
