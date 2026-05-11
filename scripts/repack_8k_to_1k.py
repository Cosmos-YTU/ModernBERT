# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mosaicml-streaming",
#   "numpy",
# ]
# ///
"""Repack an existing 8192-ctx MDS dataset into 1024-ctx samples.

Reads each 8192-token packed sample, slices into 8 chunks of 1024 tokens,
writes them as new MDS samples. No re-tokenization. Per-split resumable via
the .DONE marker file written when a split finishes.

Env vars:
  SRC_ROOT   source dataset root (with train/, val/ subdirs)
  DST_ROOT   destination dataset root (created if missing)
  SPLITS     space-separated splits to repack, default "train val"
  SRC_CTX    source ctx length, default 8192
  DST_CTX    destination ctx length, default 1024
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from streaming.base import MDSWriter, StreamingDataset

SRC_ROOT = os.environ["SRC_ROOT"]
DST_ROOT = os.environ["DST_ROOT"]
SPLITS = os.environ.get("SPLITS", "train val").split()
SRC_CTX = int(os.environ.get("SRC_CTX", "8192"))
DST_CTX = int(os.environ.get("DST_CTX", "1024"))

assert SRC_CTX % DST_CTX == 0, f"SRC_CTX={SRC_CTX} must be divisible by DST_CTX={DST_CTX}"
RATIO = SRC_CTX // DST_CTX


def repack_split(split: str) -> None:
    src_split = Path(SRC_ROOT) / split
    if not src_split.is_dir():
        print(f"SKIP {split}: no such dir at {src_split}", flush=True)
        return

    dst_split = Path(DST_ROOT) / split
    dst_split.mkdir(parents=True, exist_ok=True)

    done_marker = dst_split / ".DONE"
    if done_marker.exists():
        print(f"SKIP {split}: marker {done_marker} already present", flush=True)
        return

    ds = StreamingDataset(local=SRC_ROOT, split=split, shuffle=False, batch_size=None)
    n_in = len(ds)
    print(f"START {split}: {n_in} samples ({SRC_CTX} ctx) -> ~{n_in * RATIO} ({DST_CTX} ctx)", flush=True)

    t0 = time.time()
    n_out = 0
    with MDSWriter(columns={"tokens": "bytes"}, out=str(dst_split), compression=None) as out:
        for i, sample in enumerate(ds):
            arr = np.frombuffer(sample["tokens"], dtype=np.int64)
            if arr.size != SRC_CTX:
                raise ValueError(f"sample {i} has {arr.size} tokens, expected {SRC_CTX}")
            for j in range(RATIO):
                chunk = arr[j * DST_CTX : (j + 1) * DST_CTX]
                out.write({"tokens": chunk.tobytes()})
                n_out += 1
            if i and i % 1000 == 0:
                rate = i / (time.time() - t0)
                eta = (n_in - i) / rate
                print(f"  {split} {i}/{n_in} | rate={rate:.1f}/s | eta={eta:.0f}s | n_out={n_out}", flush=True)

    elapsed = time.time() - t0
    print(f"DONE {split}: wrote {n_out} samples in {elapsed:.1f}s", flush=True)
    done_marker.write_text(json.dumps({"src_samples": n_in, "dst_samples": n_out, "wallclock_s": elapsed}))


def main() -> None:
    print(f"SRC_ROOT={SRC_ROOT}", flush=True)
    print(f"DST_ROOT={DST_ROOT}", flush=True)
    print(f"SPLITS={SPLITS}  SRC_CTX={SRC_CTX} -> DST_CTX={DST_CTX} (x{RATIO})", flush=True)
    for split in SPLITS:
        repack_split(split)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
