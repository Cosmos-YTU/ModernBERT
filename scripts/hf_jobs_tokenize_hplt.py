# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mosaicml-streaming",
#   "datasets",
#   "transformers",
#   "tokenizers",
#   "numpy",
#   "torch",
#   "tqdm",
#   "zstandard",
# ]
# ///
"""Tokenize HPLT 3.0 data on HF Jobs (CPU) and push MDS shards to a remote host.

HPLT 3.0 is not packaged as a normal HF dataset; data lives at
`data.hplt-project.org`. This script:
  1. Downloads the language-specific `.map` file listing all shard URLs.
  2. Filters URLs by quality bin (HPLT_TOP_BINS env var).
  3. Calls convert_dataset.py with --data_files <urls> so HF datasets streams
     directly from the URLs via the "json" loader (transparent zstd decode).
  4. Pushes the resulting MDS shards to a remote SSH destination.

Required env vars (besides the shared ones from hf_jobs_tokenize.py):
  HPLT_MAP_URL    .map URL, e.g. https://data.hplt-project.org/three/sorted/tur_Latn.map
  HPLT_TOP_BINS   comma-separated quality bins to include, e.g. "10" or "10,9"
  REMOTE_HOST, REMOTE_USER, REMOTE_DEST_BASE, REMOTE_SSH_PASSWORD

Optional:
  DATA_FILES_CAP  cap on number of URLs after filtering (default: no cap)
  DATASET         constants-key passed as --dataset (default: hplt3-tur_Latn)
  CONCAT_TOKENS   default "1024"
  SPLITS          default "train"
  Plus the same REPO_URL, REPO_BRANCH, TOKENIZER, EOS_TEXT, BOS_TEXT, NO_WRAP, UNIQUE_SUBDIR.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/Cosmos-YTU/ModernBERT.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "main")
REPO_DIR = "/tmp/ModernBERT"
OUT_DIR = "/tmp/out"

DATASET = os.environ.get("DATASET", "hplt3-tur_Latn")
HPLT_MAP_URL = os.environ["HPLT_MAP_URL"]
HPLT_TOP_BINS = [b.strip() for b in os.environ.get("HPLT_TOP_BINS", "10").split(",") if b.strip()]
DATA_FILES_CAP = int(os.environ["DATA_FILES_CAP"]) if os.environ.get("DATA_FILES_CAP") else None

SPLITS = os.environ.get("SPLITS", "train").split()
CONCAT_TOKENS = os.environ.get("CONCAT_TOKENS", "1024")
TOKENIZER = os.environ.get("TOKENIZER", "ytu-ce-cosmos/modernbert-tr-base-1k")
EOS_TEXT = os.environ.get("EOS_TEXT", "[EOS]")
BOS_TEXT = os.environ.get("BOS_TEXT") or None
NO_WRAP = os.environ.get("NO_WRAP", "0") == "1"

REMOTE_HOST = os.environ["REMOTE_HOST"]
REMOTE_USER = os.environ["REMOTE_USER"]
REMOTE_DEST_BASE = os.environ["REMOTE_DEST_BASE"]
UNIQUE_SUBDIR = os.environ.get("UNIQUE_SUBDIR")
if not UNIQUE_SUBDIR:
    ds_slug = DATASET.replace("/", "_")
    bins_slug = "-".join(HPLT_TOP_BINS)
    UNIQUE_SUBDIR = f"{ds_slug}__bins{bins_slug}__ctx{CONCAT_TOKENS}"
REMOTE_DEST = f"{REMOTE_DEST_BASE}/{UNIQUE_SUBDIR}"


def run(cmd, check=True, env=None, capture=False):
    print(f"$ {' '.join(cmd)}", flush=True)
    if capture:
        return subprocess.run(cmd, check=check, env=env, capture_output=True, text=True)
    return subprocess.run(cmd, check=check, env=env)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_urls() -> list[str]:
    """Fetch the .map file and return URLs matching HPLT_TOP_BINS."""
    print(f"\n=== Fetching HPLT map: {HPLT_MAP_URL} ===", flush=True)
    with urllib.request.urlopen(HPLT_MAP_URL, timeout=60) as r:
        data = r.read().decode("utf-8", errors="replace")
    all_urls = [line.strip() for line in data.splitlines() if line.strip()]
    print(f"map contains {len(all_urls)} URLs", flush=True)

    filtered: list[str] = []
    for url in all_urls:
        # URL pattern: .../<lang>/<bin>_<idx>.jsonl.zst
        basename = url.rsplit("/", 1)[-1]
        bin_str = basename.split("_", 1)[0]
        if bin_str in HPLT_TOP_BINS:
            filtered.append(url)
    # Sort by quality bin descending (10 first), then by shard index ascending
    filtered.sort(key=lambda u: (-int(u.rsplit("/", 1)[-1].split("_", 1)[0]), u))
    print(f"after filtering by bins={HPLT_TOP_BINS}: {len(filtered)} URLs", flush=True)
    for u in filtered:
        print(f"  {u}", flush=True)
    if DATA_FILES_CAP is not None and len(filtered) > DATA_FILES_CAP:
        filtered = filtered[:DATA_FILES_CAP]
        print(f"capped to first {DATA_FILES_CAP} URLs", flush=True)
    if not filtered:
        raise RuntimeError(f"No URLs matched HPLT_TOP_BINS={HPLT_TOP_BINS}")
    return filtered


def main():
    pw = os.environ.get("REMOTE_SSH_PASSWORD")
    if not pw:
        print("FATAL: REMOTE_SSH_PASSWORD not in environment", flush=True)
        sys.exit(2)
    print(f"REMOTE_SSH_PASSWORD: present, length={len(pw)} (value redacted)", flush=True)
    print(f"DATASET={DATASET}  CONCAT_TOKENS={CONCAT_TOKENS}  SPLITS={SPLITS}", flush=True)
    print(f"HPLT_MAP_URL={HPLT_MAP_URL}  HPLT_TOP_BINS={HPLT_TOP_BINS}", flush=True)
    print(f"REMOTE_DEST={REMOTE_DEST}", flush=True)

    print("\n=== Installing sshpass + rsync + git ===", flush=True)
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "sshpass", "rsync", "openssh-client", "git"])

    print("\n=== Cloning repo ===", flush=True)
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, REPO_DIR])
    head = run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"], capture=True)
    print(f"HEAD: {head.stdout.strip()}", flush=True)

    urls = fetch_urls()

    print("\n=== Running convert_dataset.py ===", flush=True)
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [
        sys.executable, f"{REPO_DIR}/src/convert_dataset.py",
        "--dataset", DATASET,
        "--out_root", OUT_DIR,
        "--splits", *SPLITS,
        "--concat_tokens", CONCAT_TOKENS,
        "--tokenizer", TOKENIZER,
        "--eos_text", EOS_TEXT,
        "--data_files", *urls,
    ]
    if BOS_TEXT:
        cmd += ["--bos_text", BOS_TEXT]
    if NO_WRAP:
        cmd += ["--no_wrap"]

    t_tok_start = time.time()
    run(cmd)
    t_tok = time.time() - t_tok_start
    print(f"TOKENIZE_WALLCLOCK_S={t_tok:.2f}", flush=True)

    print("\n=== Output listing & manifest ===", flush=True)
    out_root = Path(OUT_DIR)
    files = sorted([p for p in out_root.rglob("*") if p.is_file()])
    total_bytes = 0
    manifest = []
    for p in files:
        sz = p.stat().st_size
        total_bytes += sz
        digest = sha256_file(p)
        rel = p.relative_to(out_root)
        manifest.append((str(rel), sz, digest))
        print(f"  {digest}  {sz:>12d}  {rel}", flush=True)
    print(f"FILE_COUNT={len(files)}", flush=True)
    print(f"TOTAL_BYTES={total_bytes}", flush=True)

    print("\n=== Pushing to remote ===", flush=True)
    env = os.environ.copy()
    env["SSHPASS"] = pw
    ssh_opts = (
        "sshpass -e ssh "
        "-o StrictHostKeyChecking=accept-new "
        "-o UserKnownHostsFile=/dev/null "
        "-o LogLevel=ERROR"
    )
    run([
        "bash", "-c",
        f"sshpass -e ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "
        f"{REMOTE_USER}@{REMOTE_HOST} 'mkdir -p {REMOTE_DEST}'"
    ], env=env)

    t_xfer_start = time.time()
    run([
        "rsync", "-avz", "--info=progress2",
        f"--rsh={ssh_opts}",
        f"{OUT_DIR}/",
        f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DEST}/",
    ], env=env)
    t_xfer = time.time() - t_xfer_start
    mb = total_bytes / (1024 * 1024)
    mbps = mb / t_xfer if t_xfer > 0 else 0.0
    print(f"\nTRANSFER_WALLCLOCK_S={t_xfer:.2f}", flush=True)
    print(f"TRANSFER_MB={mb:.3f}", flush=True)
    print(f"TRANSFER_MBPS={mbps:.3f}", flush=True)

    print("\n=== SHA256 MANIFEST (sha256  size  path) ===", flush=True)
    print("---MANIFEST-BEGIN---", flush=True)
    for rel, sz, digest in manifest:
        print(f"{digest}  {sz}  {rel}", flush=True)
    print("---MANIFEST-END---", flush=True)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
