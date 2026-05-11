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
# ]
# ///
"""Tokenize a dataset on HF Jobs (CPU) and push the MDS shards to a remote host over SSH.

Runs convert_dataset.py from this repo and writes Mosaic streaming MDS shards,
then rsync's the result to a remote SSH destination. The remote may have no
outbound internet, so this script is one-way: cloud -> remote.

All knobs are read from environment variables (HF Jobs injects them via
`--env` and `--secrets`). REMOTE_SSH_PASSWORD must be passed via `--secrets`,
never baked into source.

Required env vars:
  REMOTE_HOST, REMOTE_USER, REMOTE_DEST_BASE, REMOTE_SSH_PASSWORD, DATASET

Optional env vars (with defaults):
  DATA_SUBSET, SPLITS (default "train"), CONCAT_TOKENS (default "1024"),
  TOKENIZER, EOS_TEXT (default "[EOS]"), BOS_TEXT, NO_WRAP,
  REPO_URL, REPO_BRANCH, UNIQUE_SUBDIR
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/Cosmos-YTU/ModernBERT.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "main")
REPO_DIR = "/tmp/ModernBERT"
OUT_DIR = "/tmp/out"

DATASET = os.environ["DATASET"]
DATA_SUBSET = os.environ.get("DATA_SUBSET") or None
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
    UNIQUE_SUBDIR = f"{ds_slug}__ctx{CONCAT_TOKENS}"
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


def main():
    pw = os.environ.get("REMOTE_SSH_PASSWORD")
    if not pw:
        print("FATAL: REMOTE_SSH_PASSWORD not in environment", flush=True)
        sys.exit(2)
    print(f"REMOTE_SSH_PASSWORD: present, length={len(pw)} (value redacted)", flush=True)
    print(f"DATASET={DATASET}  SUBSET={DATA_SUBSET}  SPLITS={SPLITS}", flush=True)
    print(f"TOKENIZER={TOKENIZER}  EOS_TEXT={EOS_TEXT}  CONCAT_TOKENS={CONCAT_TOKENS}", flush=True)

    print("\n=== Installing sshpass + rsync + git ===", flush=True)
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "sshpass", "rsync", "openssh-client", "git"])

    print("\n=== Cloning repo ===", flush=True)
    backoff = 5
    for attempt in range(1, 6):
        if os.path.exists(REPO_DIR):
            shutil.rmtree(REPO_DIR)
        try:
            run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, REPO_DIR])
            break
        except subprocess.CalledProcessError as e:
            if attempt == 5:
                raise
            print(f"clone failed (attempt {attempt}/5): {e}; sleeping {backoff}s", flush=True)
            time.sleep(backoff)
            backoff *= 2
    head = run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"], capture=True)
    print(f"HEAD: {head.stdout.strip()}", flush=True)

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
    ]
    if DATA_SUBSET:
        cmd += ["--data_subset", DATA_SUBSET]
    if BOS_TEXT:
        cmd += ["--bos_text", BOS_TEXT]
    if NO_WRAP:
        cmd += ["--no_wrap"]

    env = os.environ.copy()
    env["SSHPASS"] = pw
    ssh_ctl = "/tmp/ssh-bsc.sock"
    ssh_opts = (
        "sshpass -e ssh "
        f"-o ControlMaster=auto "
        f"-o ControlPath={ssh_ctl} "
        f"-o ControlPersist=8h "
        "-o StrictHostKeyChecking=accept-new "
        "-o UserKnownHostsFile=/dev/null "
        "-o LogLevel=ERROR"
    )
    # First ssh call opens the ControlMaster socket; all later rsyncs reuse it
    # over the same TCP/SSH connection (no re-auth, no fail2ban risk).
    run([
        "bash", "-c",
        f"{ssh_opts} {REMOTE_USER}@{REMOTE_HOST} 'mkdir -p {REMOTE_DEST}'"
    ], env=env)

    manifest = []
    pushed = set()
    manifest_lock = threading.Lock()
    transfer_bytes = [0]
    transfer_seconds = [0.0]

    def push_paths(paths):
        out_root = Path(OUT_DIR)
        new_entries = []
        rel_list = []
        for p in paths:
            try:
                rel = str(p.relative_to(out_root))
            except ValueError:
                continue
            if rel in pushed or not p.is_file():
                continue
            try:
                sz = p.stat().st_size
            except FileNotFoundError:
                continue
            digest = sha256_file(p)
            new_entries.append((rel, sz, digest))
            rel_list.append(rel)
        if not rel_list:
            return
        with tempfile.NamedTemporaryFile("w", delete=False) as ff:
            ff.write("\n".join(rel_list) + "\n")
            ff_path = ff.name
        try:
            t0 = time.time()
            run([
                "rsync", "-az", "--remove-source-files",
                f"--files-from={ff_path}",
                f"--rsh={ssh_opts}",
                f"{OUT_DIR}/",
                f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DEST}/",
            ], env=env)
            transfer_seconds[0] += time.time() - t0
        finally:
            os.unlink(ff_path)
        with manifest_lock:
            manifest.extend(new_entries)
            transfer_bytes[0] += sum(e[1] for e in new_entries)
            for rel in rel_list:
                pushed.add(rel)
        print(f"streamed {len(rel_list)} shard(s) to BSC (latest: {rel_list[-1]})", flush=True)

    def list_finalized_shards():
        """Shards with a higher-numbered sibling are finalized by MDSWriter."""
        out_root = Path(OUT_DIR)
        finalized = []
        if not out_root.exists():
            return finalized
        for split_dir in out_root.iterdir():
            if not split_dir.is_dir():
                continue
            shards = sorted(split_dir.glob("shard.*.mds"))
            if len(shards) < 2:
                continue
            finalized.extend(shards[:-1])
        return finalized

    stop_evt = threading.Event()

    PUSH_TICK_SECONDS = 300

    def watcher_loop():
        while not stop_evt.is_set():
            try:
                push_paths(list_finalized_shards())
            except Exception as e:
                print(f"watcher push error (will retry): {e}", flush=True)
            stop_evt.wait(PUSH_TICK_SECONDS)

    watcher = threading.Thread(target=watcher_loop, name="shard-pusher", daemon=True)
    watcher.start()
    print(f"=== Started background shard pusher ({PUSH_TICK_SECONDS}s tick, SSH ControlMaster) ===", flush=True)

    t_tok_start = time.time()
    try:
        run(cmd)
    finally:
        stop_evt.set()
        watcher.join(timeout=60)
    t_tok = time.time() - t_tok_start
    print(f"TOKENIZE_WALLCLOCK_S={t_tok:.2f}", flush=True)

    print("\n=== Final flush (latest shards + index.json) ===", flush=True)
    out_root = Path(OUT_DIR)
    remaining = sorted([p for p in out_root.rglob("*") if p.is_file()])
    push_paths(remaining)

    total_bytes = transfer_bytes[0]
    t_xfer = transfer_seconds[0]
    mb = total_bytes / (1024 * 1024)
    mbps = mb / t_xfer if t_xfer > 0 else 0.0
    print(f"FILE_COUNT={len(manifest)}", flush=True)
    print(f"TOTAL_BYTES={total_bytes}", flush=True)
    print(f"TRANSFER_WALLCLOCK_S={t_xfer:.2f}", flush=True)
    print(f"TRANSFER_MB={mb:.3f}", flush=True)
    print(f"TRANSFER_MBPS={mbps:.3f}", flush=True)

    print("\n=== SHA256 MANIFEST (sha256  size  path) ===", flush=True)
    print("---MANIFEST-BEGIN---", flush=True)
    for rel, sz, digest in sorted(manifest):
        print(f"{digest}  {sz}  {rel}", flush=True)
    print("---MANIFEST-END---", flush=True)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
