# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

"""
Offline sequence packing script for creating pre-packed MDS datasets.

This script reads from multiple local MDS sources, tokenizes if needed, shuffles,
packs using greedy best-fit algorithm, and writes to MDS format.
MLM masking is NOT applied here - it must happen at runtime to vary per epoch.
"""

import argparse
import json
import os
import sys
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from streaming import MDSWriter
from streaming.base.format import reader_from_json
from streaming.base.spanner import Spanner
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src folder root to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from sequence_packer import find_best_fit


def parse_args():
    parser = argparse.ArgumentParser(description="Pack sequences offline into MDS format")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="List of source paths in format /path/to/dataset:split. "
        "Repeating a source multiple times will oversample it.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory for packed MDS dataset")
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="Tokenizer name/path (required if input is text)"
    )
    parser.add_argument("--max_seq_len", type=int, required=True, help="Maximum sequence length for packing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--pack_length",
        type=int,
        default=None,
        help="Length of each packed sequence (default: max_seq_len * 32 for efficient packing)",
    )
    parser.add_argument("--pad_token_id", type=int, default=0, help="Pad token ID (default: 0)")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Buffer size for packing (default: 10000)")
    return parser.parse_args()


class OfflineSequencePacker:
    """Packs sequences offline using greedy best-fit algorithm."""

    def __init__(
        self,
        pack_length: int,
        pad_token_id: int = 0,
        buffer_size: int = 10000,
        seed: int = 42,
    ):
        self.pack_length = pack_length
        self.pad_token_id = pad_token_id
        self.buffer_size = buffer_size
        self.np_rng = np.random.default_rng(seed)
        self.buffer = deque()

    def add_sequences(self, sequences: List[np.ndarray]):
        """Add sequences to the buffer."""
        self.buffer.extend(sequences)

    def shuffle_buffer(self):
        """Shuffle all sequences in the buffer."""
        buffer_list = list(self.buffer)
        self.np_rng.shuffle(buffer_list)
        self.buffer = deque(buffer_list)

    def pack_batch(self, batch_size: int) -> Optional[Tuple[np.ndarray, List[List[int]], np.ndarray]]:
        """
        Pack sequences from buffer into a batch.

        Returns:
            (packed_sequences, cu_seqlens, attention_masks) or None if buffer is too small
        """
        if len(self.buffer) < batch_size:
            return None

        batch = np.full((batch_size, self.pack_length), self.pad_token_id, dtype=np.int64)
        attention_masks = np.zeros((batch_size, self.pack_length), dtype=np.int8)
        cu_seqlens = [[0] for _ in range(batch_size)]
        remaining_spaces = np.full((batch_size,), self.pack_length, dtype=np.int32)
        temp_buffer = []

        while self.buffer:
            seq = self.buffer.popleft()
            seq_len = len(seq)

            # Find the best fit (smallest space that can accommodate the sequence)
            best_fit_idx = find_best_fit(remaining_spaces, seq_len)
            if best_fit_idx != -1:
                end_pos = self.pack_length - remaining_spaces[best_fit_idx]
                batch[best_fit_idx, end_pos : end_pos + seq_len] = seq
                attention_masks[best_fit_idx, end_pos : end_pos + seq_len] = 1
                remaining_spaces[best_fit_idx] -= seq_len
                cu_seqlens[best_fit_idx].append(cu_seqlens[best_fit_idx][-1] + seq_len)
            else:
                # Can't fit the sequence, save for next batch
                temp_buffer.append(seq)

        # Add any sequences we skipped back to the start of the buffer
        self.buffer.extendleft(temp_buffer)

        # Finalize cu_seqlens by adding final position if needed
        for x in cu_seqlens:
            if x[-1] != self.pack_length:
                x.append(self.pack_length)

        return batch, cu_seqlens, attention_masks

    def pack_all(self) -> List[Tuple[np.ndarray, List[int], np.ndarray]]:
        """Pack all sequences in buffer and return individual packed sequences."""
        packed_samples = []

        # Pack in batches for efficiency
        batch_size = 128
        while len(self.buffer) >= batch_size // 2:  # Continue while we have enough sequences
            result = self.pack_batch(batch_size)
            if result is None:
                # Not enough sequences for a full batch, pack remaining individually
                break

            batch, cu_seqlens_batch, attention_masks = result

            # Convert batch to individual samples
            for i in range(batch_size):
                packed_samples.append((batch[i], cu_seqlens_batch[i], attention_masks[i]))

        # Pack any remaining sequences
        while len(self.buffer) > 0:
            result = self.pack_batch(min(len(self.buffer), batch_size))
            if result is None:
                # Add remaining sequences as individual packed sequences with padding
                while self.buffer:
                    seq = self.buffer.popleft()
                    seq_len = len(seq)
                    packed = np.full(self.pack_length, self.pad_token_id, dtype=np.int64)
                    packed[:seq_len] = seq
                    cu_seqlens = [0, seq_len, self.pack_length]
                    attention_mask = np.zeros(self.pack_length, dtype=np.int8)
                    attention_mask[:seq_len] = 1
                    packed_samples.append((packed, cu_seqlens, attention_mask))
                break

            batch, cu_seqlens_batch, attention_masks = result
            actual_batch_size = len(cu_seqlens_batch)
            for i in range(actual_batch_size):
                packed_samples.append((batch[i], cu_seqlens_batch[i], attention_masks[i]))

        return packed_samples


def load_sequences_from_mds(local: str, split: Optional[str], tokenizer=None, max_seq_len: int = 1024) -> List[np.ndarray]:
    """Load sequences from an MDS dataset."""
    sequences = []

    if split is not None:
        split_path = os.path.join(local, split)
    else:
        split_path = local

    index_file_path = os.path.join(split_path, "index.json")
    if not os.path.exists(index_file_path):
        raise ValueError(f"Index file not found: {index_file_path}")

    obj = json.load(open(index_file_path))
    shards = []
    for info in obj["shards"]:
        shard = reader_from_json(local, split, info)
        raw_filename = os.path.join(shard.dirname, shard.split if shard.split else "", shard.raw_data.basename)
        if not os.path.isfile(raw_filename):
            raise ValueError(f"Raw file {raw_filename} does not exist")
        shard.validate(True)
        shards.append(shard)

    samples_per_shard = np.array([shard.samples for shard in shards], np.int64)
    total_samples = samples_per_shard.sum()
    spanner = Spanner(samples_per_shard)

    for index in range(total_samples):
        shard_id, shard_sample_id = spanner[index]
        shard = shards[shard_id]
        sample = shard[shard_sample_id]

        if "input_ids" in sample:
            # Pre-tokenized data
            input_ids = np.frombuffer(sample["input_ids"], dtype=np.int64).copy()
            input_ids = input_ids[:max_seq_len]  # Truncate if needed
            if len(input_ids) > 0:
                sequences.append(input_ids)
        elif "text" in sample:
            # Text data - need to tokenize
            if tokenizer is None:
                raise ValueError("Tokenizer required for text data")
            encoded = tokenizer(sample["text"], truncation=True, max_length=max_seq_len, add_special_tokens=True)
            input_ids = np.array(encoded["input_ids"], dtype=np.int64)
            if len(input_ids) > 0:
                sequences.append(input_ids)
        else:
            raise ValueError("Sample must contain 'input_ids' or 'text'")

    return sequences


def main():
    args = parse_args()

    # Initialize tokenizer if provided
    tokenizer = None
    if args.tokenizer:
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Determine pack length
    pack_length = args.pack_length if args.pack_length else args.max_seq_len * 32

    print(f"Loading sequences from {len(args.sources)} sources...")
    all_sequences = []

    # Load sequences from all sources
    for source_spec in args.sources:
        # Parse source specification (format: /path/to/dataset:split)
        if ":" in source_spec:
            local, split = source_spec.rsplit(":", 1)
        else:
            local = source_spec
            split = None

        print(f"  Loading from {local} (split: {split})...")
        sequences = load_sequences_from_mds(local, split, tokenizer, args.max_seq_len)
        all_sequences.extend(sequences)
        print(f"    Loaded {len(sequences)} sequences")

    print(f"Total sequences loaded: {len(all_sequences)}")

    # Initialize packer
    packer = OfflineSequencePacker(
        pack_length=pack_length,
        pad_token_id=args.pad_token_id,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )

    # Add all sequences to packer
    print("Adding sequences to packer...")
    packer.add_sequences(all_sequences)

    # Shuffle
    print("Shuffling sequences...")
    packer.shuffle_buffer()

    # Pack sequences
    print("Packing sequences...")
    packed_samples = packer.pack_all()
    print(f"Created {len(packed_samples)} packed samples")

    # Write to MDS format
    print(f"Writing packed dataset to {args.output}...")
    os.makedirs(args.output, exist_ok=True)

    columns = {
        "input_ids": "bytes",  # Will store as int64 bytes
        "cu_seqlens": "bytes",  # Will store as int32 bytes
        "attention_mask": "bytes",  # Will store as int8 bytes
    }

    with MDSWriter(columns=columns, out=args.output, compression=None) as writer:
        for input_ids, cu_seqlens, attention_mask in tqdm(packed_samples, desc="Writing samples"):
            # Convert cu_seqlens list to numpy array
            cu_seqlens_array = np.array(cu_seqlens, dtype=np.int32)

            writer.write({
                "input_ids": input_ids.tobytes(),
                "cu_seqlens": cu_seqlens_array.tobytes(),
                "attention_mask": attention_mask.tobytes(),
            })

    print("Done!")
    print(f"Packed dataset written to: {args.output}")
    print(f"Pack length: {pack_length}")
    print(f"Number of packed samples: {len(packed_samples)}")


if __name__ == "__main__":
    main()
