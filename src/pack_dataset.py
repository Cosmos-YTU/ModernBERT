#!/usr/bin/env python3
# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

"""Pack pre-tokenized MDS datasets for efficient streaming training."""

import argparse
import json
import os
import sys
from typing import List

import numpy as np
from streaming import MDSWriter
from streaming.base.format import reader_from_json
from streaming.base.spanner import Spanner
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Pack pre-tokenized MDS dataset")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input tokenized MDS dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output packed MDS dataset")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer name or path")
    parser.add_argument("--max_seq_len", type=int, required=True, help="Maximum sequence length for packing")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to pack")
    return parser.parse_args()


def find_best_fit(remaining_spaces: np.ndarray, seq_len: int) -> int:
    """Find the best fitting space for a sequence using greedy best-fit algorithm.
    
    Args:
        remaining_spaces: Array of remaining space in each packed sequence
        seq_len: Length of sequence to fit
        
    Returns:
        Index of best fitting space, or -1 if no space fits
    """
    valid_spaces = seq_len <= remaining_spaces
    if np.any(valid_spaces):
        valid_space_sizes = remaining_spaces[valid_spaces]
        best_fit_idx = np.argmin(valid_space_sizes)
        return np.arange(len(remaining_spaces))[valid_spaces][best_fit_idx]
    return -1


def pack_sequences(sequences: List[np.ndarray], max_seq_len: int, eos_token_id: int, pad_token_id: int) -> List[np.ndarray]:
    """Pack sequences using greedy best-fit algorithm.
    
    Args:
        sequences: List of tokenized sequences to pack
        max_seq_len: Maximum sequence length for each packed sequence
        eos_token_id: EOS token ID to separate documents
        pad_token_id: Padding token ID
        
    Returns:
        List of packed sequences, each of length max_seq_len
    """
    packed_sequences = []
    current_batch = []
    remaining_spaces = np.array([], dtype=np.int32)
    
    # Start with at least one packed sequence
    current_batch.append(np.full(max_seq_len, pad_token_id, dtype=np.int64))
    remaining_spaces = np.array([max_seq_len], dtype=np.int32)
    
    for seq in tqdm(sequences, desc="Packing sequences"):
        # Add EOS token to sequence
        seq_with_eos = np.append(seq, eos_token_id).astype(np.int64)
        seq_len = len(seq_with_eos)
        
        if seq_len > max_seq_len:
            # Skip sequences that are too long
            continue
            
        # Find best fit
        best_fit_idx = find_best_fit(remaining_spaces, seq_len)
        
        if best_fit_idx != -1:
            # Fit sequence into existing packed sequence
            end_pos = max_seq_len - remaining_spaces[best_fit_idx]
            current_batch[best_fit_idx][end_pos:end_pos + seq_len] = seq_with_eos
            remaining_spaces[best_fit_idx] -= seq_len
        else:
            # Need a new packed sequence
            new_packed = np.full(max_seq_len, pad_token_id, dtype=np.int64)
            new_packed[:seq_len] = seq_with_eos
            current_batch.append(new_packed)
            remaining_spaces = np.append(remaining_spaces, max_seq_len - seq_len)
    
    return current_batch


def main():
    args = parse_args()
    
    # Load tokenizer to get special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    if eos_token_id is None:
        raise ValueError(f"Tokenizer {args.tokenizer} has no EOS token")
    if pad_token_id is None:
        raise ValueError(f"Tokenizer {args.tokenizer} has no PAD token")
    
    print(f"Using EOS token ID: {eos_token_id}")
    print(f"Using PAD token ID: {pad_token_id}")
    
    # Read input dataset
    split_path = os.path.join(args.input_path, args.split) if args.split else args.input_path
    index_file_path = os.path.join(split_path, "index.json")
    
    if not os.path.exists(index_file_path):
        raise ValueError(f"Index file not found: {index_file_path}")
    
    print(f"Reading from: {split_path}")
    
    obj = json.load(open(index_file_path))
    shards = []
    for info in obj["shards"]:
        shard = reader_from_json(args.input_path, args.split, info)
        raw_filename = os.path.join(shard.dirname, shard.split, shard.raw_data.basename)
        if not os.path.isfile(raw_filename):
            raise ValueError(f"Raw file {raw_filename} does not exist")
        shard.validate(True)
        shards.append(shard)
    
    samples_per_shard = np.array([shard.samples for shard in shards], np.int64)
    total_samples = samples_per_shard.sum()
    spanner = Spanner(samples_per_shard)
    
    print(f"Found {total_samples} samples in {len(shards)} shards")
    
    # Read all sequences
    sequences = []
    truncated_count = 0
    for idx in tqdm(range(total_samples), desc="Reading sequences"):
        shard_id, shard_sample_id = spanner[idx]
        shard = shards[shard_id]
        sample = shard[shard_sample_id]
        
        if "input_ids" not in sample:
            raise ValueError(f"Sample {idx} does not have 'input_ids' field")
        
        input_ids = sample["input_ids"]
        if isinstance(input_ids, bytes):
            input_ids = np.frombuffer(input_ids, dtype=np.int64)
        
        # Truncate to max_seq_len if needed
        if len(input_ids) > args.max_seq_len:
            input_ids = input_ids[:args.max_seq_len]
            truncated_count += 1
        
        # Skip empty sequences
        if len(input_ids) > 0:
            sequences.append(input_ids)
    
    if truncated_count > 0:
        print(f"Warning: Truncated {truncated_count} sequences to max_seq_len={args.max_seq_len}")
    
    print(f"Read {len(sequences)} non-empty sequences")
    
    # Pack sequences
    packed_sequences = pack_sequences(sequences, args.max_seq_len, eos_token_id, pad_token_id)
    
    print(f"Created {len(packed_sequences)} packed sequences")
    
    # Write packed dataset
    output_split_path = os.path.join(args.output_path, args.split) if args.split else args.output_path
    os.makedirs(output_split_path, exist_ok=True)
    
    print(f"Writing to: {output_split_path}")
    
    columns = {
        "input_ids": "bytes",
    }
    
    with MDSWriter(out=output_split_path, columns=columns, compression=None) as writer:
        for packed_seq in tqdm(packed_sequences, desc="Writing packed sequences"):
            writer.write({
                "input_ids": packed_seq.tobytes(),
            })
    
    print(f"Done! Wrote {len(packed_sequences)} packed sequences to {output_split_path}")


if __name__ == "__main__":
    main()
