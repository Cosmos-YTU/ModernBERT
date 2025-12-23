#!/usr/bin/env python3
# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

"""Integration test for pre-packed dataset functionality."""

import os
import sys
import tempfile
import shutil

import numpy as np
from streaming import MDSWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_tokenized_dataset(output_path, num_samples=20, max_seq_len=50):
    """Create a small tokenized MDS dataset for testing.
    
    Args:
        output_path: Path to write test dataset
        num_samples: Number of samples to create
        max_seq_len: Maximum sequence length
    """
    os.makedirs(output_path, exist_ok=True)
    
    columns = {"input_ids": "bytes"}
    
    with MDSWriter(out=output_path, columns=columns, compression=None) as writer:
        for i in range(num_samples):
            # Create random sequences of varying lengths
            seq_len = np.random.randint(10, max_seq_len + 1)
            # Use token IDs 10-100 to avoid special tokens 0, 1, 2
            input_ids = np.random.randint(10, 100, size=seq_len, dtype=np.int64)
            
            writer.write({
                "input_ids": input_ids.tobytes(),
            })
    
    print(f"Created test dataset with {num_samples} samples at {output_path}")


def test_pack_and_load_dataset():
    """Test the full workflow: create dataset -> pack -> load with PrePackedCollator."""
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    try:
        tokenized_path = os.path.join(temp_dir, "tokenized")
        packed_path = os.path.join(temp_dir, "packed")
        
        # Step 1: Create test tokenized dataset
        print("\n=== Step 1: Creating test tokenized dataset ===")
        create_test_tokenized_dataset(tokenized_path, num_samples=20, max_seq_len=50)
        
        # Step 2: Pack the dataset
        print("\n=== Step 2: Packing dataset ===")
        from src.pack_dataset import pack_sequences
        from streaming.base.format import reader_from_json
        from streaming.base.spanner import Spanner
        import json
        
        # Read the tokenized dataset
        index_file_path = os.path.join(tokenized_path, "index.json")
        obj = json.load(open(index_file_path))
        shards = []
        for info in obj["shards"]:
            shard = reader_from_json(tokenized_path, None, info)
            shards.append(shard)
        
        samples_per_shard = np.array([shard.samples for shard in shards], np.int64)
        total_samples = samples_per_shard.sum()
        spanner = Spanner(samples_per_shard)
        
        # Read sequences
        sequences = []
        for idx in range(total_samples):
            shard_id, shard_sample_id = spanner[idx]
            shard = shards[shard_id]
            sample = shard[shard_sample_id]
            input_ids = np.frombuffer(sample["input_ids"], dtype=np.int64)
            sequences.append(input_ids)
        
        print(f"Read {len(sequences)} sequences")
        
        # Pack sequences
        eos_token_id = 2
        pad_token_id = 0
        max_seq_len = 200  # Pack to max_seq_len
        
        packed_sequences = pack_sequences(sequences, max_seq_len, eos_token_id, pad_token_id)
        print(f"Created {len(packed_sequences)} packed sequences")
        
        # Write packed dataset
        os.makedirs(packed_path, exist_ok=True)
        columns = {"input_ids": "bytes"}
        
        from streaming import MDSWriter
        with MDSWriter(out=packed_path, columns=columns, compression=None) as writer:
            for packed_seq in packed_sequences:
                writer.write({
                    "input_ids": packed_seq.tobytes(),
                })
        
        print(f"Wrote packed dataset to {packed_path}")
        
        # Step 3: Load packed dataset with PrePackedCollator
        print("\n=== Step 3: Loading packed dataset with PrePackedCollator ===")
        from src.text_data import PrePackedCollator
        
        # Read packed dataset
        index_file_path = os.path.join(packed_path, "index.json")
        obj = json.load(open(index_file_path))
        shards = []
        for info in obj["shards"]:
            shard = reader_from_json(packed_path, None, info)
            shards.append(shard)
        
        samples_per_shard = np.array([shard.samples for shard in shards], np.int64)
        total_samples = samples_per_shard.sum()
        spanner = Spanner(samples_per_shard)
        
        # Create collator
        collator = PrePackedCollator(
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            mask_token_id=103,
            mlm_probability=0.15,
            ignore_token_id=-100,
        )
        
        # Load a batch
        batch_size = min(3, total_samples)
        examples = []
        for idx in range(batch_size):
            shard_id, shard_sample_id = spanner[idx]
            shard = shards[shard_id]
            sample = shard[shard_sample_id]
            examples.append(sample)
        
        batch = collator(examples)
        
        # Verify batch structure
        print("\n=== Step 4: Verifying batch structure ===")
        assert "input_ids" in batch, "Missing input_ids"
        assert "labels" in batch, "Missing labels"
        assert "attention_mask" in batch, "Missing attention_mask"
        assert "cu_seqlens" in batch, "Missing cu_seqlens"
        assert "max_seqlen" in batch, "Missing max_seqlen"
        
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
        print(f"Number of cu_seqlens: {len(batch['cu_seqlens'])}")
        print(f"Number of max_seqlens: {len(batch['max_seqlen'])}")
        
        # Verify shapes
        assert batch["input_ids"].shape[0] == batch_size
        assert batch["input_ids"].shape[1] == max_seq_len
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert len(batch["cu_seqlens"]) == batch_size
        assert len(batch["max_seqlen"]) == batch_size
        
        # Verify cu_seqlens structure
        for i, cu_seq in enumerate(batch["cu_seqlens"]):
            assert cu_seq[0] == 0, f"cu_seqlens[{i}] should start at 0"
            assert len(cu_seq) >= 2, f"cu_seqlens[{i}] should have at least 2 elements"
            print(f"  Sample {i} cu_seqlens length: {len(cu_seq)}, max_seqlen: {batch['max_seqlen'][i]}")
        
        print("\n✓ Integration test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    test_pack_and_load_dataset()
    print("\nAll integration tests passed!")
