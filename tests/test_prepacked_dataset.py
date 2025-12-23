# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import os
import sys
import tempfile
import numpy as np
import torch

# Add tests folder root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming import MDSWriter

# Direct import - this should work since we're adding parent to path
# We need to be careful to only import what we need
import importlib
import importlib.util

# Manually load sequence_packer first to avoid full src init
spec = importlib.util.spec_from_file_location(
    "sequence_packer",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "sequence_packer.py")
)
sequence_packer = importlib.util.module_from_spec(spec)
sys.modules["sequence_packer"] = sequence_packer
spec.loader.exec_module(sequence_packer)

# Now we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

# Import text_data module components directly
from text_data import PrePackedStreamingDataset, PrePackedMLMCollator

# Import pack_dataset components
from pack_dataset import OfflineSequencePacker, load_sequences_from_mds


def create_test_mds_dataset(output_dir, num_samples=100, max_seq_len=128):
    """Create a simple test MDS dataset with pre-tokenized sequences."""
    os.makedirs(output_dir, exist_ok=True)
    
    columns = {
        "input_ids": "bytes",
    }
    
    # Create samples with varying lengths
    with MDSWriter(columns=columns, out=output_dir, compression=None) as writer:
        for i in range(num_samples):
            # Create sequences of varying length (10 to max_seq_len)
            seq_len = np.random.randint(10, max_seq_len + 1)
            # Use token IDs from 1 to 1000 (avoid 0 which is typically pad)
            input_ids = np.random.randint(1, 1000, size=seq_len, dtype=np.int64)
            
            writer.write({
                "input_ids": input_ids.tobytes(),
            })


def create_packed_test_dataset(output_dir, num_samples=10, pack_length=512):
    """Create a test packed MDS dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    columns = {
        "input_ids": "bytes",
        "cu_seqlens": "bytes",
        "attention_mask": "bytes",
    }
    
    with MDSWriter(columns=columns, out=output_dir, compression=None) as writer:
        for i in range(num_samples):
            # Create a packed sample with 2-4 sequences
            num_seqs = np.random.randint(2, 5)
            cu_seqlens = [0]
            input_ids = np.zeros(pack_length, dtype=np.int64)
            attention_mask = np.zeros(pack_length, dtype=np.int8)
            
            current_pos = 0
            for j in range(num_seqs):
                # Random sequence length
                seq_len = np.random.randint(50, 150)
                if current_pos + seq_len > pack_length:
                    seq_len = pack_length - current_pos
                if seq_len <= 0:
                    break
                    
                # Fill in the sequence
                input_ids[current_pos:current_pos + seq_len] = np.random.randint(1, 1000, size=seq_len)
                attention_mask[current_pos:current_pos + seq_len] = 1
                current_pos += seq_len
                cu_seqlens.append(current_pos)
            
            # Add final position if needed
            if cu_seqlens[-1] != pack_length:
                cu_seqlens.append(pack_length)
            
            cu_seqlens_array = np.array(cu_seqlens, dtype=np.int32)
            
            writer.write({
                "input_ids": input_ids.tobytes(),
                "cu_seqlens": cu_seqlens_array.tobytes(),
                "attention_mask": attention_mask.tobytes(),
            })


def test_offline_sequence_packer():
    """Test the OfflineSequencePacker."""
    pack_length = 100
    packer = OfflineSequencePacker(
        pack_length=pack_length,
        pad_token_id=0,
        buffer_size=100,
        seed=42,
    )
    
    # Create test sequences
    sequences = [
        np.array([1, 2, 3, 4, 5], dtype=np.int64),
        np.array([6, 7, 8], dtype=np.int64),
        np.array([9, 10], dtype=np.int64),
        np.array([11, 12, 13, 14], dtype=np.int64),
    ]
    
    packer.add_sequences(sequences)
    packer.shuffle_buffer()
    
    # Pack into a batch
    result = packer.pack_batch(batch_size=2)
    assert result is not None, "Should be able to pack a batch"
    
    batch, cu_seqlens_batch, attention_masks = result
    assert batch.shape == (2, pack_length), f"Expected batch shape (2, {pack_length}), got {batch.shape}"
    assert len(cu_seqlens_batch) == 2, "Expected 2 cu_seqlens"
    assert attention_masks.shape == (2, pack_length), f"Expected attention mask shape (2, {pack_length})"
    
    # Check that sequences are properly packed
    for i in range(2):
        cu_seqlens = cu_seqlens_batch[i]
        # cu_seqlens format: [0, end_of_seq1, end_of_seq2, ..., pack_length]
        # We verify attention mask for actual sequences (not the final pack_length marker)
        num_sequences = len(cu_seqlens) - 1  # Subtract 1 for the initial 0
        
        # Verify that non-padding positions have attention_mask = 1
        # Only check up to the second-to-last element (skip the final pack_length marker)
        CU_SEQLENS_FINAL_OFFSET = 2  # Offset to skip final pack_length marker in cu_seqlens
        for j in range(len(cu_seqlens) - CU_SEQLENS_FINAL_OFFSET):
            start = cu_seqlens[j]
            end = cu_seqlens[j + 1]
            if end > start:
                # Should have attention mask set for this range
                assert np.all(attention_masks[i, start:end] == 1), \
                    f"Attention mask should be 1 for packed sequence range [{start}:{end}]"
        
        # Check that padding positions have attention_mask = 0
        # The last sequence ends at cu_seqlens[-2] (the position before the final pack_length marker)
        last_seq_end = cu_seqlens[-CU_SEQLENS_FINAL_OFFSET]
        if last_seq_end < pack_length:
            # There should be padding
            assert np.all(attention_masks[i, last_seq_end:pack_length] == 0), \
                f"Attention mask should be 0 for padding range [{last_seq_end}:{pack_length}]"
    
    print("✓ OfflineSequencePacker test passed")


def test_prepacked_streaming_dataset():
    """Test PrePackedStreamingDataset reading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test packed dataset
        create_packed_test_dataset(tmpdir, num_samples=10, pack_length=512)
        
        # Load with PrePackedStreamingDataset
        dataset = PrePackedStreamingDataset(
            local=tmpdir,
            split=None,
            shuffle=False,
        )
        
        # Test reading a sample
        sample = dataset[0]
        assert "input_ids" in sample, "Sample should have input_ids"
        assert "cu_seqlens" in sample, "Sample should have cu_seqlens"
        assert "attention_mask" in sample, "Sample should have attention_mask"
        
        # Check types
        assert isinstance(sample["input_ids"], np.ndarray), "input_ids should be numpy array"
        assert isinstance(sample["cu_seqlens"], np.ndarray), "cu_seqlens should be numpy array"
        assert isinstance(sample["attention_mask"], np.ndarray), "attention_mask should be numpy array"
        
        # Check dtypes
        assert sample["input_ids"].dtype == np.int64, "input_ids should be int64"
        assert sample["cu_seqlens"].dtype == np.int32, "cu_seqlens should be int32"
        assert sample["attention_mask"].dtype == np.int8, "attention_mask should be int8"
        
        # Check shapes
        assert len(sample["input_ids"]) == 512, "input_ids should have pack_length elements"
        assert len(sample["attention_mask"]) == 512, "attention_mask should have pack_length elements"
        
        print("✓ PrePackedStreamingDataset test passed")


def test_prepacked_mlm_collator():
    """Test PrePackedMLMCollator masking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test packed dataset
        pack_length = 512
        create_packed_test_dataset(tmpdir, num_samples=5, pack_length=pack_length)
        
        # Load with PrePackedStreamingDataset
        dataset = PrePackedStreamingDataset(
            local=tmpdir,
            split=None,
            shuffle=False,
        )
        
        # Create collator
        mask_token_id = 103
        pad_token_id = 0
        mlm_probability = 0.15
        
        collator = PrePackedMLMCollator(
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            mlm_probability=mlm_probability,
        )
        
        # Collate a batch
        batch_size = 3
        examples = [dataset[i] for i in range(batch_size)]
        batch = collator(examples)
        
        # Check output format
        assert "input_ids" in batch, "Batch should have input_ids"
        assert "labels" in batch, "Batch should have labels"
        assert "cu_seqlens" in batch, "Batch should have cu_seqlens"
        assert "max_seqlen" in batch, "Batch should have max_seqlen"
        assert "attention_mask" in batch, "Batch should have attention_mask"
        
        # Check shapes
        assert batch["input_ids"].shape == (batch_size, pack_length), \
            f"input_ids should be shape ({batch_size}, {pack_length})"
        assert batch["labels"].shape == (batch_size, pack_length), \
            f"labels should be shape ({batch_size}, {pack_length})"
        assert batch["attention_mask"].shape == (batch_size, pack_length), \
            f"attention_mask should be shape ({batch_size}, {pack_length})"
        
        # Check types
        assert isinstance(batch["input_ids"], torch.Tensor), "input_ids should be tensor"
        assert isinstance(batch["labels"], torch.Tensor), "labels should be tensor"
        assert isinstance(batch["attention_mask"], torch.Tensor), "attention_mask should be tensor"
        
        # Check that masking was applied
        # At least some tokens should be masked (replaced with mask_token_id)
        num_masked = (batch["input_ids"] == mask_token_id).sum().item()
        assert num_masked > 0, "Some tokens should be masked"
        
        # Check that labels have ignore_index (-100) for non-masked positions
        ignore_count = (batch["labels"] == -100).sum().item()
        assert ignore_count > 0, "Some labels should be set to ignore_index"
        
        # Verify cu_seqlens is a list of tensors
        assert isinstance(batch["cu_seqlens"], list), "cu_seqlens should be a list"
        assert len(batch["cu_seqlens"]) == batch_size, "cu_seqlens should have batch_size elements"
        assert all(isinstance(x, torch.Tensor) for x in batch["cu_seqlens"]), \
            "Each cu_seqlens element should be a tensor"
        
        # Verify max_seqlen is a list
        assert isinstance(batch["max_seqlen"], list), "max_seqlen should be a list"
        assert len(batch["max_seqlen"]) == batch_size, "max_seqlen should have batch_size elements"
        
        print("✓ PrePackedMLMCollator test passed")


def test_end_to_end_packing():
    """Test end-to-end: create dataset, pack it, read it back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source dataset
        source_dir = os.path.join(tmpdir, "source")
        create_test_mds_dataset(source_dir, num_samples=50, max_seq_len=128)
        
        # Pack it
        sequences = load_sequences_from_mds(source_dir, None, tokenizer=None, max_seq_len=128)
        assert len(sequences) == 50, "Should load 50 sequences"
        
        # Create packer and pack
        pack_length = 512
        packer = OfflineSequencePacker(
            pack_length=pack_length,
            pad_token_id=0,
            buffer_size=100,
            seed=42,
        )
        packer.add_sequences(sequences)
        packer.shuffle_buffer()
        packed_samples = packer.pack_all()
        
        assert len(packed_samples) > 0, "Should create packed samples"
        
        # Write packed dataset
        output_dir = os.path.join(tmpdir, "packed")
        os.makedirs(output_dir, exist_ok=True)
        
        columns = {
            "input_ids": "bytes",
            "cu_seqlens": "bytes",
            "attention_mask": "bytes",
        }
        
        with MDSWriter(columns=columns, out=output_dir, compression=None) as writer:
            for input_ids, cu_seqlens, attention_mask in packed_samples:
                cu_seqlens_array = np.array(cu_seqlens, dtype=np.int32)
                writer.write({
                    "input_ids": input_ids.tobytes(),
                    "cu_seqlens": cu_seqlens_array.tobytes(),
                    "attention_mask": attention_mask.tobytes(),
                })
        
        # Read it back
        dataset = PrePackedStreamingDataset(
            local=output_dir,
            split=None,
            shuffle=False,
        )
        
        sample = dataset[0]
        assert sample["input_ids"].dtype == np.int64
        assert sample["cu_seqlens"].dtype == np.int32
        assert sample["attention_mask"].dtype == np.int8
        
        # Verify packing efficiency - most samples should have multiple sequences
        has_multiple_seqs = sum(1 for i in range(min(len(packed_samples), 10)) 
                                if len(dataset[i]["cu_seqlens"]) > 2)  # >2 because we have start, end, and at least one more
        assert has_multiple_seqs > 0, "Some packed samples should have multiple sequences"
        
        print("✓ End-to-end packing test passed")


if __name__ == "__main__":
    print("Running pre-packed dataset tests...")
    test_offline_sequence_packer()
    test_prepacked_streaming_dataset()
    test_prepacked_mlm_collator()
    test_end_to_end_packing()
    print("\n✓ All tests passed!")
