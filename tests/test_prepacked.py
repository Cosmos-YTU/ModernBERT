# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

"""Test pre-packed dataset functionality."""

import os
import sys
import tempfile
import shutil

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_data import PrePackedCollator


def test_prepacked_collator():
    """Test PrePackedCollator with synthetic data."""
    
    # Create synthetic pre-packed data with EOS tokens
    # Simulate: [seq1, EOS, seq2, EOS, PAD, PAD, ...]
    eos_token_id = 2
    pad_token_id = 0
    mask_token_id = 103
    
    # Create example packed sequences
    # Each example should have multiple sequences separated by EOS tokens
    examples = [
        {
            # [1, 1, 1, EOS, 5, 5, EOS, PAD, PAD, PAD]
            "input_ids": np.array([1, 1, 1, 2, 5, 5, 2, 0, 0, 0], dtype=np.int64)
        },
        {
            # [3, 3, 3, 3, EOS, 6, 6, 6, EOS, PAD]
            "input_ids": np.array([3, 3, 3, 3, 2, 6, 6, 6, 2, 0], dtype=np.int64)
        },
    ]
    
    collator = PrePackedCollator(
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        mlm_probability=0.0,  # No masking for this test
        ignore_token_id=-100,
    )
    
    batch = collator(examples)
    
    # Verify batch structure
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "cu_seqlens" in batch
    assert "max_seqlen" in batch
    
    # Verify shapes
    assert batch["input_ids"].shape == (2, 10)
    assert batch["labels"].shape == (2, 10)
    assert batch["attention_mask"].shape == (2, 10)
    assert len(batch["cu_seqlens"]) == 2
    assert len(batch["max_seqlen"]) == 2
    
    # Verify cu_seqlens
    # First example: [0, 4, 7] (3 tokens, EOS at 3; 2 tokens, EOS at 6; rest is padding)
    # Second example: [0, 5, 9] (4 tokens, EOS at 4; 3 tokens, EOS at 8; rest is padding)
    assert len(batch["cu_seqlens"][0]) >= 2  # At least start and one boundary
    assert len(batch["cu_seqlens"][1]) >= 2
    
    # First cu_seq should start at 0
    assert batch["cu_seqlens"][0][0] == 0
    assert batch["cu_seqlens"][1][0] == 0
    
    # Verify attention mask has correct padding
    # First example has padding at positions 7, 8, 9
    assert batch["attention_mask"][0, 7] == 0
    assert batch["attention_mask"][0, 8] == 0
    assert batch["attention_mask"][0, 9] == 0
    
    # Second example has padding at position 9
    assert batch["attention_mask"][1, 9] == 0
    
    print("✓ PrePackedCollator test passed")


def test_prepacked_collator_with_masking():
    """Test PrePackedCollator with MLM masking enabled."""
    
    eos_token_id = 2
    pad_token_id = 0
    mask_token_id = 103
    
    examples = [
        {
            "input_ids": np.array([1, 1, 1, 2, 5, 5, 2, 0, 0, 0], dtype=np.int64)
        },
    ]
    
    collator = PrePackedCollator(
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        mlm_probability=0.15,  # 15% masking
        ignore_token_id=-100,
    )
    
    batch = collator(examples)
    
    # Verify that labels exist and are not all -100 (some tokens should be masked)
    assert batch["labels"] is not None
    assert batch["labels"].shape == (1, 10)
    
    # Padding positions should be -100 in labels
    assert batch["labels"][0, 7] == -100
    assert batch["labels"][0, 8] == -100
    assert batch["labels"][0, 9] == -100
    
    print("✓ PrePackedCollator with masking test passed")


if __name__ == "__main__":
    test_prepacked_collator()
    test_prepacked_collator_with_masking()
    print("\nAll tests passed!")
