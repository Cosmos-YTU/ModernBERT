# Offline Sequence Packing for Streaming Datasets

This guide explains how to use offline sequence packing to pre-process your datasets for efficient training with streaming.

## Overview

Offline sequence packing allows you to:
1. **Pre-pack sequences** offline into MDS format
2. **Stream the packed dataset** during training (no memory limits!)
3. **Apply MLM masking at runtime** so it varies across epochs

This combines the efficiency of sequence packing with the scalability of streaming datasets.

## Quick Start

### Step 1: Pack Your Dataset Offline

Use `pack_dataset.py` to pack sequences from one or more MDS sources:

```bash
python src/pack_dataset.py \
    --sources \
        /path/to/dataset1:train \
        /path/to/dataset1:train \
        /path/to/dataset2:train \
    --output /path/to/packed-output \
    --tokenizer dbmdz/bert-base-turkish-cased \
    --max_seq_len 1024 \
    --seed 42
```

**Arguments:**
- `--sources`: List of dataset paths in format `/path/to/dataset:split`. Repeat sources to oversample them.
- `--output`: Directory where packed MDS dataset will be written
- `--tokenizer`: HuggingFace tokenizer name/path (required for text data, optional for pre-tokenized)
- `--max_seq_len`: Maximum sequence length (sequences are truncated to this)
- `--seed`: Random seed for shuffling (default: 42)
- `--pack_length`: Length of each packed sequence (default: `max_seq_len * 32`)
- `--pad_token_id`: Padding token ID (default: 0)
- `--buffer_size`: Buffer size for packing (default: 10000)

### Step 2: Configure Training

Update your training config to use the pre-packed dataset:

```yaml
train_loader:
  name: text
  dataset:
    local: /path/to/packed-output
    split: train  # optional
    max_seq_len: 1024
    shuffle: true
    mlm_probability: 0.3
    streaming: true
    prepacked: true  # NEW: indicates this is pre-packed data
  drop_last: true
  num_workers: 24
```

**Important config options:**
- `prepacked: true` - **Required** to enable pre-packed dataset mode
- `streaming: true` - **Required** for pre-packed datasets
- `mlm_probability` - **Required** - MLM masking is applied at runtime

### Step 3: Train as Usual

Run your training script normally. The dataloader will:
- Stream pre-packed sequences from disk
- Apply MLM masking at runtime (varies per epoch)
- Return batches in the format expected by the model

## How It Works

### Offline Packing (`pack_dataset.py`)

The packing script:
1. Loads sequences from multiple MDS sources
2. Tokenizes text data if needed (or uses pre-tokenized `input_ids`)
3. Shuffles all sequences together
4. Packs sequences using greedy best-fit algorithm
5. Writes to MDS format with:
   - `input_ids`: packed token IDs (int64 as bytes)
   - `cu_seqlens`: cumulative sequence lengths (int32 as bytes)
   - `attention_mask`: attention mask for non-padding tokens (int8 as bytes)

**Note:** MLM masking is NOT applied during packing - this happens at runtime.

### Runtime Reading (`PrePackedStreamingDataset`)

During training, `PrePackedStreamingDataset`:
- Streams pre-packed samples from MDS format
- Deserializes byte arrays to numpy arrays
- Returns samples with `input_ids`, `cu_seqlens`, and `attention_mask`

### Runtime Masking (`PrePackedMLMCollator`)

The collator:
- Takes batches of pre-packed samples
- Applies MLM masking using `SequencePacker.mlm_masking` (same as online packing)
- Returns the format expected by the model: `{input_ids, labels, cu_seqlens, max_seqlen, attention_mask}`

## Examples

### Example 1: Pack a single dataset

```bash
python src/pack_dataset.py \
    --sources /data/corpus:train \
    --output /data/packed-corpus \
    --tokenizer bert-base-uncased \
    --max_seq_len 512
```

### Example 2: Pack multiple datasets with oversampling

To oversample `dataset1` 3x relative to `dataset2`:

```bash
python src/pack_dataset.py \
    --sources \
        /data/dataset1:train \
        /data/dataset1:train \
        /data/dataset1:train \
        /data/dataset2:train \
    --output /data/packed-mixed \
    --tokenizer bert-base-uncased \
    --max_seq_len 1024
```

### Example 3: Pack pre-tokenized data

If your data is already tokenized (has `input_ids` column), omit the tokenizer:

```bash
python src/pack_dataset.py \
    --sources /data/pretokenized:train \
    --output /data/packed \
    --max_seq_len 512
```

### Example 4: Custom pack length

For longer packed sequences (more efficient packing but larger memory per batch):

```bash
python src/pack_dataset.py \
    --sources /data/corpus:train \
    --output /data/packed \
    --tokenizer bert-base-uncased \
    --max_seq_len 1024 \
    --pack_length 65536  # Pack into 64K token sequences
```

## Comparison: Streaming vs Packing vs Pre-Packed

| Feature | Regular Streaming | Online Packing | Offline Pre-Packed |
|---------|-------------------|----------------|-------------------|
| **Memory usage** | Low | High (loads full dataset) | Low (streams) |
| **Packing efficiency** | None | High | High |
| **Setup time** | None | None | Requires offline packing |
| **MLM variation** | Per epoch | Per epoch | Per epoch |
| **Dataset size limit** | Unlimited | Limited by RAM | Unlimited |

## Troubleshooting

### Error: "Pre-packed datasets require mlm_probability to be set"

**Solution:** Add `mlm_probability` to your dataset config:
```yaml
dataset:
  mlm_probability: 0.3
```

### Error: "Pre-packed datasets (prepacked=true) must use streaming=true"

**Solution:** Ensure `streaming: true` in your config.

### Packing efficiency is low

**Causes:**
- Sequences are very uniform in length
- `max_seq_len` is too small
- Buffer size is too small

**Solutions:**
- Increase `--buffer_size` for better packing
- Use a larger `--pack_length` if memory allows
- Mix datasets with varying sequence lengths

## Technical Details

### Data Format

Pre-packed MDS datasets have three columns:

```python
{
    "input_ids": bytes,      # int64 array serialized as bytes
    "cu_seqlens": bytes,     # int32 array serialized as bytes  
    "attention_mask": bytes, # int8 array serialized as bytes
}
```

### cu_seqlens Format

`cu_seqlens` is a cumulative sequence length array:
- Starts with 0
- Each element is the cumulative position after each packed sequence
- Final element is the pack_length

Example: `[0, 128, 384, 512]` means:
- Sequence 1: positions 0-127 (length 128)
- Sequence 2: positions 128-383 (length 256)  
- Sequence 3: positions 384-511 (length 128)
- Total pack length: 512

### Greedy Best-Fit Algorithm

The packing algorithm:
1. For each sequence, find the packed sample with the smallest remaining space that fits it
2. If no space fits, defer the sequence to the next batch
3. Continue until all sequences are packed or remaining sequences are too large

This maximizes packing efficiency while maintaining good throughput.
