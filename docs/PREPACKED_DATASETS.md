# Pre-Packed Streaming Datasets

This feature enables using pre-tokenized and pre-packed streaming datasets with ModernBERT, allowing:

- **Streaming with packing**: Use `streaming: true` with pre-packed data
- **No runtime overhead**: Packing happens offline, not during training
- **Multiple streaming sources**: Use the `sources` config pattern for dataset weighting
- **Same training efficiency**: Matches runtime packing performance

## Overview

Previously, sequence packing only worked with `streaming: false` via `GreedyBestFitSequencePacker` at runtime. This had two limitations:

1. Cannot use streaming datasets with packing
2. Packing overhead during training

With pre-packed datasets, you pack data offline once, then stream it efficiently during training.

## Quick Start

### 1. Tokenize Your Dataset

First, tokenize your dataset to MDS format (if not already done):

```bash
python src/convert_dataset.py \
    --dataset your-dataset \
    --out_root /path/to/tokenized \
    --tokenizer your-tokenizer \
    --splits train validation
```

### 2. Pack the Dataset

Pack the tokenized dataset offline:

```bash
python src/pack_dataset.py \
    --input_path /path/to/tokenized \
    --output_path /path/to/packed \
    --tokenizer your-tokenizer \
    --max_seq_len 8192 \
    --split train
```

The script packs sequences greedily to `max_seq_len`, separating documents with EOS tokens.

### 3. Configure Training

Use the pre-packed dataset in your training config:

```yaml
train_loader:
  name: text
  dataset:
    local: /path/to/packed
    split: train
    tokenizer_name: ${tokenizer_name}
    max_seq_len: ${max_seq_len}
    shuffle: true
    mlm_probability: ${mlm_probability}
    streaming: true
    pre_packed: true  # Enable pre-packed mode
    eos_token_id: 2   # Required for cu_seqlens computation
  drop_last: true
  num_workers: 24
  sequence_packing: false  # Data is already packed
```

### 4. Train

Run training as usual:

```bash
python main.py yamls/modernbert/your-config.yaml
```

## Multiple Sources / Dataset Weighting

You can combine multiple pre-packed datasets with different weights using the `sources` pattern:

```yaml
train_loader:
  name: text
  dataset:
    sources:
      - local: /path/to/dataset1-packed
        split: train
      - local: /path/to/dataset1-packed
        split: train
      - local: /path/to/dataset1-packed
        split: train
      # ^ Repeat to weight dataset1 3x
      - local: /path/to/dataset2-packed
        split: train
      # ^ dataset2 appears 1x
    tokenizer_name: ${tokenizer_name}
    max_seq_len: ${max_seq_len}
    shuffle: true
    mlm_probability: ${mlm_probability}
    streaming: true
    pre_packed: true
    eos_token_id: 2
  drop_last: true
  num_workers: 24
  sequence_packing: false
```

This gives a 3:1 ratio between dataset1 and dataset2.

## Data Format

Pre-packed datasets must:

- Store only `input_ids` as bytes (matching standard MDS format)
- Separate documents with EOS tokens within each packed sample
- Be a fixed length of `max_seq_len` per sample
- NOT store `cu_seqlens` or `labels` (derived at runtime)

Example packed sample structure:
```
[doc1_token1, doc1_token2, ..., doc1_tokenN, EOS,
 doc2_token1, doc2_token2, ..., doc2_tokenM, EOS,
 ...,
 PAD, PAD, PAD]
```

## Training Flow

When loading pre-packed data:

1. **cu_seqlens computation**: Derived from EOS token positions at batch collation
2. **MLM masking**: Applied on-the-fly (stochastic per epoch)
3. **Flash Attention**: Uses `cu_seqlens` and `max_seqlen` from the batch

This matches the behavior of `GreedyBestFitSequencePacker` but without runtime overhead.

## Key Configuration Options

### Required Options for Pre-Packed Mode

- `streaming: true` - Use streaming dataset
- `pre_packed: true` - Enable pre-packed collator
- `eos_token_id: <id>` - EOS token ID (required for cu_seqlens computation)
- `sequence_packing: false` - No runtime packing needed

### Optional Options

- `sources: [...]` - Multiple dataset sources with weighting
- `mlm_probability: 0.3` - MLM masking probability (default for ModernBERT)
- `shuffle: true` - Shuffle samples during streaming

## Packing Algorithm

The offline packing uses the same greedy best-fit algorithm as `GreedyBestFitSequencePacker`:

1. Maintain a pool of in-progress packed sequences
2. For each input sequence:
   - Add EOS token to the end
   - Find the smallest packed sequence that can fit it
   - If no space, create a new packed sequence
3. Pack until all sequences are consumed

This ensures consistent packing behavior between offline and runtime modes.

## Benefits

### Performance

- **No runtime packing overhead**: Packing happens once offline
- **Faster data loading**: Pre-packed data loads directly
- **Same efficiency**: Identical packing as runtime mode

### Flexibility

- **Streaming support**: Use with very large datasets
- **Dataset weighting**: Easily combine multiple sources
- **Reproducible packing**: Pack once, train many times

### Storage

- **Efficient storage**: Only `input_ids` stored (as bytes)
- **On-the-fly masking**: Different masking each epoch
- **No redundancy**: `cu_seqlens` and `labels` computed at runtime

## Example: Turkish ModernBERT

Pack Turkish datasets for ModernBERT:

```bash
# Pack BERTurk corpus
python src/pack_dataset.py \
    --input_path /data/berturk-corpus-tokenized \
    --output_path /data/berturk-corpus-packed \
    --tokenizer dbmdz/bert-base-turkish-cased \
    --max_seq_len 8192 \
    --split train

# Pack FineWeb-2-Turkish
python src/pack_dataset.py \
    --input_path /data/fineweb-2-turkish-tokenized \
    --output_path /data/fineweb-2-turkish-packed \
    --tokenizer dbmdz/bert-base-turkish-cased \
    --max_seq_len 8192 \
    --split train
```

Then use in training config:

```yaml
train_loader:
  name: text
  dataset:
    sources:
      - local: /data/berturk-corpus-packed
        split: train
      - local: /data/berturk-corpus-packed
        split: train
      - local: /data/fineweb-2-turkish-packed
        split: train
    tokenizer_name: dbmdz/bert-base-turkish-cased
    max_seq_len: 8192
    streaming: true
    pre_packed: true
    eos_token_id: 2
    mlm_probability: 0.3
  sequence_packing: false
```

## Troubleshooting

### Error: "eos_token_id must be provided"

Set `eos_token_id` in your config. Check your tokenizer's EOS token:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your-tokenizer")
print(tokenizer.eos_token_id)
```

### Verify packed data

Load and inspect packed samples:

```python
import json
import numpy as np
from streaming.base.format import reader_from_json

index_path = "/path/to/packed/train/index.json"
obj = json.load(open(index_path))
shard = reader_from_json("/path/to/packed", "train", obj["shards"][0])
sample = shard[0]
input_ids = np.frombuffer(sample["input_ids"], dtype=np.int64)
print(f"Packed sample length: {len(input_ids)}")
print(f"Should equal max_seq_len: {len(input_ids) == max_seq_len}")
print(f"EOS positions: {np.where(input_ids == 2)[0]}")
```

## See Also

- [Example config](../../yamls/modernbert/modernbert-prepacked-example.yaml)
- [Unit tests](../../tests/test_prepacked.py)
- [Integration tests](../../tests/test_prepacked_integration.py)
