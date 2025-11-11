# Batch Sizes & Tokens - Dev Notes

## Quick Reference

### Batch Size Parameters (SEQUENCES)
- `global_train_batch_size`: Total sequences across all devices per step
- `device_train_batch_size`: Sequences per device (= global / world_size)
- `device_train_microbatch_size`: Sequences per forward/backward pass

### Key Formulas
```yaml
global_train_batch_size: 4608     # Total sequences
device_train_microbatch_size: 96   # Per GPU per pass
# With 48 GPUs: device_train_batch_size = 4608 / 48 = 96
```

### Token Counting
```python
# Composer needs this to track tokens
data_loader = DataSpec(
    data_loader,
    get_num_tokens_in_batch=num_tokens_in_batch_fn,
)

def get_num_tokens_in_batch_unpadded(batch: dict):
    return batch["attention_mask"].sum().item()  # Only real tokens
```

## Core Concepts

### Sequence Packing
- `sequence_packing: true` packs multiple docs into each n-token sequence
- Token count = batch_size × max_seq_len only if `sequence_packing: true`
- Use `count_padding_tokens: false` for accurate tracking

### Duration Units
```yaml
max_duration: 1_719_000_000_000tok  # 1.7T actual tokens
eval_interval: 4000ba               # Every 4000 batches
save_interval: 4000ba
```

### Gradient Accumulation
When microbatch < device batch:
```python
grad_accum_steps = device_train_batch_size // device_train_microbatch_size
```
## Config Template
```yaml
# Batch sizes (sequences, not tokens)
global_train_batch_size: 4608
device_train_microbatch_size: 96
global_eval_batch_size: 1024

# Token tracking
count_padding_tokens: false
sequence_packing: true

# Duration
max_duration: 1_719_000_000_000tok
eval_interval: 4000ba
save_interval: 4000ba
```