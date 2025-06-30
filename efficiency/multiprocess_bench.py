import logging
import warnings

warnings.filterwarnings("ignore")

import gc
import json
import math
import multiprocessing as mp
import os
import random
import time
from multiprocessing import Process, Queue
from typing import Annotated, List

import numpy as np
import torch
import typer
from transformers import AutoModel, AutoTokenizer

try:
    import flash_attn

    FLASH_ATTN = True
except ImportError:
    FLASH_ATTN = False

app = typer.Typer()


def create_fixed_dataset(num_samples: int = 8192, seq_len: int = 512):
    tokens = torch.randint(100, 16000, (num_samples, seq_len))  # keep tokens within standard bert models vocab range
    mask = torch.ones(num_samples, seq_len)
    return {"input_ids": tokens.long(), "attention_mask": mask.float()}


def create_variable_dataset(tokenizer, num_samples: int = 8192, seq_len: int = 512):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    lengths = torch.normal(mean=seq_len // 2, std=seq_len // 4, size=(num_samples,)).int().clamp(16, seq_len)
    tokens_list = []
    masks_list = []
    for length in lengths:
        tokens = torch.randint(100, 16000, (length.item(),))  # keep tokens within standard bert models vocab range
        mask = torch.ones(length.item())
        padded_tokens = torch.full((seq_len,), tokenizer.pad_token_id, dtype=torch.long)
        padded_mask = torch.zeros(seq_len)
        padded_tokens[:length] = tokens
        padded_mask[:length] = mask
        tokens_list.append(padded_tokens)
        masks_list.append(padded_mask)

    return {
        "input_ids": torch.stack(tokens_list),
        "attention_mask": torch.stack(masks_list),
    }


def create_all_datasets(tokenizer, num_samples: int = 8192, max_seq_len: int = 512):
    return {
        f"fixed_{max_seq_len}": create_fixed_dataset(num_samples, max_seq_len),
        f"variable_{max_seq_len}": create_variable_dataset(tokenizer, num_samples, max_seq_len),
    }


def test_batch_size_worker(q, model_name, input_ids, attention_mask, bsize, device, use_xformers, variable):
    """
    Worker that:
    1. Loads the model
    2. Tries given batch size
    3. Returns success or fail
    """
    try:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        if "gte" in model_name.lower() and use_xformers:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
            model.config.use_memory_efficient_attention = True
        else:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
        model = model.to(device)

        # roberta fails if the variable range is too small
        with torch.inference_mode():
            for i in range(4 if variable else 2):
                batch_ids = input_ids[i * bsize : (i + 1) * bsize].to(device)
                batch_mask = attention_mask[i * bsize : (i + 1) * bsize].to(device)
                model(input_ids=batch_ids, attention_mask=batch_mask)
        q.put(("success", True))
    except RuntimeError:
        q.put(("success", False))
    except Exception as e:
        print(f"Error in test_batch_size_worker: {e}", flush=True)
        q.put(("error", str(e)))


def find_max_batch_size_worker(
    q,
    model_name,
    input_ids,
    attention_mask,
    device,
    use_xformers,
    start_batch_size: int,
    max_batch_size_limit: int,
    variable: bool,
):
    """
    Worker that runs the batch size finding logic.
    Each attempt is run in its own worker to ensure full memory isolation.
    """

    class EarlyStop(Exception):
        pass

    fixed_bs = start_batch_size
    tried_success = set()
    tried_fail = set()

    def try_batch_size(bsize):
        if bsize in tried_success:
            print(f"Batch size {bsize} already succeeded, skipping")
            return True
        if bsize in tried_fail:
            print(f"Batch size {bsize} already failed, skipping")
            return False
        print(f"Attempting batch size: {bsize}")
        # Spawn a worker for each attempt
        attempt_q = Queue()
        p = Process(
            target=test_batch_size_worker,
            args=(
                attempt_q,
                model_name,
                input_ids,
                attention_mask,
                bsize,
                device,
                use_xformers,
                variable,
            ),
        )
        p.start()
        p.join()
        result = attempt_q.get()
        p = None
        if result[0] == "error":
            # If there's an error unrelated to OOM, raise it
            print(f"Error occurred: {result[1]}")
            raise RuntimeError(result[1])
        success = result[1]
        print(f"Batch size {bsize}: {'succeeded' if success else 'failed'}")
        if success:
            tried_success.add(bsize)
            if variable and bsize > fixed_bs:
                raise EarlyStop()
        else:
            tried_fail.add(bsize)
        print("Clearing CUDA cache and garbage collection")
        torch.cuda.empty_cache()
        gc.collect()
        return success

    try:
        typer.secho("\nStarting batch size search...", fg=typer.colors.BLUE)
        batch_size = start_batch_size
        typer.secho(
            "\nPhase 1: Increasing batch size by 128 until OOM",
            fg=typer.colors.BLUE,
        )
        while try_batch_size(batch_size) and batch_size < max_batch_size_limit:
            batch_size += 128
            print(f"Increasing to {batch_size}")

        typer.secho("\nPhase 2: Backing off by 64 until stable", fg=typer.colors.BLUE)
        while not try_batch_size(batch_size) and batch_size > 64:
            batch_size -= 64
            print(f"Decreasing to {batch_size}")

        # If still not working, try smaller decrements
        if not try_batch_size(batch_size):
            typer.secho(
                "\nPhase 3: Fine-tuning with smaller decrements",
                fg=typer.colors.BLUE,
            )
            while not try_batch_size(batch_size) and batch_size > 4:
                batch_size -= 4
                print(f"Fine-tuning decrease to {batch_size}")
            if batch_size <= 4 and not try_batch_size(batch_size):
                print("Attempting minimum batch size of 1")
                batch_size = 1
                if not try_batch_size(batch_size):
                    raise RuntimeError("Cannot find a working batch size.")
        else:
            typer.secho("\nSkipping Phase 3 for this model", fg=typer.colors.BLUE)

        typer.secho("\nPhase 4: Final optimization", fg=typer.colors.BLUE)

        # Try increments of 32
        test_size = batch_size + 32
        while test_size < max_batch_size_limit:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 32
            print(f"Testing increment to {test_size}")

        # Try increments of 16
        test_size = batch_size + 16
        while test_size < max_batch_size_limit:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 16
            print(f"Testing increment to {test_size}")

        # Try increments of 8
        test_size = batch_size + 8
        while test_size < max_batch_size_limit:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 8
            print(f"Testing increment to {test_size}")

        # Try increments of 4
        test_size = batch_size + 4
        while test_size < max_batch_size_limit:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 4
            print(f"Testing increment to {test_size}")

        # Try increments of 2
        test_size = batch_size + 2
        while test_size < max_batch_size_limit:
            success = try_batch_size(test_size)
            if not success:
                test_size = batch_size
                break
            batch_size = test_size
            test_size += 2
            print(f"Testing increment to {test_size}")

        final_batch_size = min(batch_size, max_batch_size_limit)
        sorted_successes = sorted(tried_success)
        if len(sorted_successes) >= 2:
            second_best = sorted_successes[-2]
            typer.secho(f"\nUsing second highest working batch size {second_best} to prevent OOM errors", fg=typer.colors.BLUE)  # fmt: skip
            final_batch_size = second_best
        else:
            typer.secho(f"\nLess than 2 successful batch sizes, using {final_batch_size}", fg=typer.colors.YELLOW)  # fmt: skip
        typer.secho(f"\nFinal batch size determined: {final_batch_size}", fg=typer.colors.BLUE)  # fmt: skip
        q.put(("success", final_batch_size))
    except EarlyStop:
        typer.secho(f"\nVariable dataset capacity exceeded fixed batch size {fixed_bs}, stopping search early",fg=typer.colors.BLUE)  # fmt: skip
        q.put(("success", fixed_bs))
        return
    except Exception as e:
        typer.secho(f"Error in batch size search: {str(e)}", fg=typer.colors.RED)
        q.put(("error", str(e)))


def inference_worker(
    q,
    model_name,
    dataset_name,
    input_ids,
    attention_mask,
    num_batches,
    max_batch_size,
    n_iters,
    device,
    use_xformers,
):
    """
    Worker to run inference multiple times and report mean/std of times.
    Model loading is done here to isolate memory usage.
    """
    try:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        if "gte" in model_name.lower() and use_xformers:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
            model.config.use_memory_efficient_attention = True
        else:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
        model = model.to(device)
        model.eval()

        print(f"\nRunning {model_name} on {dataset_name} (batch_size={max_batch_size}) for {n_iters} iters…", flush=True)  # fmt: skip

        total_tokens = 0
        for i in range(0, int(num_batches * max_batch_size), max_batch_size):
            total_tokens += attention_mask[i : i + max_batch_size].sum().item()

        times = []
        tokens_per_second = []
        for j in range(n_iters + 1):
            with torch.inference_mode():
                iter_batches = min(num_batches, 2) if j == 0 else num_batches
                start_time = time.time()
                for i in range(0, int(iter_batches * max_batch_size), max_batch_size):
                    batch_ids = input_ids[i : i + max_batch_size].clone().to(device)
                    batch_mask = attention_mask[i : i + max_batch_size].clone().to(device).bool()
                    model(input_ids=batch_ids, attention_mask=batch_mask)
                torch.cuda.synchronize()
                end_time = time.time()
            if j > 0:
                times.append(end_time - start_time)
                tokens_per_second.append(total_tokens / (end_time - start_time))
                print(f"Time taken: {times[-1]:.2f} seconds. Tokens per second: {tokens_per_second[-1]:.0f}", flush=True)  # fmt: skip

        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_tokens_per_second = np.mean(tokens_per_second)
        std_tokens_per_second = np.std(tokens_per_second)
        if q is not None:
            q.put(
                (
                    dataset_name,
                    mean_time,
                    std_time,
                    mean_tokens_per_second,
                    std_tokens_per_second,
                )
            )
        else:
            return (
                dataset_name,
                mean_time,
                std_time,
                mean_tokens_per_second,
                std_tokens_per_second,
            )
    except Exception as e:
        if q is not None:
            q.put(("error", str(e)))
        else:
            raise e


def run_inference_benchmark(
    model_name: str,
    use_xformers: bool = False,
    n_iters: int = 10,
    max_seq_len: int = 512,
    start_batch_size: int = 1024,
    max_batch_size_limit: int = 4096,
    device: str = "cuda",
    target_tokens: int = 5242880,
    min_inference_batches: int = 8,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
    datasets = create_all_datasets(tokenizer, int(4 * target_tokens / max_seq_len), max_seq_len)

    processing_times = {}
    batch_sizes = []

    # Ensure a clean GPU state before starting
    torch.cuda.empty_cache()
    gc.collect()

    for dataset_name, dataset in datasets.items():
        input_ids = dataset["input_ids"]
        attention_mask = dataset["attention_mask"].int()

        # Run batch size finding in its own worker
        q = Queue()
        p = Process(
            target=find_max_batch_size_worker,
            args=(
                q,
                model_name,
                input_ids,
                attention_mask,
                device,
                use_xformers,
                start_batch_size if len(batch_sizes) == 0 else batch_sizes[-1],
                max_batch_size_limit,
                dataset_name.startswith("variable_"),
            ),
        )
        p.start()
        p.join()
        result = q.get()
        p = None
        if result[0] == "error":
            print(f"Error finding batch size for {dataset_name}: {result[1]}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        max_batch_size = result[1]
        batch_sizes.append(max_batch_size)

        torch.cuda.empty_cache()
        gc.collect()

        # Determine number of batches to target a specific number of tokens
        batch_size = min(batch_sizes)
        # use the minimum batch size between fixed and variable for inference
        target_tokens = target_tokens
        tokens_per_batch = batch_size * max_seq_len
        num_batches_needed = math.ceil(target_tokens / tokens_per_batch)
        if num_batches_needed < min_inference_batches:
            num_batches_needed = min_inference_batches
        print(f"Using {num_batches_needed} batches for {dataset_name} to target tokens", flush=True)  # fmt: skip

        # Run inference in its own worker
        q = Queue()
        p = Process(
            target=inference_worker,
            args=(
                q,
                model_name,
                dataset_name,
                input_ids,
                attention_mask,
                num_batches_needed,
                batch_size,
                n_iters,
                device,
                use_xformers,
            ),
        )
        p.start()
        p.join()
        result = q.get()
        p = None
        if result[0] == "error":
            print(f"Error during inference for {dataset_name}: {result[1]}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        (
            dataset_name_ret,
            mean_time,
            std_time,
            mean_tokens_per_second,
            std_tokens_per_second,
        ) = result
        processing_times[dataset_name_ret] = {
            "time_mean": mean_time,
            "time_std": std_time,
            "max_batch_size": max_batch_size,
            "tokens_per_second_mean": mean_tokens_per_second,
            "tokens_per_second_std": std_tokens_per_second,
        }
        if dataset_name.startswith("fixed_"):
            processing_times[dataset_name_ret]["num_tokens_per_batch"] = max_batch_size * max_seq_len
        typer.secho(f"\n{dataset_name_ret}: {mean_tokens_per_second:.0f} ± {std_tokens_per_second:.0f} tokens/sec, {mean_time:.2f} ± {std_time:.2f} sec (batch_size: {max_batch_size})", fg=typer.colors.BLUE)  # fmt: skip

        torch.cuda.empty_cache()
        gc.collect()

    return processing_times


@app.command()
def main(
    model: Annotated[str, typer.Option(help="Model name to benchmark")] = "answerdotai/ModernBERT-base",
    xformers: Annotated[bool, typer.Option(help="Use XFormers for GTE models, all other models will use defaults")] = False,
    n_iters: Annotated[int, typer.Option(help="Number of iterations to average timings over")] = 5,
    seq_lens: Annotated[List[int], typer.Option(help="List of maximum sequence lengths to benchmark")] = [512, 8192],
    device: Annotated[str, typer.Option(help="GPU device to use")] = "cuda",
    start_batch_size: Annotated[int, typer.Option(help="Batch size search starting batch size")] = 128,
    max_batch_size: Annotated[int, typer.Option(help="Maximum allowed batch size")] = 4096,
    target_tokens: Annotated[int, typer.Option(help="Target total tokens for dataset inference")] = 5_242_880,
    min_inference_batches: Annotated[int, typer.Option(help="Minimum number of inference batches per dataset")] = 8,
):  # fmt: skip
    if not FLASH_ATTN and "modernbert" in model.lower():
        typer.secho(
            "WARNING: Flash Attention 2.0 is not installed. ModernBERT requires Flash Attention for accurate benchmarking.",
            fg=typer.colors.RED,
        )
        typer.Exit(1)

    mp.set_start_method("spawn", force=True)
    results = {}
    for seq_len in seq_lens:
        typer.secho(
            f"\n=== Running {model} {seq_len} sequence length benchmark ===",
            fg=typer.colors.GREEN,
        )
        res = run_inference_benchmark(
            model_name=model,
            use_xformers=xformers,
            n_iters=n_iters,
            max_seq_len=seq_len,
            start_batch_size=start_batch_size,
            max_batch_size_limit=max_batch_size,
            device=device,
            target_tokens=target_tokens,
            min_inference_batches=min_inference_batches,
        )
        results[seq_len] = res

    typer.secho(f"\n{model} Processing Time Summary:", fg=typer.colors.GREEN)
    typer.secho("-" * 50, fg=typer.colors.BLUE)
    for dataset_name, result in results.items():
        for name in ["fixed", "variable"]:
            metrics = result.get(f"{name}_{dataset_name}", None)
            if metrics is None:
                continue
            typer.echo(f"{name}_{dataset_name}: {metrics['tokens_per_second_mean']:.0f} ± {metrics['tokens_per_second_std']:.0f} tokens/sec, {metrics['time_mean']:.2f} ± {metrics['time_std']:.2f} sec (batch_size: {metrics['max_batch_size']})")  # fmt: skip

    try:
        if xformers:
            os.makedirs(f"results/{model.replace('/', '_')}_xformers", exist_ok=True)
            with open(f"results/{model.replace('/', '_')}_xformers_inference_times.json", "w") as f:
                json.dump(results, f, indent=2)
        else:
            os.makedirs(f"results/{model.replace('/', '_')}", exist_ok=True)
            with open(f"results/{model.replace('/', '_')}_inference_times.json", "w") as f:
                json.dump(results, f, indent=2)
    except Exception as e:
        typer.secho(f"Error saving results: {e}", fg=typer.colors.RED)
        typer.Exit(1)

    typer.Exit(0)


if __name__ == "__main__":
    app()
