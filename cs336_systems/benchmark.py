from __future__ import annotations

import argparse
import statistics
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# Hyperparameters from the assignment handout (override with explicit flags).
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
    description="End-to-end benchmarking for the basics Transformer LM."
)
    parser.add_argument("--model-size", choices=list(MODEL_CONFIGS), default="small")
    parser.add_argument("--d-model", type=int, help="if provided, overrides d_model from --model-size")
    parser.add_argument("--d-ff", type=int, help="if provided, overrides d_ff from --model-size")
    parser.add_argument("--num-layers", type=int, help="if provided, overrides num_layers from --model-size")
    parser.add_argument("--num-heads", type=int, help="if provided, overrides num_heads from --model-size")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "train"],
        default="train",
        help="forward only / forward+backward / forward+backward+optimizer_step",
    )
    parser.add_argument("--warmup-steps", type=int, default=5, help="untimed steps before measuring")
    parser.add_argument("--num-steps", type=int, default=10, help="timed steps")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 如果没有覆盖就根据规定的表格规格传入参数
    config = MODEL_CONFIGS[args.model_size]
    for name in ("d_model", "d_ff", "num_layers", "num_heads"):
        if getattr(args, name) is None:
            setattr(args, name, config[name])
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    def step() -> None:
        with torch.set_grad_enabled(args.mode != "forward"):
            logits = model(input_ids)
            if args.mode == "forward":
                return
            loss = cross_entropy(logits, targets)
        loss.backward()
        if args.mode == "train":
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def synchronize() -> None:
        # CUDA kernels are asynchronous: without this we would time only the
        # launch of the work, not its execution on the GPU.
        if use_cuda:
            torch.cuda.synchronize()

    for _ in range(args.warmup_steps):
        step()
    synchronize()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.num_steps):
        start = timeit.default_timer()
        step()
        synchronize()
        times.append(timeit.default_timer() - start)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    tokens_per_step = args.batch_size * args.context_length

    print(f"model={args.model_size} d_model={args.d_model} d_ff={args.d_ff} "
          f"layers={args.num_layers} heads={args.num_heads} params={model.get_num_params():,}")
    print(f"device={device} mode={args.mode} batch={args.batch_size} ctx={args.context_length}")
    print(f"time per step: {mean * 1e3:.2f} ms (std {std * 1e3:.2f} ms) over {args.num_steps} steps")
    print(f"throughput: {tokens_per_step / mean:,.0f} tokens/s")
    if use_cuda:
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak memory allocated: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
