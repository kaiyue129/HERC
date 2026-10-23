import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch import autocast


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_gib(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def _allocated_memory_gib(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return torch.cuda.memory_allocated(device) / (1024 ** 3)


def _summarize_ms(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        raise ValueError("No timing values were collected.")
    std_ms = statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0
    mean_ms = statistics.mean(values_ms)
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def _time_callable(
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int,
    repeat: int,
) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    _sync(device)

    _reset_peak_memory(device)
    allocated_before = _allocated_memory_gib(device)
    times_ms: List[float] = []

    for _ in range(repeat):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            fn()
            _sync(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    _sync(device)
    summary = _summarize_ms(times_ms)
    summary["repeat"] = repeat
    summary["warmup"] = warmup
    summary["allocated_before_gib"] = allocated_before
    summary["peak_allocated_gib"] = _peak_memory_gib(device)
    return summary


def _make_dummy_batch(trainer) -> Dict[str, Any]:
    trainer._set_batch_size_and_oversample()
    num_input_channels = determine_num_input_channels(
        trainer.plans_manager,
        trainer.configuration_manager,
        trainer.dataset_json,
    )
    patch_size = tuple(int(i) for i in trainer.configuration_manager.patch_size)
    data = torch.rand(
        (trainer.batch_size, num_input_channels, *patch_size),
        device=trainer.device,
    )
    max_label = int(max(trainer.label_manager.all_labels))

    def make_target_for_scale(scale):
        shape = tuple(int(i * j) for i, j in zip(patch_size, scale))
        atomic = torch.randint(
            low=0,
            high=max_label + 1,
            size=(trainer.batch_size, *shape),
            device=trainer.device,
        )
        if trainer.label_manager.has_regions:
            channels = []
            for region in trainer.label_manager.foreground_regions:
                values = region if isinstance(region, (tuple, list)) else (region,)
                mask = torch.zeros_like(atomic, dtype=torch.bool)
                for value in values:
                    mask |= atomic == int(value)
                channels.append(mask.float())
            return torch.stack(channels, dim=1)
        return atomic[:, None]

    if trainer.enable_deep_supervision:
        target = [make_target_for_scale(scale) for scale in trainer._get_deep_supervision_scales()]
    else:
        target = make_target_for_scale((1,) * len(patch_size))
    return {"data": data, "target": target}


def _count_parameters(network: torch.nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    total = sum(p.numel() for p in network.parameters())
    return {"total": int(total), "trainable": int(trainable)}


def _load_optional_checkpoint(trainer, checkpoint_name: Optional[str]) -> Optional[str]:
    if not checkpoint_name:
        return None
    checkpoint_path = join(trainer.output_folder, checkpoint_name)
    trainer.load_checkpoint(checkpoint_path)
    return checkpoint_path


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    trainer = get_trainer_from_args(
        args.dataset,
        args.configuration,
        args.fold,
        args.trainer,
        args.plans,
        args.use_compressed,
        device=device,
    )
    trainer.initialize()
    checkpoint_path = _load_optional_checkpoint(trainer, args.checkpoint_name)

    dummy_batch = _make_dummy_batch(trainer)
    parameter_counts = _count_parameters(trainer.network)
    batch_size = int(dummy_batch["data"].shape[0])

    trainer.network.eval()
    data = dummy_batch["data"]

    def forward_only():
        with torch.no_grad():
            with autocast(device.type, enabled=device.type == "cuda") if device.type == "cuda" else torch.no_grad():
                output = trainer.network(data)
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output.detach()

    inference = _time_callable(forward_only, device, args.warmup, args.repeat)
    inference["throughput_patches_per_s"] = batch_size / (inference["mean_ms"] / 1000.0)

    training = None
    if args.measure_training_step:
        trainer.network.train()
        trainer.optimizer.zero_grad(set_to_none=True)

        def train_step():
            return trainer.train_step(dummy_batch)

        training = _time_callable(train_step, device, args.warmup_train, args.repeat_train)
        training["throughput_patches_per_s"] = batch_size / (training["mean_ms"] / 1000.0)

    result = {
        "benchmark_scope": "synthetic patch-level trainer benchmark; run with the same dataset plans, trainer, patch size, and checkpoint for baseline/proposed comparisons",
        "dataset": str(args.dataset),
        "configuration": args.configuration,
        "fold": str(args.fold),
        "trainer": args.trainer,
        "plans": args.plans,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": batch_size,
        "num_input_channels": int(data.shape[1]),
        "patch_size": [int(i) for i in data.shape[2:]],
        "deep_supervision": bool(trainer.enable_deep_supervision),
        "amp_in_trainer": device.type == "cuda",
        "parameters": parameter_counts,
        "inference_forward_patch": inference,
        "training_step": training,
    }
    return result


def _flatten_for_csv(result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "dataset": result["dataset"],
        "configuration": result["configuration"],
        "fold": result["fold"],
        "trainer": result["trainer"],
        "plans": result["plans"],
        "checkpoint_path": result["checkpoint_path"],
        "gpu_name": result["gpu_name"],
        "torch_version": result["torch_version"],
        "cuda_version": result["cuda_version"],
        "cudnn_version": result["cudnn_version"],
        "batch_size": result["batch_size"],
        "num_input_channels": result["num_input_channels"],
        "patch_size": "x".join(str(i) for i in result["patch_size"]),
        "parameters_total": result["parameters"]["total"],
        "parameters_trainable": result["parameters"]["trainable"],
    }
    for prefix, metrics in (
        ("inference", result.get("inference_forward_patch")),
        ("training", result.get("training_step")),
    ):
        if metrics is None:
            continue
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def save_outputs(result: Dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    output_csv = output_json.with_suffix(".csv")
    row = _flatten_for_csv(result)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure patch-level wall-clock time, throughput, peak GPU memory, and parameter count for nnU-Net trainers."
    )
    parser.add_argument("dataset", type=str, help="Dataset ID or DatasetXXX name.")
    parser.add_argument("configuration", type=str, help="nnU-Net configuration, for example 3d_fullres.")
    parser.add_argument("fold", type=str, help="Fold ID.")
    parser.add_argument("-tr", "--trainer", type=str, default="nnUNetTrainer", help="Trainer class name.")
    parser.add_argument("-p", "--plans", type=str, default="nnUNetPlans", help="Plans identifier.")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Checkpoint file inside the trainer output folder.")
    parser.add_argument("--use_compressed", action="store_true", help="Use compressed preprocessed data when constructing the trainer.")
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu", "mps"))
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations for inference forward timing.")
    parser.add_argument("--repeat", type=int, default=50, help="Measured iterations for inference forward timing.")
    parser.add_argument("--measure_training_step", action="store_true", help="Also time one optimizer training step on a synthetic batch.")
    parser.add_argument("--warmup_train", type=int, default=5, help="Warm-up iterations for training-step timing.")
    parser.add_argument("--repeat_train", type=int, default=20, help="Measured iterations for training-step timing.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path. A CSV with the same stem is also written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    if args.output is None:
        output_json = Path.cwd() / f"efficiency_benchmark_{args.trainer}_{args.configuration}_fold{args.fold}.json"
    else:
        output_json = Path(args.output)
    save_outputs(result, output_json)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
