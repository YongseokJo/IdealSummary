import os
import psutil
import torch
import datetime


def get_process_memory_mb() -> float:
    """Return Resident Set Size (RSS) of current process in MiB."""
    try:
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        return float(rss) / (1024.0 * 1024.0)
    except Exception:
        return float('nan')


def get_gpu_memory_mb() -> dict:
    """Return dict with GPU memory metrics in MiB (allocated, reserved, max_allocated).

    If CUDA is unavailable, returns zeros.
    """
    if not torch.cuda.is_available():
        return {
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_max_allocated_mb": 0.0,
        }
    try:
        allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        reserved = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        max_alloc = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return {
            "cuda_allocated_mb": float(allocated),
            "cuda_reserved_mb": float(reserved),
            "cuda_max_allocated_mb": float(max_alloc),
        }
    except Exception:
        return {
            "cuda_allocated_mb": float('nan'),
            "cuda_reserved_mb": float('nan'),
            "cuda_max_allocated_mb": float('nan'),
        }


def log_memory(prefix: str, run=None, step: int = None, log_file: str = None, trial: int = None):
    """Print and optionally log memory stats under W&B `run` and append to `log_file`.

    `trial` (int) is optional extra context that will be included in the
    printed/logged line.
    """
    cpu_mb = get_process_memory_mb()
    gpu = get_gpu_memory_mb()
    ts = datetime.datetime.now().isoformat()
    trial_str = f" trial={trial}" if trial is not None else ""
    msg = (
        f"{ts} [mem] {prefix}:{trial_str} cpu_rss_mb={cpu_mb:.1f} "
        f"cuda_alloc_mb={gpu['cuda_allocated_mb']:.1f} "
        f"cuda_res_mb={gpu['cuda_reserved_mb']:.1f} "
        f"cuda_max_alloc_mb={gpu['cuda_max_allocated_mb']:.1f}"
    )
    print(msg)
    if run is not None:
        payload = {
            f"mem/{prefix}/cpu_rss_mb": cpu_mb,
            f"mem/{prefix}/cuda_alloc_mb": gpu["cuda_allocated_mb"],
            f"mem/{prefix}/cuda_reserved_mb": gpu["cuda_reserved_mb"],
            f"mem/{prefix}/cuda_max_alloc_mb": gpu["cuda_max_allocated_mb"],
        }
        try:
            run.log(payload, step=step)
        except Exception:
            pass

    if log_file:
        try:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

