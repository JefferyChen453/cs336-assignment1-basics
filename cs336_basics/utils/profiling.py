import time
import tracemalloc
import threading
from functools import wraps

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def profile_time_memory_gpu(interval=0.05):
    """
    Profile a function's runtime, Python peak memory, RSS (system) peak memory,
    and GPU memory usage if CUDA is available.

    interval: sampling period for RSS memory (in seconds)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # ---- Time start ----
            start_time = time.perf_counter()

            # ---- Python memory start ----
            tracemalloc.start()

            # ---- RSS memory monitor start ----
            rss_peak = 0
            stop_event = threading.Event()

            def monitor_rss():
                nonlocal rss_peak
                if psutil is None:
                    return
                proc = psutil.Process()
                while not stop_event.is_set():
                    rss_peak = max(rss_peak, proc.memory_info().rss)
                    time.sleep(interval)

            if psutil is not None:
                rss_thread = threading.Thread(target=monitor_rss)
                rss_thread.start()

            # ---- GPU memory monitor start ----
            gpu_enabled = (
                torch is not None and
                torch.cuda.is_available() and
                hasattr(torch.cuda, "reset_peak_memory_stats")
            )
            if gpu_enabled:
                num_gpus = torch.cuda.device_count()
                for device_id in range(num_gpus):
                    device = torch.device(f"cuda:{device_id}")
                    torch.cuda.set_device(device)
                    torch.cuda.reset_peak_memory_stats()

            try:
                result = func(*args, **kwargs)

            finally:
                # ---- Time end ----
                elapsed = time.perf_counter() - start_time

                # ---- Python memory end ----
                current, py_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # ---- RSS memory end ----
                if psutil is not None:
                    stop_event.set()
                    rss_thread.join()

                # ---- GPU memory end ----
                gpu_peaks = {}
                if gpu_enabled:
                    num_gpus = torch.cuda.device_count()
                    for device_id in range(num_gpus):
                        device = torch.device(f"cuda:{device_id}")
                        torch.cuda.set_device(device)
                        peak = torch.cuda.max_memory_allocated()
                        gpu_peaks[f"cuda:{device_id}"] = peak

                # ---- Print summary ----
                print(f"[PROFILE] {func.__name__}:")
                print(f"  Time elapsed: {elapsed:.2f} sec")
                print(f"  Python peak memory: {py_peak / 1024**2:.2f} MB")

                if psutil is not None:
                    print(f"  RSS peak memory: {rss_peak / 1024**2:.2f} MB")
                else:
                    print(f"  RSS peak memory: psutil not installed, skipped")

                if gpu_enabled:
                    for dev, mem in gpu_peaks.items():
                        print(f"  GPU peak memory ({dev}): {mem / 1024**2:.2f} MB")
                else:
                    print(f"  GPU peak memory: CUDA not available, skipped")

            return result

        return wrapper

    return decorator

def print_func_name(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("=" * 80, func.__name__, "=" * 80)
        result = func(*args, **kwargs)
        return result
    return wrapper