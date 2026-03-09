#!/usr/bin/env python3
"""
spark-gpu-throttle-check.py — Load the GPU and check for clock throttling
indicative of bad USB PD power negotiation on DGX Spark.

Expected behavior:
  - Healthy PD:  graphics clock reaches ~2400 MHz under load
  - Bad PD:      graphics clock stays around ~850 MHz under load (P0 but capped)

Usage: python3 spark-gpu-throttle-check.py [--samples N] [--threshold MHZ] [--warmup SECONDS] [--quiet]
"""

import argparse
import subprocess
import sys
import threading
import time

# ANSI escape codes
RED = "\033[31m"
BOLD_RED = "\033[1;31m"
RESET = "\033[0m"


def fmt(val, spec: str) -> str:
    """Format a value with a format spec, returning 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{val:{spec}}"


def query_gpu() -> dict | None:
    """Sample current GPU state from nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.current.graphics,clocks.max.graphics,"
                "pstate,power.draw,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 5:
            return None
        def safe_float(s):
            try:
                return float(s)
            except ValueError:
                return None
        return {
            "clk_mhz": safe_float(parts[0]),
            "clk_max_mhz": safe_float(parts[1]),
            "pstate": parts[2],
            "power_w": safe_float(parts[3]),
            "throttle_reason": parts[4] if parts[4] not in ("[N/A]", "0x0000000000000000", "") else None,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def gpu_load(stop_event: threading.Event, load_ready: threading.Event):
    """Sustained GPU load via cuBLAS sgemm through ctypes.
    No pip packages required — just the CUDA runtime libs from the driver."""
    import ctypes
    import ctypes.util

    def find_lib(names):
        for name in names:
            path = ctypes.util.find_library(name)
            if path:
                return ctypes.CDLL(path)
            # Try common paths directly
            for prefix in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
                           "/usr/local/cuda/lib64", "/usr/lib64"]:
                for suffix in [".so", ".so.12", ".so.11"]:
                    try:
                        return ctypes.CDLL(f"{prefix}/lib{name}{suffix}")
                    except OSError:
                        continue
        return None

    cudart = find_lib(["cudart"])
    cublas = find_lib(["cublas"])
    if not cudart or not cublas:
        missing = []
        if not cudart:
            missing.append("libcudart")
        if not cublas:
            missing.append("libcublas")
        print(f"  ERROR: Cannot load {', '.join(missing)} — is the CUDA toolkit/driver installed?")
        return

    # Type aliases
    c_size_t = ctypes.c_size_t
    c_int = ctypes.c_int
    c_float = ctypes.c_float
    c_void_p = ctypes.c_void_p

    # cudaMalloc / cudaFree
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(c_void_p), c_size_t]
    cudart.cudaMalloc.restype = c_int
    cudart.cudaFree.argtypes = [c_void_p]
    cudart.cudaFree.restype = c_int
    cudart.cudaDeviceSynchronize.restype = c_int

    # cublasCreate / cublasDestroy / cublasSgemm
    handle = c_void_p()
    cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(c_void_p)]
    cublas.cublasCreate_v2.restype = c_int
    cublas.cublasDestroy_v2.argtypes = [c_void_p]
    cublas.cublasDestroy_v2.restype = c_int
    cublas.cublasSgemm_v2.argtypes = [
        c_void_p, c_int, c_int,  # handle, transa, transb
        c_int, c_int, c_int,     # m, n, k
        ctypes.POINTER(c_float), # alpha
        c_void_p, c_int,         # A, lda
        c_void_p, c_int,         # B, ldb
        ctypes.POINTER(c_float), # beta
        c_void_p, c_int,         # C, ldc
    ]
    cublas.cublasSgemm_v2.restype = c_int

    N = 4096
    nbytes = N * N * ctypes.sizeof(c_float)
    d_a, d_b, d_c = c_void_p(), c_void_p(), c_void_p()

    try:
        for ptr in [d_a, d_b, d_c]:
            rc = cudart.cudaMalloc(ctypes.byref(ptr), nbytes)
            if rc != 0:
                print(f"  ERROR: cudaMalloc failed (rc={rc})")
                return

        rc = cublas.cublasCreate_v2(ctypes.byref(handle))
        if rc != 0:
            print(f"  ERROR: cublasCreate failed (rc={rc})")
            return

        alpha = c_float(1.0)
        beta = c_float(0.0)
        CUBLAS_OP_N = 0

        load_ready.set()

        while not stop_event.is_set():
            cublas.cublasSgemm_v2(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                ctypes.byref(alpha),
                d_a, N,
                d_b, N,
                ctypes.byref(beta),
                d_c, N,
            )
        cudart.cudaDeviceSynchronize()
    finally:
        cublas.cublasDestroy_v2(handle)
        for ptr in [d_a, d_b, d_c]:
            if ptr.value:
                cudart.cudaFree(ptr)


def run_test(num_samples: int, threshold_mhz: float, warmup: float, quiet: bool):
    if not quiet:
        print("=" * 60)
        print("  Spark GPU Throttle Check")
        print("=" * 60)
        print()

    # Pre-flight: check GPU is visible
    baseline = query_gpu()
    if baseline is None:
        print("ERROR: Cannot query GPU via nvidia-smi. Is the driver loaded?")
        sys.exit(1)

    if not quiet:
        print(f"GPU state at idle:")
        print(f"  Clock:       {fmt(baseline['clk_mhz'], '.0f')} / {fmt(baseline['clk_max_mhz'], '.0f')} MHz")
        print(f"  P-state:     {baseline['pstate']}")
        print(f"  Power:       {fmt(baseline['power_w'], '.1f')} W")
        print()

    # Start load
    stop_event = threading.Event()
    load_ready = threading.Event()
    load_thread = threading.Thread(target=gpu_load, args=(stop_event, load_ready), daemon=True)

    load_thread.start()

    # Let GPU ramp up briefly before we start recording
    if not quiet and warmup > 0:
        print(f"\nWarming up GPU ({warmup:.1f}s)...")
    time.sleep(warmup)

    if not load_ready.wait(timeout=10):
        print("ERROR: GPU load failed to start — see error above.")
        stop_event.set()
        load_thread.join(timeout=5)
        sys.exit(1)

    if not quiet:
        print(f"\nCollecting {num_samples} samples under load (0.5s interval)...")
        print(f"Threshold: {threshold_mhz:.0f} MHz\n")
        print(f"  {'#':>5s}  {'Clock (MHz)':>11s}  {'Max (MHz)':>9s}  {'PState':>6s}  {'Power (W)':>9s}")
        print(f"  {'─' * 5}  {'─' * 11}  {'─' * 9}  {'─' * 6}  {'─' * 9}")

    samples = []
    try:
        for i in range(1, num_samples + 1):
            reading = query_gpu()
            if reading and reading["clk_mhz"] is not None:
                samples.append(reading)
                if not quiet:
                    line = (
                        f"  {i:5d}  {fmt(reading['clk_mhz'], '11.0f')}  {fmt(reading['clk_max_mhz'], '9.0f')}"
                        f"  {reading['pstate']:>6s}  {fmt(reading['power_w'], '9.1f')}"
                    )
                    if reading["clk_mhz"] < threshold_mhz:
                        print(f"{RED}{line}{RESET}")
                    else:
                        print(line)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  (interrupted)")
    finally:
        stop_event.set()
        load_thread.join(timeout=5)

    # Analysis
    if not samples:
        print("\nERROR: No samples collected.")
        sys.exit(1)

    clocks = [s["clk_mhz"] for s in samples if s["clk_mhz"] is not None]
    powers = [s["power_w"] for s in samples if s["power_w"] is not None]
    throttle_reasons = {s["throttle_reason"] for s in samples if s["throttle_reason"] is not None}
    peak_clk = max(clocks)
    avg_clk = sum(clocks) / len(clocks)
    avg_pwr = sum(powers) / len(powers) if powers else 0
    pct_below = sum(1 for c in clocks if c < threshold_mhz) / len(clocks) * 100

    if not quiet:
        print()
        print("─" * 60)
        print("  RESULTS")
        print("─" * 60)
        print(f"  Samples:         {len(clocks)}")
        print(f"  Peak clock:      {peak_clk:.0f} MHz")
        print(f"  Average clock:   {avg_clk:.0f} MHz")
        print(f"  Avg power draw:  {avg_pwr:.1f} W")
        print(f"  Below threshold: {pct_below:.0f}% of samples < {threshold_mhz:.0f} MHz")
        if throttle_reasons:
            print(f"  Throttle reason: {', '.join(sorted(throttle_reasons))}")
        print()

    BOX_W = 56  # inner width between border chars

    if peak_clk < threshold_mhz:
        if quiet:
            print(f"FAIL peak={peak_clk:.0f}MHz avg={avg_clk:.0f}MHz threshold={threshold_mhz:.0f}MHz")
        else:
            print(f"{BOLD_RED}  " + "█" * (BOX_W + 2))
            for line in [
                "FAIL — GPU IS THROTTLED",
                "Clock never exceeded threshold under load.",
                "Likely cause: bad USB PD power negotiation.",
                "Try: disconnect power brick from wall and Spark,",
                "wait a minute, then reconnect.",
            ]:
                print(f"  █  {line:<{BOX_W - 2}}█")
            print("  " + "█" * (BOX_W + 2) + RESET)
        return 1
    elif pct_below > 50:
        if quiet:
            print(f"WARNING peak={peak_clk:.0f}MHz avg={avg_clk:.0f}MHz below={pct_below:.0f}%")
        else:
            print("  ┌" + "─" * BOX_W + "┐")
            for line in [
                "WARNING — GPU clocks are intermittently low.",
                f"{pct_below:.0f}% of samples below {threshold_mhz:.0f} MHz.",
                "Possible unstable PD negotiation.",
            ]:
                print(f"  │  {line:<{BOX_W - 2}}│")
            print("  └" + "─" * BOX_W + "┘")
        return 1
    else:
        if quiet:
            print(f"PASS peak={peak_clk:.0f}MHz avg={avg_clk:.0f}MHz")
        else:
            print("  ┌" + "─" * BOX_W + "┐")
            for line in [
                "PASS — GPU clocks look healthy under load.",
                f"Peak: {peak_clk:.0f} MHz, Avg: {avg_clk:.0f} MHz",
            ]:
                print(f"  │  {line:<{BOX_W - 2}}│")
            print("  └" + "─" * BOX_W + "┘")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Check for GPU throttling due to bad USB PD")
    parser.add_argument("-n", "--samples", type=int, default=20,
                        help="Number of samples to collect (default: 20)")
    parser.add_argument("-t", "--threshold", type=float, default=1400.0,
                        help="Clock threshold in MHz below which throttling is suspected (default: 1400)")
    parser.add_argument("-w", "--warmup", type=float, default=2.0,
                        help="Warm-up time in seconds before sampling begins (default: 2.0)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress sample table and details; print only PASS/FAIL result line")
    args = parser.parse_args()

    sys.exit(run_test(args.samples, args.threshold, args.warmup, args.quiet))


if __name__ == "__main__":
    main()
