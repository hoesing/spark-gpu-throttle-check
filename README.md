# spark-gpu-throttle-check

A diagnostic tool for detecting GPU clock throttling on NVIDIA DGX Spark (and related GB10) systems suspected to be caused by bad USB Power Delivery (PD) negotiation.

## The Problem

DGX Spark systems can occasionally end up in a degraded power state where the GPU remains in P0 state but its clock speed is capped far below normal — typically below 850 MHz instead of the expected ~2400 MHz. This causes significant performance degradation that can be difficult to diagnose since the GPU otherwise appears healthy. Faulty USB PD negotiation with the power brick is a suspected cause.

## What This Tool Does

The script loads the GPU with a sustained compute workload (4096×4096 FP32 matrix multiplications via cuBLAS) and monitors the clock speed. Under normal power delivery, the GPU should ramp up well past 1400 MHz. If the clocks stay below the threshold under full load, the GPU is being power-throttled.

No pip packages are required — the script calls cuBLAS directly via ctypes using the CUDA runtime libraries that ship with the NVIDIA driver.

## Usage

```bash
python3 spark-gpu-throttle-check.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--samples` | 20 | Number of samples to collect |
| `-t`, `--threshold` | 1400 | Clock threshold (MHz) below which throttling is suspected |
| `-w`, `--warmup` | 2.0 | Warm-up time (seconds) before sampling begins |
| `-q`, `--quiet` | off | Suppress sample table and details; print only a PASS/FAIL result line |

### Examples

```bash
# Quick check with defaults
python3 spark-gpu-throttle-check.py

# Longer run with more samples
python3 spark-gpu-throttle-check.py -n 50

# Lower threshold for a less strict check
python3 spark-gpu-throttle-check.py -t 1000

# Longer warm-up if the first sample still shows low power draw
python3 spark-gpu-throttle-check.py -w 3

# Quiet mode for scripting — prints only PASS/FAIL result line
python3 spark-gpu-throttle-check.py -q
```

### Exit Codes

- `0` — PASS, GPU clocks are healthy
- `1` — FAIL or WARNING, GPU appears throttled

This makes it easy to use in scripts:

```bash
python3 spark-gpu-throttle-check.py -q && echo "GPU clocks OK" || echo "Possible PD issue detected"
```

## Sample Output

### Healthy System

```
============================================================
  Spark GPU Throttle Check
============================================================

GPU state at idle:
  Clock:       208 / 3003 MHz
  P-state:     P8
  Power:       4.5 W

Collecting 20 samples under load (0.5s interval)...
Threshold: 1400 MHz

      #  Clock (MHz)  Max (MHz)  PState  Power (W)
  ─────  ───────────  ─────────  ──────  ─────────
      1         2424       3003      P0       87.2
      2         2424       3003      P0       87.6
     ...
     20         2483       3003      P0       89.4

────────────────────────────────────────────────────────────
  RESULTS
────────────────────────────────────────────────────────────
  Samples:         20
  Peak clock:      2483 MHz
  Average clock:   2446 MHz
  Avg power draw:  84.0 W
  Below threshold: 0% of samples < 1400 MHz

  ┌────────────────────────────────────────────────────────┐
  │  PASS — GPU clocks look healthy under load.            │
  │  Peak: 2483 MHz, Avg: 2446 MHz                         │
  └────────────────────────────────────────────────────────┘
```

### Throttled System (Suspected Bad PD)

```
      #  Clock (MHz)  Max (MHz)  PState  Power (W)
  ─────  ───────────  ─────────  ──────  ─────────
      1          481       3003      P0        5.8   (red)
      2          858       3003      P0       15.8   (red)
     ...
     20          507       3003      P0       11.1   (red)

  ██████████████████████████████████████████████████████████
  █  FAIL — GPU IS THROTTLED                               █
  █  Clock never exceeded threshold under load.            █
  █  Likely cause: bad USB PD power negotiation.           █
  █  Try: disconnect power brick from wall and Spark,      █
  █  wait a minute, then reconnect.                        █
  ██████████████████████████████████████████████████████████
```

Failing samples and the FAIL banner are displayed in red in the terminal.

## Fix

If the tool reports a failure, try disconnecting the power brick from both the wall outlet and the Spark. Wait a minute, then reconnect. Run the check again to verify the issue is resolved.

## Requirements

- Python 3.10+
- NVIDIA GPU driver with cuBLAS and CUDA runtime libraries (standard on DGX OS)
- `nvidia-smi` in PATH
