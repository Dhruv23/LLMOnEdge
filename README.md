# LLMOnEdge
**End-to-End Profiling and Prediction of GPT-2 Inference Latency on Edge GPUs**

This repository is designed to profile GPT-2 Medium inference latency on edge GPUs using TensorRT. It measures and models the relationship between context length and inference latency, considering memory utilization, PCIe bandwidth, and CPU/RAM usage.

## Key Scripts

### 1. `src/measure_all.py`
Generates an ML dataset by running `trtexec` with a specified execution plan. It exports inference latencies to a CSV file. It continuously monitors and records hardware stats using `pynvml` and `psutil`, such as:
- Device Execution Memory
- Device Context Memory
- Maximum Used Memory
- Memory Utilization
- Peak PCIe TX/RX Bandwidth
- CPU Utilization and System RAM Usage

*Note: If an older CSV schema is detected, missing metrics will be backfilled with `-1`.*

### 2. `src/measure_combined.py`
A combined measurement script that runs the contexts defined in the standard execution plan and automatically extends the test by running hardcoded high-latency buckets (`16250`, `17500`, `18750`, `30000`). This is useful for testing edge cases, stress testing memory limits, and capturing oversubscription metrics. It writes to the same dataset schema as `measure_all.py`.

### 3. `src/corunner.py`
A script to simulate GPU interference. It allocates matrices and runs a tight `cp.matmul` and `cp.sin` loop utilizing CuPy. This allows you to evaluate how concurrent GPU workloads affect GPT-2 inference latency and system resource metrics.

### 4. `src/plot_all_measurements.py`
A comprehensive visualization script that generates plots from the collected datasets. It handles:
- **Standard Latency:** Median vs P99, Jitter, Throughput.
- **Predictor Training:** Execution time and memory usage prediction across different percentiles using Linear Regression and XGBoost.
- **Growth Curves:** Illustrates the "Latency Cliff" and VRAM limit oversubscription.
- **Memory Oversubscription Metrics:**
  - Saturation points (Memory Utilization % vs Context Length).
  - PCIe Thrashing signatures (Peak PCIe Bandwidth vs Context Length).
  - Latency cost colored by memory utilization or PCIe bandwidth.

## Output

The standard ML dataset outputs to `training_dataset.csv` or `combined_dataset.csv`. The plots are generated within the `plots/` directory, broken down into raw latency stats, combined model plots, ML predictions, system resource metrics, and oversubscription signatures.
