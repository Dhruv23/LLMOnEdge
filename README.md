# LLMOnEdge  
**End-to-End Profiling and Prediction of GPT-2 Inference Latency on Edge GPUs**

This repository contains a complete experimental pipeline for measuring and modeling the relationship between **prompt length (token count)** and **inference latency** for large language models deployed on edge GPUs. Using **GPT-2 Medium** compiled with **TensorRT**, we collect empirical latency measurements and train lightweight machine learning models to predict runtime cost as a function of token count.

The project is motivated by the need for **latency-aware decision making** in real-time and resource-constrained systems such as robotics, embedded AI, and edge computing platforms.

---

## 7-Step Experimental Pipeline

This project follows a clear, reproducible 7-step workflow, from data collection to latency prediction.

---

## Step 1: Select an Open-Source Prompt Dataset

We use an **open-source Amazon reviews dataset** as a source of realistic natural-language prompts. Amazon reviews are well-suited for this study because:

- They resemble real user prompts
- Their lengths vary naturally across a wide range
- They are publicly available and reproducible

The raw dataset is downloaded locally using Hugging Face tooling.  
Large derived data files (e.g., tokenized prompt tables) are **not committed** to the repository and are regenerated as needed.

---

## Step 2: Tokenize Prompts and Construct Length Buckets

Each review is tokenized using the **GPT-2 tokenizer** to compute its token length.

Prompts are then grouped into token-length buckets to ensure uniform coverage across context sizes, for example:

- 8–16 tokens  
- 17–32 tokens  
- 33–64 tokens  
- 65–128 tokens  
- 129–256 tokens  
- 257–512 tokens  

A fixed number of prompts is sampled from each bucket.  
This produces a controlled dataset spanning the full range of prompt lengths supported by the model.

---

## Step 3: Build a TensorRT Engine with Dynamic Shapes

The GPT-2 Medium model is exported from Hugging Face to ONNX using a **causal language model with past key/value caching**.

The ONNX model is then compiled into a **TensorRT engine** with:

- FP16 precision
- Dynamic input shapes
- Optimization profiles covering the full prompt-length range

Dynamic shapes allow a single engine to run inference efficiently across different context lengths.

The resulting serialized engine (`.plan`) is used for all subsequent experiments.

---

## Step 4: Define the “Single-Round” Inference Experiment

In this project, a **single round** is defined as:

- One decode step (single token generation)
- With a variable-length context represented by the past key/value cache

Prompt length is therefore modeled by **past sequence length**, not by feeding the full prompt tokens each time.

This approach:
- Matches common autoregressive decoding behavior
- Keeps experiments fast and controlled
- Allows isolation of latency effects due purely to context length

Batch size is fixed to 1 for all experiments.

---

## Step 5: Measure Latency with Repeated Trials

For each prompt-length bucket:

1. The TensorRT execution context is configured with the corresponding dynamic shapes  
2. A number of **warmup runs** are executed (not recorded)  
3. The inference is run repeatedly (30–100 times)  
4. Latency measurements are recorded  

The following metrics are collected:
- End-to-end latency (milliseconds)
- GPU compute time
- Throughput (tokens per second)

All measurements are written to CSV files for downstream analysis.

---

## Step 6: Aggregate Results and Visualize Trends

The raw measurements are aggregated by prompt-length bucket to compute:

- Mean latency
- Variance and coefficient of variation
- Throughput trends
- Outliers

The repository includes a rich set of plots in the `plots/` directory, including:

- Latency vs. token count
- Latency distributions by bucket
- Variability and stability analysis
- Compute vs. end-to-end breakdown
- Throughput scaling

An interactive summary dashboard is available at:

plots/index.html


---

## Step 7: Train a Lightweight Latency Predictor

Using the collected measurements, we train lightweight regression models to predict inference latency from token count.

Models explored include:
- Linear regression (baseline)
- Random Forest regression

Features include:
- Token count
- Log-transformed token count

Models are evaluated using standard regression metrics (MAE, RMSE, R²) with careful train/test separation to avoid data leakage.

The resulting predictor enables **fast, hardware-specific latency estimation** without executing inference.

---

## Repository Structure

```text
LLMOnEdge/
├── engines/ # TensorRT serialized engines (.plan)
├── onnx/ # Exported ONNX models
├── src/
│ ├── profile_memory.cpp # TensorRT inference + timing harness
│ ├── build_prompts.py # Prompt dataset preparation
│ └── train_predictor.py # Latency prediction model
├── plots/
│ ├── index.html # Interactive results dashboard
│ ├── *.png # Visualization outputs
│ └── *.csv # Aggregated statistics
├── .gitignore
├── README.md
└── requirements.txt
```
---

## Reproducibility Notes

- All models and datasets are open-source
- Large intermediate data files are regenerated locally
- TensorRT version, GPU architecture, and clock settings affect results
- The pipeline is designed to be deterministic where possible

---

## Applications

- Latency-aware scheduling for LLMs
- Edge AI and robotics systems
- Real-time inference planning
- Performance modeling and system design

---

## Future Work

- Prefill vs. decode latency comparison
- Multi-batch inference analysis
- Power and energy modeling
- Cross-model comparisons (larger LLMs)
- Integration into real-time planners and controllers

---

## License

This project is intended for research and educational use.  
Model weights and datasets follow their respective upstream licenses.
