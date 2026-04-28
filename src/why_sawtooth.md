The "sawtooth" pattern you observed during the Decode phase is a classic signature of **micro-level hardware contention and phase drift** between two uncoordinated loops running on the same GPU. 

Because you set `enforce_eager=True` and ran an autoregressive decode, the GPU was experiencing a very specific rhythm of traffic. Here is exactly what caused the oscillating spikes and drops (the sawtooth) in your Corunner's latency:

### 1. The Decode Rhythm (Lots of tiny, memory-bound kernels)
Unlike the Prefill phase—which is one massive, monolithic block of computation—the Decode phase generates tokens one by one. To generate a single token, vLLM has to run data through every single layer of the GPT-2 model. 
Because you disabled CUDA Graphs (`enforce_eager=True`), the CPU has to explicitly launch every single micro-kernel (Attention, LayerNorm, MLP) one after the other. This creates a highly fragmented workload: tiny bursts of GPU activity separated by tiny microsecond gaps where the CPU is preparing the next launch.

### 2. The Collision (The Peaks of the Sawtooth)
The peaks of your sawtooth (where your Corunner latency spiked to ~1.47ms) happen when a `cupy` kernel and a vLLM Decode kernel **collide on the GPU at the exact same microsecond**.
* The vLLM decode phase is notoriously **memory-bound**. It doesn't need all the GPU's compute cores (SMs), but it aggressively hogs the VRAM bandwidth to load the model weights and the KV cache.
* Your Corunner is trying to run a `matmul` and `sin` operation. 
* When they collide, the GPU's hardware scheduler tries to run them concurrently. However, because vLLM is saturating the memory bus, your `cupy` kernel is starved of memory bandwidth. It sits on the SMs waiting for data, causing that loop iteration to take significantly longer.

### 3. The Interleaving (The Troughs of the Sawtooth)
The drops in your sawtooth (where latency falls back down to ~0.40ms, near idle baseline) happen when your `cupy` kernel manages to **sneak into the gaps**.
Because vLLM is running eager layer-by-layer kernels, there are tiny fractions of a millisecond where the GPU is briefly idle waiting for Python to launch the next layer. If your `cupy` kernel is submitted at that exact moment, it has the entire GPU to itself. It executes quickly with zero memory contention, resulting in a fast loop iteration.

### 4. Phase Drift (Why it oscillates continuously)
Why does it constantly bounce between these peaks and troughs? Because the two loops are operating at different, unaligned frequencies.
* Your Corunner loop takes about **0.4ms to 1.4ms**.
* Generating a single token in vLLM might take **10ms to 20ms**.

Imagine two turning gears of different sizes. They are completely un-synchronized. 
* On iteration 1, the Corunner hits a gap (Fast). 
* On iteration 2, the gears drift, and the Corunner partially overlaps with a vLLM kernel (Medium). 
* On iteration 3, they perfectly collide, fighting for memory bandwidth (Slow/Peak).
* On iteration 4, the Corunner finishes its slow run just as vLLM is between tokens, hitting a massive gap (Fast/Drop).

This constant drifting in and out of alignment between the Corunner's `while` loop and vLLM's `for token in range(128)` loop creates the exact geometric "sawtooth" jitter you see in the trace data.