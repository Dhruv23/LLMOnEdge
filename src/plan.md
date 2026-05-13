Both your LLM and your Corunner will exist in the same memory space, fighting for the same Global Interpreter Lock (GIL) on the CPU, and the same scheduler on the GPU.
Here is the exact architectural blueprint to build this single-process, multi-threaded system.

---

### Step 1: Threading and GIL Management

Python’s Global Interpreter Lock (GIL) is your first enemy. If your Corunner thread hogs the CPU, it will prevent the vLLM thread from launching its GPU kernels, causing catastrophic latency spikes.

* **The Architecture:** * **Thread 1 (Main/vLLM):** Runs the vLLM async engine.
* **Thread 2 (Corunner):** Runs your background workload.


* **The Rule:** The Corunner thread must do almost *zero* CPU work. It must simply dispatch GPU kernels and immediately yield. Do not use heavy `for` loops in Python to do math. Dispatch the math to the GPU.
* **Avoid `stream.synchronize()`:** Calling `stream.synchronize()` in your Corunner thread forces the CPU to wait and can cause GIL contention. Instead, use CUDA Events (`cp.cuda.Event` or `torch.cuda.Event`) to check if the GPU work is done without blocking the CPU thread.

### Step 2: Asymmetric CUDA Stream Priorities (The Core Solution)

Since you cannot use MPS to isolate the processes, you must use **hardware priorities** to tell the GPU scheduler who is boss.

By default, all streams have the same priority (Priority 0). If you put the Corunner on Stream B and vLLM on Stream A with equal priority, the GPU scheduler will just round-robin them, causing the sawtooth interference you saw earlier.

You must explicitly create a **Low Priority Stream** for the Corunner.

* **How it works:** Modern GPUs support a range of stream priorities (usually 0 for default/high, and lower negative numbers for lower priority).
* **The Hardware Logic:** When the GPU finishes a micro-kernel, its hardware scheduler looks at the queues. If vLLM (Priority 0) and the Corunner (Priority -1) both have kernels ready, the hardware *guarantees* the vLLM kernel is dispatched to the SMs immediately. The Corunner only gets to launch when the vLLM queue is completely empty.

*(Note: In CuPy, you can check priority ranges with `cupy.cuda.runtime.streamGetPriorityRange()` and create a stream with `cupy.cuda.Stream(non_blocking=True, priority=lowest_priority)`).*

### Step 3: Micro-Slicing the Corunner Workload (Crucial)

This is the most critical step for a single-process setup. **Stream priority only dictates *when* a kernel starts, not if it can be interrupted.**

GPUs generally do not support preemption in the middle of a kernel execution.

* **The Danger:** If your Corunner launches a massive `cupy.matmul` that takes 5 milliseconds to run, and 0.1ms into that execution, vLLM needs the GPU, **vLLM has to wait 4.9ms for the Corunner to finish.** Stream priority cannot stop a kernel once it is running on the Streaming Multiprocessors.
* **The Fix:** You must manually slice your Corunner workload into **micro-kernels** that execute in under 50-100 microseconds.
* Instead of doing a `1024x1024` matrix multiplication all at once, break it into tiny tiles (e.g., `64x64`).
* By submitting a rapid-fire stream of microscopic kernels to the Low Priority Stream, you give the GPU scheduler hundreds of opportunities per millisecond to pause the Corunner and yield the hardware back to the high-priority vLLM stream.

### Step 4: Turn CUDA Graphs Back On

Your trace data proved that `enforce_eager=True` creates unpredictable microscopic gaps due to CPU launch overhead.

In a single-process architecture, these gaps are deadly because they invite the Corunner to step in, only for the CPU to suddenly launch the next vLLM kernel, resulting in a bandwidth collision (the peak of your sawtooth).

1. Allow vLLM to use **CUDA Graphs**.
2. CUDA Graphs bundle the entire layer's execution into a single, predictable block on the GPU.
3. Because the LLM execution is now a solid, predictable block on the High Priority stream, your micro-sliced Corunner kernels on the Low Priority stream will naturally pool into the *actual* memory-bound bottlenecks (like Attention KV cache loading) or idle periods, rather than colliding with eager-mode jitter.

### Summary of the Single-Process Architecture

1. **Spawn a Python Thread** for the Corunner.
2. **Create a CUDA Stream with the lowest possible priority** for the Corunner thread.
3. **Slice the Corunner's workload** into highly fragmented, sub-millisecond micro-kernels.
4. **Submit the micro-kernels** to the low-priority stream.
5. **Use CUDA Events to poll** for completion, completely avoiding `stream.synchronize()` to keep the Python GIL free for vLLM.