import cupy as cp
import time 
def run_compute_loop():
    size = 2048
    data_a = cp.ones((size, size), dtype=cp.float32)
    data_b = cp.ones((size, size), dtype=cp.float32)
    res = cp.empty((size, size), dtype=cp.float32)

    print("[Co-Runner] Data allocated. Forcing JIT compilation and library warmup...")
    
    # --- THE WARMUP ---
    # Run the math once so CuPy loads cuBLAS and compiles the 'sin' kernel.
    # This gets the hidden HtoD transfers out of the way.
    cp.matmul(data_a, data_b, out=res)
    cp.sin(res, out=res)
    
    # Block the CPU until the GPU finishes this setup
    cp.cuda.Device().synchronize() 
    
    print("[Co-Runner] Warmup complete. Starting pure compute loop...")

    try:
        while True:
            cp.matmul(data_a, data_b, out=res)
            cp.sin(res, out=res)
            
            # Keep this synchronize here. Without it, the async CPU loop runs 
            # so fast it floods the CUDA command queue and can cause deadlocks.
            cp.cuda.Device().synchronize()

            time.sleep(0.001)  # Sleep briefly to prevent 100% CPU usage
    except KeyboardInterrupt:
        print("\n[Co-Runner] Terminated.")

if __name__ == "__main__":
    run_compute_loop()