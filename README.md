# GPU Vector Operations

## Objective
The objective of this project is to implement and analyze vector operations (addition and multiplication) on the GPU using CUDA threads, and compare their performance with sequential execution on the CPU.

## Team Members
- Aditya R Gavimath (01FE24BCI120)
- Manasa B Vasare (01FE24BCI113)
- Renuka Kagadal (01FE24BCI119)

## How to Build and Run
1. Navigate to the `src` directory.
2. Compile the CUDA source code using `nvcc`:
   ```bash
   nvcc -o vector_operations your_source_file.cu
   ```
   *(Note: Please ensure the main CUDA source file is added to the `src` folder).*
3. Run the executable to generate the performance metrics.
4. Run the Python script to plot the results:
   ```bash
   python plot_results.py
   ```

## Results Summary
The parallel execution on the GPU demonstrates significant speedup compared to sequential CPU execution, especially as the vector size (N) increases. The results were verified for correctness ("PASS") and evaluated across varying vector sizes and threads-per-block configurations.

### 1. Execution Time (CPU vs GPU)
The graph below illustrates the total execution time of the CPU (addition + multiplication) compared to the GPU kernel execution time as the vector size increases. The GPU drastically outperforms the CPU for larger vectors.

![CPU vs GPU Execution Time](graphs/graph1_time_vs_size.png)

### 2. GPU Speedup
This graph highlights the speedup multiplier achieved by the GPU over the CPU. As N reaches 10 million elements, the GPU achieves a speedup of roughly 10-12x for individual kernels.

![GPU Speedup vs Size](graphs/graph2_speedup_vs_size.png)

### 3. Impact of Threads per Block
The final graph analyzes the performance of the GPU kernels using different threads-per-block configurations (e.g., 128, 256, 512) for the largest vector size.

![Kernel Time vs Threads](graphs/graph3_kernel_vs_threads.png)

For a comprehensive analysis, observations, and conclusion, please refer to the detailed report in the `report/` directory.
