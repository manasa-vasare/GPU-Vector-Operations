# GPU Vector Operations using CUDA

## Overview

This experiment implements and evaluates **vector addition** and **element-wise vector multiplication** using both sequential CPU execution and parallel GPU execution using NVIDIA CUDA.

The experiment compares CPU and GPU performance for different vector sizes and different CUDA thread-block configurations.

The study focuses on:

- CPU execution time
- GPU kernel execution time
- Total GPU phase time
- Kernel speedup
- End-to-end speedup
- Correctness verification
- Effect of vector size
- Effect of threads per block
- GPU memory-transfer and kernel-launch overhead
- Performance analysis using graphs

## Team Members
- Aditya R Gavimath (01FE24BCI120)
- Manasa B Vasare (01FE24BCI113)
- Renuka Kagadal (01FE24BCI119)

---

# Problem Statement

Given two vectors `A` and `B` of length `N`, perform the following operations.

### Vector Addition

```text
C[i] = A[i] + B[i]
```

### Element-wise Vector Multiplication

```text
C[i] = A[i] * B[i]
```

The CPU performs the operations sequentially, while CUDA assigns independent vector elements to GPU threads so that the operations can execute in parallel.

---

# Objectives

1. Implement sequential vector addition on the CPU.
2. Implement sequential vector multiplication on the CPU.
3. Implement vector addition using CUDA.
4. Implement vector multiplication using CUDA.
5. Verify CPU and GPU correctness.
6. Measure CPU execution time.
7. Measure GPU kernel execution time.
8. Measure total GPU phase time.
9. Calculate kernel speedup.
10. Calculate end-to-end speedup.
11. Study different vector sizes.
12. Study different CUDA thread-block sizes.
13. Store experimental measurements in CSV format.
14. Generate performance graphs.
15. Analyze GPU performance and overhead.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| C++ | Sequential CPU implementation |
| NVIDIA CUDA | Parallel GPU implementation |
| `nvcc` | CUDA compiler |
| Microsoft Visual C++ | Windows host compiler |
| NVIDIA RTX 4500 Ada Generation | GPU |
| Python 3 | Result processing and graph generation |
| Pandas | CSV data processing |
| Matplotlib | Graph generation |
| PowerShell | Command execution |

---

# Hardware and Software Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA RTX 4500 Ada Generation |
| NVIDIA Driver | 610.47 |
| CUDA Toolkit | 13.4 |
| `nvcc` | 13.4.92 |
| Host Compiler | Microsoft Visual C++ |
| Operating System | Windows |

---

# CUDA Execution Model

CUDA organizes GPU execution using a hierarchy of:

```text
Grid
 |
 +---- Block
        |
        +---- Thread
        +---- Thread
        +---- Thread
        +---- ...
```

Each CUDA thread is responsible for one logical vector element.

The global thread index is calculated as:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

The kernel checks whether the calculated index is within the vector:

```cpp
if (i < N)
```

The number of blocks required is calculated using:

```cpp
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
```

This allows the program to handle vector sizes that are not exact multiples of the selected number of threads per block.

---

# CPU and GPU Approach

## Sequential CPU

The CPU processes the vector elements one after another:

```text
for i = 0 ... N-1

    C[i] = A[i] + B[i]

    C[i] = A[i] * B[i]
```

---

## Parallel GPU

CUDA assigns vector elements to independent GPU threads:

```text
Thread 0 → C[0]
Thread 1 → C[1]
Thread 2 → C[2]
Thread 3 → C[3]
...
```

The GPU kernel performs:

```cpp
C[i] = A[i] + B[i];
```

and:

```cpp
C[i] = A[i] * B[i];
```

---

# Experimental Configuration

## Vector Size Experiment

The following vector sizes were tested:

| Test | Vector Size |
|---:|---:|
| 1 | 1,000 |
| 2 | 10,000 |
| 3 | 100,000 |
| 4 | 1,000,000 |
| 5 | 10,000,000 |

For the vector-size experiment:

```text
Threads per Block = 256
```

Each configuration was executed three times.

```text
5 vector sizes × 3 repetitions = 15 executions
```

---

## Thread Configuration Experiment

For the thread study:

```text
Vector Size = 10,000,000
```

The following configurations were tested:

| Configuration | Threads per Block |
|---|---:|
| A | 128 |
| B | 256 |
| C | 512 |

Each configuration was executed three times.

```text
3 thread configurations × 3 repetitions = 9 executions
```

---

# Execution Commands

## Compilation and Execution

The CPU and GPU logic are combined in a single program (`src/vector_operations.cu`). The program first calculates the baseline sequentially on the CPU, then executes the parallel version on the GPU, and compares the timings automatically.

```powershell
# ============================================================
# CUDA COMPILATION
# ============================================================

# Compile the CUDA program
nvcc -O2 src\vector_operations.cu -o vector_operations.exe

# Verify CUDA executable
dir vector_operations.exe
```

## Running the Program

Run the program manually:

```powershell
# ============================================================
# EXECUTION
# ============================================================

# Start the CUDA program
.\vector_operations.exe
```

Enter your desired configurations when prompted:

```text
Vector Size:
1000

Threads per Block:
256
```

The program will execute both CPU and GPU versions and verify that:

```text
Addition Correctness    = PASS
Multiplication Correctness = PASS
```

The program will also automatically append the timing results to `results/results.csv`.

---

# Python Environment and Graphing

Python is used to process the experimental CSV data and generate the performance graphs.

```powershell
# ============================================================
# GRAPH GENERATION
# ============================================================

# Run the graphing script located in the src directory
python src\plot_results.py

# Verify generated graph files
dir graphs

# Verify generated summary
dir results\summary.csv
```

The script generates:

```text
results/summary.csv
graphs/graph1_time_vs_size.png
graphs/graph2_speedup_vs_size.png
graphs/graph3_kernel_vs_threads.png
```

---

# Performance Analysis

## Effect of Vector Size

For small vectors, GPU overhead can dominate the actual computation.

The GPU has overhead associated with:

- Memory allocation
- Host-to-device transfer
- Device-to-host transfer
- Kernel launch
- Synchronization
- CUDA runtime initialization

Therefore, the GPU may not provide an overall advantage for very small vector sizes (e.g. `N = 1,000`).

As the vector size increases, the GPU receives more independent work and can utilize its parallel processing capability more effectively. At `N = 10,000,000`, the GPU vastly outperforms the CPU in computation (Kernel time).

---

# Kernel Speedup vs End-to-End Speedup

An important result from this experiment is:

```text
Kernel Speedup ≠ End-to-End Speedup
```

The difference occurs because the end-to-end measurement includes GPU memory-transfer and execution overhead.

Therefore:

```text
Kernel timing
```

answers:

> How fast can the GPU perform the computation itself?

while:

```text
End-to-end timing
```

answers:

> How fast is the complete GPU-based application after including data movement and GPU overhead?

---

#  Thread Configuration Analysis

The thread configuration study was performed using `N = 10,000,000`.

The lowest recorded total GPU phase time among the tested configurations was `512 threads/block`.

Changing the block size changes the number of blocks, thread scheduling, GPU resource utilization, and kernel execution behavior. Therefore, different thread configurations can produce different execution times.

---

# Conclusion

This experiment demonstrates the use of NVIDIA CUDA for parallel vector operations and compares GPU execution with a sequential CPU baseline.

The experiment shows that GPU acceleration depends strongly on workload size and execution overhead. For small vectors, GPU launch and memory-transfer overhead can dominate the computation. For large vectors, the GPU provides much greater parallelism and the CUDA kernels achieve significant speedups.

The overall conclusion is:

> **CUDA can provide substantial kernel-level acceleration for sufficiently large parallel workloads, but the complete application performance is also affected by memory transfers, kernel-launch overhead, synchronization, and other GPU runtime costs.**

---

# Repository Structure

```text
gpu-vector-operations/
│
├── README.md
├── data/
├── graphs/
│   ├── graph1_time_vs_size.png
│   ├── graph2_speedup_vs_size.png
│   └── graph3_kernel_vs_threads.png
│
├── report/
│   ├── CUDA_Vector_Operations_Observations_Analysis_Conclusion.docx
│   └── screenshots/
│       ├── Screenshot 2026-09-18 102643.png
│       ├── ...
│       └── Screenshot 2026-10-06 111216.png
│
├── results/
│   ├── results.csv
│   └── summary.csv
│
└── src/
    ├── cuda_test.cu
    ├── cuda_test.cu.txt
    ├── plot_results.py
    └── vector_operations.cu
```
