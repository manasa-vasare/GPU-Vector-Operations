#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>

using namespace std;


// ============================================================
// CUDA VECTOR ADDITION KERNEL
// ============================================================
__global__ void vectorAddKernel(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}


// ============================================================
// CUDA VECTOR MULTIPLICATION KERNEL
// ============================================================
__global__ void vectorMultiplyKernel(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] * B[i];
}


// ============================================================
// CPU VECTOR ADDITION
// ============================================================
void cpuVectorAdd(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}


// ============================================================
// CPU VECTOR MULTIPLICATION
// ============================================================
void cpuVectorMultiply(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    for (int i = 0; i < N; i++)
        C[i] = A[i] * B[i];
}


// ============================================================
// CPU-GPU RESULT VERIFICATION
// ============================================================
bool verify(
    const vector<float>& cpu,
    const vector<float>& gpu)
{
    if (cpu.size() != gpu.size())
        return false;

    for (size_t i = 0; i < cpu.size(); i++)
    {
        if (fabs(cpu[i] - gpu[i]) > 1e-4f)
            return false;
    }

    return true;
}


// ============================================================
// CUDA ERROR CHECKING
// ============================================================
void checkCuda(
    cudaError_t err,
    const char* msg)
{
    if (err != cudaSuccess)
    {
        cerr << "CUDA error: "
             << msg
             << " : "
             << cudaGetErrorString(err)
             << endl;

        exit(EXIT_FAILURE);
    }
}


// ============================================================
// MAIN PROGRAM
// ============================================================
int main()
{
    int N = 0;
    int threadsPerBlock = 256;

    cout << "Enter vector size N: ";
    cin >> N;

    cout << "Enter threads per block (for example 128, 256 or 512): ";
    cin >> threadsPerBlock;


    // --------------------------------------------------------
    // INPUT VALIDATION
    // --------------------------------------------------------
    if (N <= 0)
    {
        cerr << "N must be greater than 0.\n";
        return 1;
    }

    if (threadsPerBlock <= 0 || threadsPerBlock > 1024)
    {
        cerr << "Threads per block must be between 1 and 1024.\n";
        return 1;
    }


    // --------------------------------------------------------
    // HOST VECTORS
    // --------------------------------------------------------
    vector<float> A(N);
    vector<float> B(N);

    vector<float> cpuAdd(N);
    vector<float> cpuMul(N);

    vector<float> gpuAdd(N);
    vector<float> gpuMul(N);


    // --------------------------------------------------------
    // INITIALISE INPUT VECTORS
    // --------------------------------------------------------
    for (int i = 0; i < N; i++)
    {
        A[i] = static_cast<float>(i % 100);
        B[i] = static_cast<float>((i % 100) + 1);
    }


    // ========================================================
    // CPU ADDITION
    // ========================================================
    auto cpuAddStart =
        chrono::high_resolution_clock::now();

    cpuVectorAdd(
        A.data(),
        B.data(),
        cpuAdd.data(),
        N);

    auto cpuAddStop =
        chrono::high_resolution_clock::now();

    double cpuAddMs =
        chrono::duration<double, milli>(
            cpuAddStop - cpuAddStart
        ).count();


    // ========================================================
    // CPU MULTIPLICATION
    // ========================================================
    auto cpuMulStart =
        chrono::high_resolution_clock::now();

    cpuVectorMultiply(
        A.data(),
        B.data(),
        cpuMul.data(),
        N);

    auto cpuMulStop =
        chrono::high_resolution_clock::now();

    double cpuMulMs =
        chrono::duration<double, milli>(
            cpuMulStop - cpuMulStart
        ).count();


    // ========================================================
    // GPU WARM-UP
    // ========================================================
    // The first CUDA call creates the CUDA context.
    // This one-time startup cost should not distort
    // the measured GPU times.

    checkCuda(
        cudaFree(0),
        "CUDA context initialisation"
    );


    // ========================================================
    // GPU MEMORY ALLOCATION
    // ========================================================
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_Add = nullptr;
    float *d_Mul = nullptr;

    size_t bytes =
        static_cast<size_t>(N) * sizeof(float);

    checkCuda(
        cudaMalloc(&d_A, bytes),
        "cudaMalloc d_A"
    );

    checkCuda(
        cudaMalloc(&d_B, bytes),
        "cudaMalloc d_B"
    );

    checkCuda(
        cudaMalloc(&d_Add, bytes),
        "cudaMalloc d_Add"
    );

    checkCuda(
        cudaMalloc(&d_Mul, bytes),
        "cudaMalloc d_Mul"
    );


    // ========================================================
    // GRID CONFIGURATION
    // ========================================================
    int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;


    // ========================================================
    // UNTIMED GPU WARM-UP KERNEL
    // ========================================================
    vectorAddKernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_A,
        d_B,
        d_Add,
        N
    );

    checkCuda(
        cudaGetLastError(),
        "warm-up kernel launch"
    );

    checkCuda(
        cudaDeviceSynchronize(),
        "warm-up synchronise"
    );


    // ========================================================
    // CUDA EVENTS
    // ========================================================
    cudaEvent_t totalStart;
    cudaEvent_t totalStop;

    cudaEvent_t addStart;
    cudaEvent_t addStop;

    cudaEvent_t mulStart;
    cudaEvent_t mulStop;


    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);

    cudaEventCreate(&addStart);
    cudaEventCreate(&addStop);

    cudaEventCreate(&mulStart);
    cudaEventCreate(&mulStop);


    // ========================================================
    // TOTAL GPU PHASE START
    // ========================================================
    cudaEventRecord(totalStart);


    // --------------------------------------------------------
    // COPY A TO GPU
    // --------------------------------------------------------
    checkCuda(
        cudaMemcpy(
            d_A,
            A.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "Copy A to GPU"
    );


    // --------------------------------------------------------
    // COPY B TO GPU
    // --------------------------------------------------------
    checkCuda(
        cudaMemcpy(
            d_B,
            B.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "Copy B to GPU"
    );


    // ========================================================
    // GPU VECTOR ADDITION
    // ========================================================
    cudaEventRecord(addStart);

    vectorAddKernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_A,
        d_B,
        d_Add,
        N
    );

    cudaEventRecord(addStop);

    checkCuda(
        cudaGetLastError(),
        "vectorAddKernel launch"
    );


    // ========================================================
    // GPU VECTOR MULTIPLICATION
    // ========================================================
    cudaEventRecord(mulStart);

    vectorMultiplyKernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_A,
        d_B,
        d_Mul,
        N
    );

    cudaEventRecord(mulStop);

    checkCuda(
        cudaGetLastError(),
        "vectorMultiplyKernel launch"
    );


    // ========================================================
    // COPY GPU RESULTS BACK TO CPU
    // ========================================================
    checkCuda(
        cudaMemcpy(
            gpuAdd.data(),
            d_Add,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "Copy addition result"
    );


    checkCuda(
        cudaMemcpy(
            gpuMul.data(),
            d_Mul,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "Copy multiplication result"
    );


    // ========================================================
    // TOTAL GPU PHASE END
    // ========================================================
    cudaEventRecord(totalStop);

    checkCuda(
        cudaEventSynchronize(totalStop),
        "synchronise total timer"
    );


    // ========================================================
    // CALCULATE GPU TIMES
    // ========================================================
    float gpuAddMs = 0.0f;
    float gpuMulMs = 0.0f;
    float totalGpuMs = 0.0f;


    cudaEventElapsedTime(
        &gpuAddMs,
        addStart,
        addStop
    );

    cudaEventElapsedTime(
        &gpuMulMs,
        mulStart,
        mulStop
    );

    cudaEventElapsedTime(
        &totalGpuMs,
        totalStart,
        totalStop
    );


    // ========================================================
    // VERIFY RESULTS
    // ========================================================
    bool addOk =
        verify(cpuAdd, gpuAdd);

    bool mulOk =
        verify(cpuMul, gpuMul);


    // ========================================================
    // CALCULATE SPEEDUPS
    // ========================================================
    double addSpeedup =
        (gpuAddMs > 0)
        ? cpuAddMs / gpuAddMs
        : 0.0;

    double mulSpeedup =
        (gpuMulMs > 0)
        ? cpuMulMs / gpuMulMs
        : 0.0;

    double endToEnd =
        (totalGpuMs > 0)
        ? (cpuAddMs + cpuMulMs) / totalGpuMs
        : 0.0;


    // ========================================================
    // DISPLAY RESULTS
    // ========================================================
    cout << fixed << setprecision(6);

    cout << "\n========== GPU VECTOR OPERATIONS ==========\n";

    cout << "Vector Size = "
         << N
         << "\n";

    cout << "Threads per Block = "
         << threadsPerBlock
         << "\n";

    cout << "Blocks per Grid = "
         << blocksPerGrid
         << "\n";


    // --------------------------------------------------------
    // ADDITION RESULTS
    // --------------------------------------------------------
    cout << "\nVector Addition\n";

    cout << "CPU Time = "
         << cpuAddMs
         << " ms\n";

    cout << "GPU Kernel Time = "
         << gpuAddMs
         << " ms\n";

    cout << "Kernel Speedup = "
         << addSpeedup
         << "x\n";

    cout << "Correctness = "
         << (addOk ? "PASS" : "FAIL")
         << "\n";


    // --------------------------------------------------------
    // MULTIPLICATION RESULTS
    // --------------------------------------------------------
    cout << "\nVector Multiplication\n";

    cout << "CPU Time = "
         << cpuMulMs
         << " ms\n";

    cout << "GPU Kernel Time = "
         << gpuMulMs
         << " ms\n";

    cout << "Kernel Speedup = "
         << mulSpeedup
         << "x\n";

    cout << "Correctness = "
         << (mulOk ? "PASS" : "FAIL")
         << "\n";


    // --------------------------------------------------------
    // TOTAL GPU TIME
    // --------------------------------------------------------
    cout << "\nTotal GPU Phase Time "
         << "(copy in + both kernels + copy out) = "
         << totalGpuMs
         << " ms\n";

    cout << "End-to-End Speedup "
         << "(CPU add+mul / total GPU phase) = "
         << endToEnd
         << "x\n";


    // ========================================================
    // SAVE RESULTS TO CSV
    // ========================================================
    bool fileExists = false;

    {
        ifstream check("results.csv");
        fileExists = check.good();
    }


    ofstream csv(
        "results.csv",
        ios::app
    );


    // --------------------------------------------------------
    // WRITE CSV HEADER
    // --------------------------------------------------------
    if (!fileExists)
    {
        csv
            << "N,ThreadsPerBlock,"
            << "CPU_Add_ms,CPU_Mul_ms,"
            << "GPU_Add_Kernel_ms,"
            << "GPU_Mul_Kernel_ms,"
            << "Total_GPU_ms,"
            << "Add_Speedup,"
            << "Mul_Speedup,"
            << "EndToEnd_Speedup,"
            << "Add_Correct,"
            << "Mul_Correct\n";
    }


    // --------------------------------------------------------
    // WRITE CSV DATA
    // --------------------------------------------------------
    csv << fixed << setprecision(6);

    csv
        << N << ","
        << threadsPerBlock << ","
        << cpuAddMs << ","
        << cpuMulMs << ","
        << gpuAddMs << ","
        << gpuMulMs << ","
        << totalGpuMs << ","
        << addSpeedup << ","
        << mulSpeedup << ","
        << endToEnd << ","
        << (addOk ? "PASS" : "FAIL") << ","
        << (mulOk ? "PASS" : "FAIL")
        << "\n";


    csv.close();


    cout << "\nResult appended to results.csv\n";


    // ========================================================
    // CLEANUP
    // ========================================================
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Add);
    cudaFree(d_Mul);

    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalStop);

    cudaEventDestroy(addStart);
    cudaEventDestroy(addStop);

    cudaEventDestroy(mulStart);
    cudaEventDestroy(mulStop);


    // ========================================================
    // PROGRAM EXIT
    // ========================================================
    return (addOk && mulOk) ? 0 : 2;
}
