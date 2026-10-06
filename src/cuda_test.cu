#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU()
{
    printf("Hello from GPU! Thread %d\n", threadIdx.x);
}

int main()
{
    std::cout << "CUDA test started.\n";

    helloFromGPU<<<1, 5>>>();

    cudaDeviceSynchronize();

    std::cout << "CUDA test completed.\n";

    return 0;

}