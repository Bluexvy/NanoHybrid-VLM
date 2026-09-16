#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float*B, float* C, int M, int N, int K, float alpha, float beta){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < M && col < N){
        float sum = 0.0f;

        for(int t = 0; t < K; ++t){
            float a = A[row * K + t];
            float b = B[t * N + col];
            sum += a * b;
        }

        int index = row * N + col;
        float oldC = C[index];

        float result = oldC * beta + sum * alpha;
        C[index] = result;       
    }
}

extern "C" void solve(const float* A, const float*B, float* C, int M, int N, int K, float alpha, float beta){
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid(
        (M + 16 - 1) / 16,
        (N + 16 - 1) / 16
    );

    gemm_kernel<<<threadsPerBlock,blocksPerGrid>>>(A, B, C, M, N, K, alpha, beta);

    cudaDeviceSynchronize();
}