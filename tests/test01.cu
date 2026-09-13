#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int t = 0; t < K; t++) {
            sum += __half2float(A[row * K + t])
                *  __half2float(B[t * N + col]);
        }

        int index = row * N + col;
        float oldC = __half2float(C[index]);

        C[index] = __float2half(
            alpha * sum + beta * oldC
        );
    }
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta){
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + 16 -1 ) / 16,
        (M + 16 -1 ) / 16
    )

    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);

    cudaDeviceSynchronize();
}