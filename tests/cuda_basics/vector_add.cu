#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 统一处理 CUDA 错误，避免主流程到处写 if。
void check_cuda(cudaError_t status)
{
    if (status != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA error: %s\n",
            cudaGetErrorString(status)
        );

        std::exit(EXIT_FAILURE);
    }
}

__global__ void sum_one_warp(
    const float* input,
    float* output)
{
    int lane = threadIdx.x;

    float value = input[lane];

    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(
            0xffffffffu,
            value,
            offset
        );

        value += other;
    }

    if (lane == 0) {
        output[0] = value;
    }
}

// GPU 执行：每个线程负责一个元素。
__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CPU 执行：准备数据、启动计算、取回结果。
int main()
{
    // 1. 准备 CPU 数据。
    const int n = 1003;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = 100.0f;
    }

    // 2. 分配 GPU 内存。
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    check_cuda(cudaMalloc(&d_a, bytes));
    check_cuda(cudaMalloc(&d_b, bytes));
    check_cuda(cudaMalloc(&d_c, bytes));

    // 3. 将两个输入数组复制到 GPU。
    check_cuda(cudaMemcpy(
        d_a,
        h_a.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        d_b,
        h_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 4. 配置线程。
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // 4.1 预热，不计入正式测量。
    for (int repeat = 0; repeat < 10; ++repeat) {
        vector_add<<<blocks, threads>>>(
            d_a,
            d_b,
            d_c,
            n
        );
    }

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    // 4.2 创建两个计时标记。
    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    const int repeats = 1000;

    // 4.3 将开始标记放入默认 stream。
    check_cuda(cudaEventRecord(start));

    // 4.4 连续执行多次，测量平均时间。
    for (int repeat = 0; repeat < repeats; ++repeat) {
        vector_add<<<blocks, threads>>>(
            d_a,
            d_b,
            d_c,
            n
        );
    }

    // 4.5 将结束标记放入同一个 stream。
    check_cuda(cudaEventRecord(stop));

    check_cuda(cudaGetLastError());

    // CPU 等待 GPU 执行到结束标记。
    check_cuda(cudaEventSynchronize(stop));

    // 4.6 读取两次标记之间的时间，单位是毫秒。
    float total_ms = 0.0f;

    check_cuda(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    double average_us = (
        static_cast<double>(total_ms)
        * 1000.0
        / repeats
    );

    std::printf(
        "GPU event interval: %.3f ms for %d launches\n",
        total_ms,
        repeats
    );

    std::printf(
        "Average interval per launch: %.3f us\n",
        average_us
    );

    // 4.7 释放 Event 资源。
    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    // 5. 将计算结果复制回 CPU。
    check_cuda(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 6. 检查所有元素。
    bool passed = true;

    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];

        if (!(std::fabs(h_c[i] - expected) <= 1e-6f)) {
            std::printf(
                "Mismatch at i=%d: got=%f, expected=%f\n",
                i,
                h_c[i],
                expected
            );

            passed = false;
            break;
        }
    }

    std::printf(
        "n=%d, blocks=%d, threads_per_block=%d\n",
        n,
        blocks,
        threads
    );

    std::printf(
        "c[0]=%.1f, c[1]=%.1f, c[%d]=%.1f\n",
        h_c[0],
        h_c[1],
        n - 1,
        h_c[n - 1]
    );

    std::printf(
        "Verification: %s\n",
        passed ? "PASSED" : "FAILED"
    );

    // 7. 释放 GPU 内存。
    check_cuda(cudaFree(d_a));
    check_cuda(cudaFree(d_b));
    check_cuda(cudaFree(d_c));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}