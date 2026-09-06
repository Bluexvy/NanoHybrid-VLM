#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>


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


/*
输入：

    q:
        [key_dim]

    k:
        [key_dim]

    v:
        [value_dim]

    state:
        [key_dim, value_dim]

输出：

    output:
        [value_dim]

副作用：

    state 被原地更新。
*/
__global__ void gdn_single_head_decode(
    const float* q,
    const float* k,
    const float* v,
    float g,
    float beta,
    float* state,
    float* output,
    int key_dim,
    int value_dim)
{
    /*
    一个线程负责状态矩阵的一列。

    例如：

        thread 0 -> 第 0 个 V 维度
        thread 1 -> 第 1 个 V 维度
        thread 2 -> 第 2 个 V 维度
    */
    int value_index = (
        blockIdx.x * blockDim.x
        + threadIdx.x
    );

    if (value_index >= value_dim) {
        return;
    }

    /*
    g 是对数衰减值。

        decay = exp(g)

    如果：

        g = log(0.5)

    那么：

        decay = 0.5
    */
    float decay = expf(g);

    /*
    第一步：

        S_decay = exp(g) * S_old

    第二步：

        remembered_value
            = k^T * S_decay

    当前线程只处理一个 V 列，所以沿着
    key_dim 遍历这一列的所有元素。
    */
    float remembered_value = 0.0f;

    for (
        int key_index = 0;
        key_index < key_dim;
        ++key_index
    ) {
        /*
        State 使用行优先连续布局：

            state[key_index][value_index]

        展开为一维地址：

            key_index * value_dim
            + value_index
        */
        int state_offset = (
            key_index * value_dim
            + value_index
        );

        float old_state = state[state_offset];

        float decayed_state = (
            old_state * decay
        );

        /*
        先把衰减后的状态写回原位置。

        因为每个线程负责不同的 value_index，
        所以不同线程不会写同一个状态元素。
        */
        state[state_offset] = decayed_state;

        remembered_value += (
            k[key_index] * decayed_state
        );
    }

    /*
    计算状态预测误差：

        delta = beta * (v - remembered_value)
    */
    float delta = beta * (
        v[value_index] - remembered_value
    );

    /*
    更新状态：

        S_new = S_decay + k * delta^T

    然后用 Q 读取更新后的状态：

        output = q^T * S_new
    */
    float result = 0.0f;

    for (
        int key_index = 0;
        key_index < key_dim;
        ++key_index
    ) {
        int state_offset = (
            key_index * value_dim
            + value_index
        );

        float decayed_state = state[state_offset];

        float updated_state = (
            decayed_state
            + k[key_index] * delta
        );

        /*
        原地写回新的 recurrent state。
        */
        state[state_offset] = updated_state;

        /*
        当前线程计算一个 output[value_index]。
        */
        result += (
            q[key_index] * updated_state
        );
    }

    output[value_index] = result;
}


int main()
{
    /*
    使用前面可以手算的例子：

        key_dim   = 2
        value_dim = 3
    */
    const int key_dim = 2;
    const int value_dim = 3;

    /*
    假设原始 Q 是 [1, 1]。

    L2Norm 后：

        [1/sqrt(2), 1/sqrt(2)]

    再乘 scale = 1/sqrt(Dk) = 1/sqrt(2)：

        q = [0.5, 0.5]

    因此这里的 q 已经包含归一化和 scale。
    */
    std::vector<float> h_q = {
        0.5f,
        0.5f,
    };

    /*
    [1, 0] 的 L2 范数是 1，
    因此归一化后仍然是 [1, 0]。
    */
    std::vector<float> h_k = {
        1.0f,
        0.0f,
    };

    std::vector<float> h_v = {
        3.0f,
        2.0f,
        7.0f,
    };

    /*
    状态矩阵：

        [
            [2,  4,  6],
            [8, 10, 12],
        ]

    实际内存中连续存放为：

        [2, 4, 6, 8, 10, 12]
    */
    std::vector<float> h_state = {
        2.0f,  4.0f,  6.0f,
        8.0f, 10.0f, 12.0f,
    };

    std::vector<float> h_output(
        value_dim,
        0.0f
    );

    /*
    exp(log(0.5)) = 0.5
    */
    const float g = std::log(0.5f);
    const float beta = 0.5f;

    const size_t qk_bytes = (
        key_dim * sizeof(float)
    );

    const size_t value_bytes = (
        value_dim * sizeof(float)
    );

    const size_t state_bytes = (
        key_dim
        * value_dim
        * sizeof(float)
    );

    /*
    GPU 地址。
    */
    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_state = nullptr;
    float* d_output = nullptr;

    /*
    分配 GPU Global Memory。
    */
    check_cuda(cudaMalloc(
        &d_q,
        qk_bytes
    ));

    check_cuda(cudaMalloc(
        &d_k,
        qk_bytes
    ));

    check_cuda(cudaMalloc(
        &d_v,
        value_bytes
    ));

    check_cuda(cudaMalloc(
        &d_state,
        state_bytes
    ));

    check_cuda(cudaMalloc(
        &d_output,
        value_bytes
    ));

    /*
    将输入和旧状态从 CPU 复制到 GPU。
    */
    check_cuda(cudaMemcpy(
        d_q,
        h_q.data(),
        qk_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        d_k,
        h_k.data(),
        qk_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        d_v,
        h_v.data(),
        value_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        d_state,
        h_state.data(),
        state_bytes,
        cudaMemcpyHostToDevice
    ));

    /*
    启动一个 Block，里面有 32 个线程。

    value_dim 只有 3：

        thread 0、1、2 有效；
        其余线程直接 return。
    */
    const int threads = 32;

    const int blocks = (
        value_dim + threads - 1
    ) / threads;

    gdn_single_head_decode<<<
        blocks,
        threads
    >>>(
        d_q,
        d_k,
        d_v,
        g,
        beta,
        d_state,
        d_output,
        key_dim,
        value_dim
    );

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    /*
    取回输出和更新后的状态。
    */
    check_cuda(cudaMemcpy(
        h_output.data(),
        d_output,
        value_bytes,
        cudaMemcpyDeviceToHost
    ));

    check_cuda(cudaMemcpy(
        h_state.data(),
        d_state,
        state_bytes,
        cudaMemcpyDeviceToHost
    ));

    /*
    手算的正确答案。
    */
    const std::vector<float> expected_output = {
        3.0f,
        3.5f,
        5.5f,
    };

    const std::vector<float> expected_state = {
        2.0f, 2.0f, 5.0f,
        4.0f, 5.0f, 6.0f,
    };

    bool passed = true;

    for (
        int value_index = 0;
        value_index < value_dim;
        ++value_index
    ) {
        float error = std::fabs(
            h_output[value_index]
            - expected_output[value_index]
        );

        if (error > 1e-6f) {
            std::printf(
                "Output mismatch at %d: "
                "got=%f expected=%f\n",
                value_index,
                h_output[value_index],
                expected_output[value_index]
            );

            passed = false;
        }
    }

    for (
        int index = 0;
        index < key_dim * value_dim;
        ++index
    ) {
        float error = std::fabs(
            h_state[index]
            - expected_state[index]
        );

        if (error > 1e-6f) {
            std::printf(
                "State mismatch at %d: "
                "got=%f expected=%f\n",
                index,
                h_state[index],
                expected_state[index]
            );

            passed = false;
        }
    }

    std::printf(
        "Output: [%.1f, %.1f, %.1f]\n",
        h_output[0],
        h_output[1],
        h_output[2]
    );

    std::printf("Updated state:\n");

    for (
        int key_index = 0;
        key_index < key_dim;
        ++key_index
    ) {
        std::printf("[");

        for (
            int value_index = 0;
            value_index < value_dim;
            ++value_index
        ) {
            int offset = (
                key_index * value_dim
                + value_index
            );

            std::printf(
                "%.1f%s",
                h_state[offset],
                value_index + 1 == value_dim
                    ? ""
                    : ", "
            );
        }

        std::printf("]\n");
    }

    std::printf(
        "Verification: %s\n",
        passed ? "PASSED" : "FAILED"
    );

    check_cuda(cudaFree(d_q));
    check_cuda(cudaFree(d_k));
    check_cuda(cudaFree(d_v));
    check_cuda(cudaFree(d_state));
    check_cuda(cudaFree(d_output));

    return passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}