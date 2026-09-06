#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>


void check_cuda(cudaError_t status)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}


/*
输入形状：

    q:              [B, H, Dk]
    k:              [B, H, Dk]
    v:              [B, H, Dv]
    g:              [B, H]
    beta:           [B, H]
    state_slot_ids: [B]

状态池：

    state_pool:
        [num_slots, num_layers, H, Dk, Dv]

输出：

    output:
        [B, H, Dv]

线程分工：

    一个线程处理：
        一个 batch
        × 一个 head
        × 一个 value_index

副作用：

    直接更新 state_pool 中相应 slot 的状态。
*/
__global__ void gdn_state_pool_decode_fp32(
    const float* q,
    const float* k,
    const float* v,
    const float* g,
    const float* beta,
    const int* state_slot_ids,
    float* state_pool,
    float* output,
    int batch_size,
    int num_layers,
    int num_heads,
    int key_dim,
    int value_dim,
    int gdn_index)
{
    int work_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = batch_size * num_heads * value_dim;

    if (work_index >= total_work) {
        return;
    }

    /*
    将一维 work_index 拆成三维逻辑坐标：

        batch_index
        head_index
        value_index
    */
    int value_index = work_index % value_dim;
    int remaining = work_index / value_dim;
    int head_index = remaining % num_heads;
    int batch_index = remaining / num_heads;

    /*
    batch_index 只是本轮 batch 中的位置。

    真正的状态槽位要通过 state_slot_ids 查找。
    */
    int state_slot = state_slot_ids[batch_index];

    /*
    q、k 的逻辑形状是 [B, H, Dk]。

    当前 batch 和 head 的 Q/K 起始地址：

        (batch_index * H + head_index) * Dk
    */
    long long qk_base =
        static_cast<long long>(batch_index * num_heads + head_index)
        * key_dim;

    /*
    v、output 的逻辑形状是 [B, H, Dv]。

    当前元素的一维地址：

        (batch_index * H + head_index) * Dv
        + value_index
    */
    long long value_offset =
        static_cast<long long>(batch_index * num_heads + head_index)
        * value_dim
        + value_index;

    /*
    g、beta 的逻辑形状是 [B, H]。
    */
    int gate_offset = batch_index * num_heads + head_index;

    /*
    state_pool 的逻辑形状：

        [num_slots, num_layers, H, Dk, Dv]

    先定位到：

        state_pool[
            state_slot,
            gdn_index,
            head_index,
            0,
            0
        ]

    也就是当前状态矩阵的起始地址。
    */
    long long state_matrix_base =
        static_cast<long long>(state_slot) * num_layers * num_heads * key_dim * value_dim
        + static_cast<long long>(gdn_index) * num_heads * key_dim * value_dim
        + static_cast<long long>(head_index) * key_dim * value_dim;

    float decay = expf(g[gate_offset]);
    float remembered_value = 0.0f;

    /*
    第一次沿 Dk 遍历：

        S_decay = exp(g) * S_old
        remembered = k^T * S_decay
    */
    for (int key_index = 0; key_index < key_dim; ++key_index) {
        long long state_offset =
            state_matrix_base
            + static_cast<long long>(key_index) * value_dim
            + value_index;

        float old_state = state_pool[state_offset];
        float decayed_state = old_state * decay;

        state_pool[state_offset] = decayed_state;
        remembered_value += k[qk_base + key_index] * decayed_state;
    }

    /*
    delta = beta * (v - remembered)
    */
    float delta = beta[gate_offset] * (v[value_offset] - remembered_value);
    float result = 0.0f;

    /*
    第二次沿 Dk 遍历：

        S_new = S_decay + k * delta^T
        output = q^T * S_new
    */
    for (int key_index = 0; key_index < key_dim; ++key_index) {
        long long state_offset =
            state_matrix_base
            + static_cast<long long>(key_index) * value_dim
            + value_index;

        float decayed_state = state_pool[state_offset];
        float updated_state = decayed_state + k[qk_base + key_index] * delta;

        state_pool[state_offset] = updated_state;
        result += q[qk_base + key_index] * updated_state;
    }

    output[value_offset] = result;
}


/*
CPU Reference 使用相同的数学过程。

它的作用不是加速，而是提供正确答案，
验证 CUDA Kernel 的状态更新和输出。
*/
void gdn_state_pool_reference(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& g,
    const std::vector<float>& beta,
    const std::vector<int>& state_slot_ids,
    std::vector<float>& state_pool,
    std::vector<float>& output,
    int batch_size,
    int num_layers,
    int num_heads,
    int key_dim,
    int value_dim,
    int gdn_index)
{
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        int state_slot = state_slot_ids[batch_index];

        for (int head_index = 0; head_index < num_heads; ++head_index) {
            long long qk_base =
                static_cast<long long>(batch_index * num_heads + head_index)
                * key_dim;

            int gate_offset = batch_index * num_heads + head_index;

            long long state_matrix_base =
                static_cast<long long>(state_slot) * num_layers * num_heads * key_dim * value_dim
                + static_cast<long long>(gdn_index) * num_heads * key_dim * value_dim
                + static_cast<long long>(head_index) * key_dim * value_dim;

            float decay = std::exp(g[gate_offset]);

            for (int value_index = 0; value_index < value_dim; ++value_index) {
                long long value_offset =
                    static_cast<long long>(batch_index * num_heads + head_index)
                    * value_dim
                    + value_index;

                float remembered_value = 0.0f;

                for (int key_index = 0; key_index < key_dim; ++key_index) {
                    long long state_offset =
                        state_matrix_base
                        + static_cast<long long>(key_index) * value_dim
                        + value_index;

                    float decayed_state = state_pool[state_offset] * decay;

                    state_pool[state_offset] = decayed_state;
                    remembered_value += k[qk_base + key_index] * decayed_state;
                }

                float delta = beta[gate_offset] * (v[value_offset] - remembered_value);
                float result = 0.0f;

                for (int key_index = 0; key_index < key_dim; ++key_index) {
                    long long state_offset =
                        state_matrix_base
                        + static_cast<long long>(key_index) * value_dim
                        + value_index;

                    float updated_state =
                        state_pool[state_offset]
                        + k[qk_base + key_index] * delta;

                    state_pool[state_offset] = updated_state;
                    result += q[qk_base + key_index] * updated_state;
                }

                output[value_offset] = result;
            }
        }
    }
}


int main()
{
    /*
    小尺寸验证：

        B = 2
        num_slots = 4
        num_layers = 2
        H = 2
        Dk = 2
        Dv = 3

    本次只更新 gdn_index = 1。
    */
    const int batch_size = 2;
    const int num_slots = 4;
    const int num_layers = 2;
    const int num_heads = 2;
    const int key_dim = 2;
    const int value_dim = 3;
    const int gdn_index = 1;

    /*
    batch row 0 使用物理 slot 3。
    batch row 1 使用物理 slot 1。

    因此 slot 0 和 slot 2 不应被修改。
    */
    std::vector<int> h_state_slot_ids = {3, 1};

    /*
    q、k 已经归一化。

    每个 batch、每个 head 都使用：

        q = [0.5, 0.5]
        k = [1.0, 0.0]
    */
    std::vector<float> h_q = {
        0.5f, 0.5f,
        0.5f, 0.5f,
        0.5f, 0.5f,
        0.5f, 0.5f,
    };

    std::vector<float> h_k = {
        1.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 0.0f,
    };

    /*
    v 的逻辑形状是 [2, 2, 3]。
    */
    std::vector<float> h_v = {
        3.0f, 2.0f, 7.0f,
        4.0f, 6.0f, 8.0f,

        5.0f, 1.0f, 9.0f,
        2.0f, 8.0f, 4.0f,
    };

    std::vector<float> h_g(
        batch_size * num_heads,
        std::log(0.5f)
    );

    std::vector<float> h_beta(
        batch_size * num_heads,
        0.5f
    );

    int state_element_count =
        num_slots * num_layers * num_heads * key_dim * value_dim;

    std::vector<float> h_initial_state(state_element_count);

    /*
    为状态池填入可区分的数据。

    这样如果地址公式写错，通常会明显影响结果。
    */
    for (int index = 0; index < state_element_count; ++index) {
        h_initial_state[index] = 0.01f * static_cast<float>(index + 1);
    }

    std::vector<float> h_cuda_state = h_initial_state;
    std::vector<float> h_reference_state = h_initial_state;

    int output_element_count = batch_size * num_heads * value_dim;

    std::vector<float> h_cuda_output(output_element_count, 0.0f);
    std::vector<float> h_reference_output(output_element_count, 0.0f);

    /*
    先在 CPU 上计算正确答案。
    */
    gdn_state_pool_reference(
        h_q,
        h_k,
        h_v,
        h_g,
        h_beta,
        h_state_slot_ids,
        h_reference_state,
        h_reference_output,
        batch_size,
        num_layers,
        num_heads,
        key_dim,
        value_dim,
        gdn_index
    );

    size_t qk_bytes = h_q.size() * sizeof(float);
    size_t value_bytes = h_v.size() * sizeof(float);
    size_t gate_bytes = h_g.size() * sizeof(float);
    size_t slot_bytes = h_state_slot_ids.size() * sizeof(int);
    size_t state_bytes = h_cuda_state.size() * sizeof(float);
    size_t output_bytes = h_cuda_output.size() * sizeof(float);

    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_g = nullptr;
    float* d_beta = nullptr;
    float* d_state_pool = nullptr;
    float* d_output = nullptr;
    int* d_state_slot_ids = nullptr;

    check_cuda(cudaMalloc(&d_q, qk_bytes));
    check_cuda(cudaMalloc(&d_k, qk_bytes));
    check_cuda(cudaMalloc(&d_v, value_bytes));
    check_cuda(cudaMalloc(&d_g, gate_bytes));
    check_cuda(cudaMalloc(&d_beta, gate_bytes));
    check_cuda(cudaMalloc(&d_state_slot_ids, slot_bytes));
    check_cuda(cudaMalloc(&d_state_pool, state_bytes));
    check_cuda(cudaMalloc(&d_output, output_bytes));

    check_cuda(cudaMemcpy(d_q, h_q.data(), qk_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_k, h_k.data(), qk_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_v, h_v.data(), value_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_g, h_g.data(), gate_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_beta, h_beta.data(), gate_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_state_slot_ids, h_state_slot_ids.data(), slot_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_state_pool, h_cuda_state.data(), state_bytes, cudaMemcpyHostToDevice));

    int total_work = batch_size * num_heads * value_dim;
    int threads = 128;
    int blocks = (total_work + threads - 1) / threads;

    gdn_state_pool_decode_fp32<<<blocks, threads>>>(
        d_q,
        d_k,
        d_v,
        d_g,
        d_beta,
        d_state_slot_ids,
        d_state_pool,
        d_output,
        batch_size,
        num_layers,
        num_heads,
        key_dim,
        value_dim,
        gdn_index
    );

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    check_cuda(cudaMemcpy(h_cuda_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(h_cuda_state.data(), d_state_pool, state_bytes, cudaMemcpyDeviceToHost));

    float max_output_error = 0.0f;
    float max_state_error = 0.0f;

    for (int index = 0; index < output_element_count; ++index) {
        float error = std::fabs(h_cuda_output[index] - h_reference_output[index]);
        max_output_error = std::max(max_output_error, error);
    }

    for (int index = 0; index < state_element_count; ++index) {
        float error = std::fabs(h_cuda_state[index] - h_reference_state[index]);
        max_state_error = std::max(max_state_error, error);
    }

    bool passed = max_output_error <= 1e-6f && max_state_error <= 1e-6f;

    std::printf("Batch to slot mapping:\n");

    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        std::printf(
            "batch %d -> state slot %d\n",
            batch_index,
            h_state_slot_ids[batch_index]
        );
    }

    std::printf("\nCUDA output:\n");

    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        for (int head_index = 0; head_index < num_heads; ++head_index) {
            std::printf("batch=%d head=%d: [", batch_index, head_index);

            for (int value_index = 0; value_index < value_dim; ++value_index) {
                int output_offset =
                    (batch_index * num_heads + head_index)
                    * value_dim
                    + value_index;

                const char* separator = value_index + 1 == value_dim ? "" : ", ";
                std::printf("%.6f%s", h_cuda_output[output_offset], separator);
            }

            std::printf("]\n");
        }
    }

    std::printf("\nMax output error: %.9f\n", max_output_error);
    std::printf("Max state error:  %.9f\n", max_state_error);
    std::printf("Verification: %s\n", passed ? "PASSED" : "FAILED");

    check_cuda(cudaFree(d_q));
    check_cuda(cudaFree(d_k));
    check_cuda(cudaFree(d_v));
    check_cuda(cudaFree(d_g));
    check_cuda(cudaFree(d_beta));
    check_cuda(cudaFree(d_state_slot_ids));
    check_cuda(cudaFree(d_state_pool));
    check_cuda(cudaFree(d_output));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}