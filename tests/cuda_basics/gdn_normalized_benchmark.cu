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

    float query_squared_sum = 0.0f;
    float key_squared_sum = 0.0f;

    for (int key_index = 0; key_index < key_dim; ++key_index) {
        float query_value = q[qk_base + key_index];
        float key_value = k[qk_base + key_index];

        query_squared_sum += query_value * query_value;
        key_squared_sum += key_value * key_value;
    }

    const float epsilon = 1e-6f;

    float query_inverse_norm = 1.0f / sqrtf(query_squared_sum + epsilon);
    float key_inverse_norm = 1.0f / sqrtf(key_squared_sum + epsilon);
    float query_scale = 1.0f / sqrtf(static_cast<float>(key_dim));

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
        long long state_offset = state_matrix_base + static_cast<long long>(key_index) * value_dim + value_index;

        float old_state = state_pool[state_offset];
        float decayed_state = old_state * decay;
        float normalized_key = k[qk_base + key_index] * key_inverse_norm;

        state_pool[state_offset] = decayed_state;
        remembered_value += normalized_key * decayed_state;
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
        long long state_offset = state_matrix_base + static_cast<long long>(key_index) * value_dim + value_index;

        float decayed_state = state_pool[state_offset];
        float normalized_key = k[qk_base + key_index] * key_inverse_norm;
        float normalized_query = q[qk_base + key_index] * query_inverse_norm * query_scale;
        float updated_state = decayed_state + normalized_key * delta;

        state_pool[state_offset] = updated_state;
        result += normalized_query * updated_state;
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
            float query_squared_sum = 0.0f;
            float key_squared_sum = 0.0f;

            for (int key_index = 0; key_index < key_dim; ++key_index) {
                float query_value = q[qk_base + key_index];
                float key_value = k[qk_base + key_index];

                query_squared_sum += query_value * query_value;
                key_squared_sum += key_value * key_value;
            }

            const float epsilon = 1e-6f;

            float query_inverse_norm = 1.0f / std::sqrt(query_squared_sum + epsilon);
            float key_inverse_norm = 1.0f / std::sqrt(key_squared_sum + epsilon);
            float query_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));

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
                    float normalized_key = k[qk_base + key_index] * key_inverse_norm;
                    remembered_value += normalized_key * decayed_state;
                }

                float delta = beta[gate_offset] * (v[value_offset] - remembered_value);
                float result = 0.0f;

                for (int key_index = 0; key_index < key_dim; ++key_index) {
                    long long state_offset = state_matrix_base + static_cast<long long>(key_index) * value_dim + value_index;

                    float normalized_key = k[qk_base + key_index] * key_inverse_norm;
                    float normalized_query = q[qk_base + key_index] * query_inverse_norm * query_scale;
                    float updated_state = state_pool[state_offset] + normalized_key * delta;

                    state_pool[state_offset] = updated_state;
                    result += normalized_query * updated_state;
                }

                output[value_offset] = result;
            }
        }
    }
}


int main()
{
    const int max_batch_size = 16;
    const int num_slots = 16;
    const int num_layers = 24;
    const int num_heads = 32;
    const int key_dim = 128;
    const int value_dim = 128;
    const int warmup_iterations = 20;
    const int benchmark_iterations = 200;

    size_t qk_element_count =
        static_cast<size_t>(max_batch_size)
        * num_heads
        * key_dim;

    size_t value_element_count =
        static_cast<size_t>(max_batch_size)
        * num_heads
        * value_dim;

    size_t gate_element_count =
        static_cast<size_t>(max_batch_size)
        * num_heads;

    size_t state_element_count =
        static_cast<size_t>(num_slots)
        * num_layers
        * num_heads
        * key_dim
        * value_dim;

    std::vector<float> h_q(qk_element_count);
    std::vector<float> h_k(qk_element_count);
    std::vector<float> h_v(value_element_count);
    std::vector<float> h_g(gate_element_count);
    std::vector<float> h_beta(gate_element_count);
    std::vector<int> h_state_slot_ids(max_batch_size);

    for (size_t index = 0; index < qk_element_count; ++index) {
        h_q[index] = 0.1f + std::sin(static_cast<float>(index) * 0.001f);
        h_k[index] = 0.2f + std::cos(static_cast<float>(index) * 0.0013f);
    }

    for (size_t index = 0; index < value_element_count; ++index) {
        h_v[index] = 0.3f + std::sin(static_cast<float>(index) * 0.0007f);
    }

    for (size_t index = 0; index < gate_element_count; ++index) {
        h_g[index] = -0.1f;
        h_beta[index] = 0.5f;
    }

    for (int batch_index = 0; batch_index < max_batch_size; ++batch_index) {
        h_state_slot_ids[batch_index] = batch_index;
    }

    size_t qk_bytes = qk_element_count * sizeof(float);
    size_t value_bytes = value_element_count * sizeof(float);
    size_t gate_bytes = gate_element_count * sizeof(float);
    size_t slot_bytes = h_state_slot_ids.size() * sizeof(int);
    size_t state_bytes = state_element_count * sizeof(float);

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
    check_cuda(cudaMalloc(&d_output, value_bytes));

    check_cuda(cudaMemcpy(d_q, h_q.data(), qk_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_k, h_k.data(), qk_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_v, h_v.data(), value_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_g, h_g.data(), gate_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_beta, h_beta.data(), gate_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_state_slot_ids, h_state_slot_ids.data(), slot_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    check_cuda(cudaEventCreate(&start_event));
    check_cuda(cudaEventCreate(&stop_event));

#ifdef BLOCK_OPTIMIZED
    const char* kernel_name = "block_optimized";
#else
    const char* kernel_name = "normalized_baseline";
#endif

    double state_pool_mib =
        static_cast<double>(state_bytes)
        / 1024.0
        / 1024.0;

    std::printf("Kernel: %s\n", kernel_name);
    std::printf("State Pool: %.2f MiB\n", state_pool_mib);
    std::printf(
        "Shape: H=%d, Dk=%d, Dv=%d, layers=%d\n\n",
        num_heads,
        key_dim,
        value_dim,
        num_layers
    );

    const int batch_sizes[] = {1, 2, 4, 8, 16};
    const int num_batch_cases = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    for (int case_index = 0; case_index < num_batch_cases; ++case_index) {
        int batch_size = batch_sizes[case_index];
        int threads = 128;
        int blocks = 0;

#ifdef BLOCK_OPTIMIZED
        blocks = batch_size * num_heads;
#else
        int total_work = batch_size * num_heads * value_dim;
        blocks = (total_work + threads - 1) / threads;
#endif

        check_cuda(cudaMemset(d_state_pool, 0, state_bytes));
        check_cuda(cudaMemset(d_output, 0, value_bytes));

        for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
            for (int layer_index = 0; layer_index < num_layers; ++layer_index) {
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
                    layer_index
                );
            }
        }

        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());

        /*
        Warmup 会修改 recurrent state。

        计时前重新清零，让两个 Kernel 都从相同的
        初始状态开始执行 benchmark。
        */
        check_cuda(cudaMemset(d_state_pool, 0, state_bytes));
        check_cuda(cudaMemset(d_output, 0, value_bytes));
        check_cuda(cudaDeviceSynchronize());

        check_cuda(cudaEventRecord(start_event));

        for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
            for (int layer_index = 0; layer_index < num_layers; ++layer_index) {
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
                    layer_index
                );
            }
        }

        check_cuda(cudaEventRecord(stop_event));
        check_cuda(cudaEventSynchronize(stop_event));
        check_cuda(cudaGetLastError());

        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

        float recurrent_step_us = elapsed_ms * 1000.0f / static_cast<float>(benchmark_iterations);
        float average_layer_us = recurrent_step_us / static_cast<float>(num_layers);
        double recurrent_tokens_per_second = static_cast<double>(batch_size) * 1000000.0 / static_cast<double>(recurrent_step_us);

        size_t active_output_elements =
            static_cast<size_t>(batch_size)
            * num_heads
            * value_dim;

        std::vector<float> h_output(active_output_elements);

        check_cuda(cudaMemcpy(
            h_output.data(),
            d_output,
            active_output_elements * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        double output_checksum = 0.0;

        for (float output_value : h_output) {
            output_checksum += static_cast<double>(output_value);
        }

        std::printf(
            "B=%2d | blocks=%4d | layer=%8.3f us | 24-layer step=%9.3f us | recurrent tok/s=%10.2f | checksum=% .6e\n",
            batch_size,
            blocks,
            average_layer_us,
            recurrent_step_us,
            recurrent_tokens_per_second,
            output_checksum
        );
    }

    check_cuda(cudaEventDestroy(start_event));
    check_cuda(cudaEventDestroy(stop_event));

    check_cuda(cudaFree(d_q));
    check_cuda(cudaFree(d_k));
    check_cuda(cudaFree(d_v));
    check_cuda(cudaFree(d_g));
    check_cuda(cudaFree(d_beta));
    check_cuda(cudaFree(d_state_slot_ids));
    check_cuda(cudaFree(d_state_pool));
    check_cuda(cudaFree(d_output));

    return EXIT_SUCCESS;
}
