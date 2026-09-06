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

__device__ __forceinline__ float warp_reduce_sum(float value)
{
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);

    return value;
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
    constexpr int block_size = 128;
    constexpr int warp_size = 32;
    constexpr int num_warps = block_size / warp_size;

    int batch_head_index = blockIdx.x;
    int batch_index = batch_head_index / num_heads;
    int head_index = batch_head_index % num_heads;

    int thread_index = threadIdx.x;
    int value_index = thread_index;

    int lane_index = thread_index % warp_size;
    int warp_index = thread_index / warp_size;

    __shared__ float shared_query[block_size];
    __shared__ float shared_key[block_size];

    __shared__ float shared_query_warp_sums[num_warps];
    __shared__ float shared_key_warp_sums[num_warps];

    __shared__ float shared_query_multiplier;
    __shared__ float shared_key_multiplier;
    __shared__ float shared_decay;
    __shared__ float shared_beta;

    long long qk_base = static_cast<long long>(batch_index * num_heads + head_index) * key_dim;
    int gate_offset = batch_index * num_heads + head_index;

    float query_value = 0.0f;
    float key_value = 0.0f;

    if (thread_index < key_dim) {
        query_value = q[qk_base + thread_index];
        key_value = k[qk_base + thread_index];
    }

    shared_query[thread_index] = query_value;
    shared_key[thread_index] = key_value;

    float query_squared_sum = query_value * query_value;
    float key_squared_sum = key_value * key_value;

    query_squared_sum = warp_reduce_sum(query_squared_sum);
    key_squared_sum = warp_reduce_sum(key_squared_sum);

    /*
    每个 Warp 的 lane 0 保存本 Warp 的局部和。

    block_size = 128，因此共有4个 Warp，
    Shared Memory 中最终只有4个局部结果。
    */
    if (lane_index == 0) {
        shared_query_warp_sums[warp_index] = query_squared_sum;
        shared_key_warp_sums[warp_index] = key_squared_sum;
    }

    /*
    保证四个 Warp 都已经写完自己的局部和，
    Warp 0 才能读取这四个值。
    */
    __syncthreads();

    /*
    只有 Warp 0 执行第二级归约。

    lane 0～3 分别读取四个 Warp 的结果；
    lane 4～31 使用0，不影响最终求和。
    */
    if (warp_index == 0) {
        float query_block_sum =
            lane_index < num_warps
            ? shared_query_warp_sums[lane_index]
            : 0.0f;

        float key_block_sum =
            lane_index < num_warps
            ? shared_key_warp_sums[lane_index]
            : 0.0f;

        query_block_sum = warp_reduce_sum(query_block_sum);
        key_block_sum = warp_reduce_sum(key_block_sum);

        /*
        Warp 0 的 lane 0 得到整个 Block 的最终和。
        */
        if (lane_index == 0) {
            float query_inverse_norm = 1.0f / sqrtf(query_block_sum + 1e-6f);
            float key_inverse_norm = 1.0f / sqrtf(key_block_sum + 1e-6f);
            float query_scale = 1.0f / sqrtf(static_cast<float>(key_dim));

            shared_query_multiplier = query_inverse_norm * query_scale;
            shared_key_multiplier = key_inverse_norm;

            shared_decay = expf(g[gate_offset]);
            shared_beta = beta[gate_offset];
        }
    }

    /*
    所有线程必须等待 thread 0 写完：
        shared_query_multiplier
        shared_key_multiplier
        shared_decay
        shared_beta
    */
    __syncthreads();

    if (thread_index < key_dim) {
        shared_query[thread_index] *= shared_query_multiplier;
        shared_key[thread_index] *= shared_key_multiplier;
    }

    /*
    等待所有 Q/K 元素归一化完成，
    后面每个 Value 线程才可以读取完整 Q/K。
    */
    __syncthreads();

    /*
    所有需要参与 __syncthreads() 的代码已经结束，
    无效 Value 线程现在才可以退出。
    */
    if (value_index >= value_dim) {
        return;
    }

    int state_slot = state_slot_ids[batch_index];

    long long state_matrix_base =
        static_cast<long long>(state_slot) * num_layers * num_heads * key_dim * value_dim
        + static_cast<long long>(gdn_index) * num_heads * key_dim * value_dim
        + static_cast<long long>(head_index) * key_dim * value_dim;

    long long value_offset =
        static_cast<long long>(batch_index * num_heads + head_index) * value_dim
        + value_index;

    float remembered_value = 0.0f;

    for (int key_index = 0; key_index < key_dim; ++key_index) {
        long long state_offset =
            state_matrix_base
            + static_cast<long long>(key_index) * value_dim
            + value_index;

        float old_state = state_pool[state_offset];
        float decayed_state = old_state * shared_decay;

        remembered_value += shared_key[key_index] * decayed_state;
    }

    float delta = shared_beta * (v[value_offset] - remembered_value);
    float result = 0.0f;

    for (int key_index = 0; key_index < key_dim; ++key_index) {
        long long state_offset =
            state_matrix_base
            + static_cast<long long>(key_index) * value_dim
            + value_index;

        float old_state = state_pool[state_offset];
        float decayed_state = old_state * shared_decay;
        float updated_state = decayed_state + shared_key[key_index] * delta;

        state_pool[state_offset] = updated_state;
        result += shared_query[key_index] * updated_state;
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
    const char* kernel_name = "warp_reduction";
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
