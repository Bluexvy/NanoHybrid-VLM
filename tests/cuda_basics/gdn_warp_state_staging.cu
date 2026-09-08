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
__global__ void gdn_warp_state_staging_fp32(
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

    extern __shared__ float shared_state[];

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

    /*
    第一次遍历 State。

    每个线程负责固定的 value_index 列：

        thread 0   -> State[:, 0]
        thread 1   -> State[:, 1]
        ...
        thread 127 -> State[:, 127]

    从 Global Memory 读取旧状态，完成 decay 后，
    把 decayed_state 保存到 Shared Memory。
    */
    for (int key_index = 0; key_index < key_dim; ++key_index) {
        int shared_state_index =
            key_index * value_dim + value_index;

        long long state_offset =
            state_matrix_base + shared_state_index;

        float old_state = state_pool[state_offset];
        float decayed_state = old_state * shared_decay;

        shared_state[shared_state_index] =
            decayed_state;

        remembered_value +=
            shared_key[key_index] * decayed_state;
    }

    /*
    完成全部 Dk 的累加后，才能得到 prediction，
    然后计算当前 Value 列的写入误差。
    */
    float delta =
        shared_beta
        * (v[value_offset] - remembered_value);

    float result = 0.0f;

    /*
    第二次遍历不再读取 Global State。

    直接读取第一次循环保存在 Shared Memory 中的
    decayed_state，然后生成最终 updated_state。
    */
    for (int key_index = 0; key_index < key_dim; ++key_index) {
        int shared_state_index =
            key_index * value_dim + value_index;

        long long state_offset =
            state_matrix_base + shared_state_index;

        float decayed_state =
            shared_state[shared_state_index];

        float updated_state =
            decayed_state
            + shared_key[key_index] * delta;

        state_pool[state_offset] =
            updated_state;

        result +=
            shared_query[key_index] * updated_state;
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
    q、k 是模型产生的原始数据。

    Kernel 和 CPU Reference 都会在内部完成：
        Q L2Norm
        K L2Norm
        Q scale = 1 / sqrt(Dk)
    */
    std::vector<float> h_q = {
        1.0f, 1.0f,
        3.0f, 4.0f,
        -2.0f, 1.0f,
        1.0f, -3.0f,
    };

    std::vector<float> h_k = {
        2.0f, 0.0f,
        3.0f, 4.0f,
        1.0f, -2.0f,
        -4.0f, 3.0f,
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

    int threads = 128;
    int blocks = batch_size * num_heads;

    size_t shared_state_bytes =
        static_cast<size_t>(key_dim)
        * value_dim
        * sizeof(float);

    /*
    允许 Kernel 在真实 Dk=Dv=128 时申请约64 KiB
    动态 Shared Memory。
    */
    check_cuda(cudaFuncSetAttribute(
        gdn_warp_state_staging_fp32,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_state_bytes)
    ));

    gdn_warp_state_staging_fp32<<<
        blocks,
        threads,
        shared_state_bytes
    >>>(
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

    bool passed = max_output_error <= 1e-5f && max_state_error <= 1e-5f;

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