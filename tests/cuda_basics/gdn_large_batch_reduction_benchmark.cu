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


__device__ __forceinline__ void update_state_column(
    const float* v,
    const float* shared_query,
    const float* shared_key,
    float decay,
    float beta_value,
    float* state_pool,
    float* output,
    long long state_matrix_base,
    long long value_offset,
    int key_dim,
    int value_dim,
    int value_index)
{
    float remembered_value = 0.0f;

    for (int key_index = 0; key_index < key_dim; ++key_index) {
        long long state_offset = state_matrix_base + static_cast<long long>(key_index) * value_dim + value_index;
        float decayed_state = state_pool[state_offset] * decay;
        remembered_value += shared_key[key_index] * decayed_state;
    }

    float delta = beta_value * (v[value_offset] - remembered_value);
    float result = 0.0f;

    for (int key_index = 0; key_index < key_dim; ++key_index) {
        long long state_offset = state_matrix_base + static_cast<long long>(key_index) * value_dim + value_index;
        float decayed_state = state_pool[state_offset] * decay;
        float updated_state = decayed_state + shared_key[key_index] * delta;
        state_pool[state_offset] = updated_state;
        result += shared_query[key_index] * updated_state;
    }

    output[value_offset] = result;
}


// Shared Memory tree reduction: 128 -> 64 -> 32 -> ... -> 1.
__global__ void gdn_block_reduction_fp32(
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

    int batch_head_index = blockIdx.x;
    int batch_index = batch_head_index / num_heads;
    int head_index = batch_head_index % num_heads;
    int thread_index = threadIdx.x;
    int value_index = thread_index;

    if (batch_index >= batch_size) {
        return;
    }

    __shared__ float shared_query[block_size];
    __shared__ float shared_key[block_size];
    __shared__ float shared_query_squared[block_size];
    __shared__ float shared_key_squared[block_size];
    __shared__ float shared_decay;
    __shared__ float shared_beta;

    long long qk_base = static_cast<long long>(batch_index * num_heads + head_index) * key_dim;
    int gate_offset = batch_index * num_heads + head_index;

    float query_value = thread_index < key_dim ? q[qk_base + thread_index] : 0.0f;
    float key_value = thread_index < key_dim ? k[qk_base + thread_index] : 0.0f;

    shared_query[thread_index] = query_value;
    shared_key[thread_index] = key_value;
    shared_query_squared[thread_index] = query_value * query_value;
    shared_key_squared[thread_index] = key_value * key_value;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (thread_index < stride) {
            shared_query_squared[thread_index] += shared_query_squared[thread_index + stride];
            shared_key_squared[thread_index] += shared_key_squared[thread_index + stride];
        }
        __syncthreads();
    }

    float query_inverse_norm = rsqrtf(shared_query_squared[0] + 1e-6f);
    float key_inverse_norm = rsqrtf(shared_key_squared[0] + 1e-6f);
    float query_scale = rsqrtf(static_cast<float>(key_dim));

    if (thread_index < key_dim) {
        shared_query[thread_index] *= query_inverse_norm * query_scale;
        shared_key[thread_index] *= key_inverse_norm;
    }

    if (thread_index == 0) {
        shared_decay = expf(g[gate_offset]);
        shared_beta = beta[gate_offset];
    }
    __syncthreads();

    if (value_index >= value_dim) {
        return;
    }

    int state_slot = state_slot_ids[batch_index];
    long long state_matrix_base =
        static_cast<long long>(state_slot) * num_layers * num_heads * key_dim * value_dim
        + static_cast<long long>(gdn_index) * num_heads * key_dim * value_dim
        + static_cast<long long>(head_index) * key_dim * value_dim;
    long long value_offset = static_cast<long long>(batch_index * num_heads + head_index) * value_dim + value_index;

    update_state_column(
        v,
        shared_query,
        shared_key,
        shared_decay,
        shared_beta,
        state_pool,
        output,
        state_matrix_base,
        value_offset,
        key_dim,
        value_dim,
        value_index
    );
}


// Hybrid reduction: Shared Memory 128 -> 64 -> 32, then Warp Shuffle 32 -> 1.
__global__ void gdn_hybrid_reduction_fp32(
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

    int batch_head_index = blockIdx.x;
    int batch_index = batch_head_index / num_heads;
    int head_index = batch_head_index % num_heads;
    int thread_index = threadIdx.x;
    int value_index = thread_index;

    if (batch_index >= batch_size) {
        return;
    }

    __shared__ float shared_query[block_size];
    __shared__ float shared_key[block_size];
    __shared__ float shared_query_squared[block_size];
    __shared__ float shared_key_squared[block_size];
    __shared__ float shared_decay;
    __shared__ float shared_beta;

    long long qk_base = static_cast<long long>(batch_index * num_heads + head_index) * key_dim;
    int gate_offset = batch_index * num_heads + head_index;

    float query_value = thread_index < key_dim ? q[qk_base + thread_index] : 0.0f;
    float key_value = thread_index < key_dim ? k[qk_base + thread_index] : 0.0f;

    shared_query[thread_index] = query_value;
    shared_key[thread_index] = key_value;
    shared_query_squared[thread_index] = query_value * query_value;
    shared_key_squared[thread_index] = key_value * key_value;
    __syncthreads();

    if (thread_index < 64) {
        shared_query_squared[thread_index] += shared_query_squared[thread_index + 64];
        shared_key_squared[thread_index] += shared_key_squared[thread_index + 64];
    }
    __syncthreads();

    if (thread_index < 32) {
        float query_sum = shared_query_squared[thread_index] + shared_query_squared[thread_index + 32];
        float key_sum = shared_key_squared[thread_index] + shared_key_squared[thread_index + 32];
        query_sum = warp_reduce_sum(query_sum);
        key_sum = warp_reduce_sum(key_sum);

        if (thread_index == 0) {
            shared_query_squared[0] = rsqrtf(query_sum + 1e-6f) * rsqrtf(static_cast<float>(key_dim));
            shared_key_squared[0] = rsqrtf(key_sum + 1e-6f);
            shared_decay = expf(g[gate_offset]);
            shared_beta = beta[gate_offset];
        }
    }
    __syncthreads();

    if (thread_index < key_dim) {
        shared_query[thread_index] *= shared_query_squared[0];
        shared_key[thread_index] *= shared_key_squared[0];
    }
    __syncthreads();

    if (value_index >= value_dim) {
        return;
    }

    int state_slot = state_slot_ids[batch_index];
    long long state_matrix_base =
        static_cast<long long>(state_slot) * num_layers * num_heads * key_dim * value_dim
        + static_cast<long long>(gdn_index) * num_heads * key_dim * value_dim
        + static_cast<long long>(head_index) * key_dim * value_dim;
    long long value_offset = static_cast<long long>(batch_index * num_heads + head_index) * value_dim + value_index;

    update_state_column(
        v,
        shared_query,
        shared_key,
        shared_decay,
        shared_beta,
        state_pool,
        output,
        state_matrix_base,
        value_offset,
        key_dim,
        value_dim,
        value_index
    );
}


// Full Warp Shuffle: four Warps reduce locally, then Warp 0 reduces four partial sums.
__global__ void gdn_full_warp_shuffle_fp32(
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

    if (batch_index >= batch_size) {
        return;
    }

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

    float query_value = thread_index < key_dim ? q[qk_base + thread_index] : 0.0f;
    float key_value = thread_index < key_dim ? k[qk_base + thread_index] : 0.0f;

    shared_query[thread_index] = query_value;
    shared_key[thread_index] = key_value;

    float query_squared_sum = warp_reduce_sum(query_value * query_value);
    float key_squared_sum = warp_reduce_sum(key_value * key_value);

    if (lane_index == 0) {
        shared_query_warp_sums[warp_index] = query_squared_sum;
        shared_key_warp_sums[warp_index] = key_squared_sum;
    }
    __syncthreads();

    if (warp_index == 0) {
        float query_block_sum = lane_index < num_warps ? shared_query_warp_sums[lane_index] : 0.0f;
        float key_block_sum = lane_index < num_warps ? shared_key_warp_sums[lane_index] : 0.0f;
        query_block_sum = warp_reduce_sum(query_block_sum);
        key_block_sum = warp_reduce_sum(key_block_sum);

        if (lane_index == 0) {
            shared_query_multiplier = rsqrtf(query_block_sum + 1e-6f) * rsqrtf(static_cast<float>(key_dim));
            shared_key_multiplier = rsqrtf(key_block_sum + 1e-6f);
            shared_decay = expf(g[gate_offset]);
            shared_beta = beta[gate_offset];
        }
    }
    __syncthreads();

    if (thread_index < key_dim) {
        shared_query[thread_index] *= shared_query_multiplier;
        shared_key[thread_index] *= shared_key_multiplier;
    }
    __syncthreads();

    if (value_index >= value_dim) {
        return;
    }

    int state_slot = state_slot_ids[batch_index];
    long long state_matrix_base =
        static_cast<long long>(state_slot) * num_layers * num_heads * key_dim * value_dim
        + static_cast<long long>(gdn_index) * num_heads * key_dim * value_dim
        + static_cast<long long>(head_index) * key_dim * value_dim;
    long long value_offset = static_cast<long long>(batch_index * num_heads + head_index) * value_dim + value_index;

    update_state_column(
        v,
        shared_query,
        shared_key,
        shared_decay,
        shared_beta,
        state_pool,
        output,
        state_matrix_base,
        value_offset,
        key_dim,
        value_dim,
        value_index
    );
}


struct KernelArguments {
    const float* q;
    const float* k;
    const float* v;
    const float* g;
    const float* beta;
    const int* state_slot_ids;
    float* state_pool;
    float* output;
    int num_layers;
    int num_heads;
    int key_dim;
    int value_dim;
};


enum class KernelKind {
    BlockReduction,
    HybridReduction,
    FullWarpShuffle
};


void launch_kernel(KernelKind kernel_kind, const KernelArguments& arguments, int batch_size)
{
    int blocks = batch_size * arguments.num_heads;
    int threads = 128;

    if (kernel_kind == KernelKind::HybridReduction) {
        gdn_hybrid_reduction_fp32<<<blocks, threads>>>(
            arguments.q,
            arguments.k,
            arguments.v,
            arguments.g,
            arguments.beta,
            arguments.state_slot_ids,
            arguments.state_pool,
            arguments.output,
            batch_size,
            arguments.num_layers,
            arguments.num_heads,
            arguments.key_dim,
            arguments.value_dim,
            0
        );
    } else if (kernel_kind == KernelKind::FullWarpShuffle) {
        gdn_full_warp_shuffle_fp32<<<blocks, threads>>>(
            arguments.q,
            arguments.k,
            arguments.v,
            arguments.g,
            arguments.beta,
            arguments.state_slot_ids,
            arguments.state_pool,
            arguments.output,
            batch_size,
            arguments.num_layers,
            arguments.num_heads,
            arguments.key_dim,
            arguments.value_dim,
            0
        );
    } else {
        gdn_block_reduction_fp32<<<blocks, threads>>>(
            arguments.q,
            arguments.k,
            arguments.v,
            arguments.g,
            arguments.beta,
            arguments.state_slot_ids,
            arguments.state_pool,
            arguments.output,
            batch_size,
            arguments.num_layers,
            arguments.num_heads,
            arguments.key_dim,
            arguments.value_dim,
            0
        );
    }
}


int benchmark_iterations(int batch_size)
{
    if (batch_size <= 64) {
        return 200;
    }
    if (batch_size <= 256) {
        return 100;
    }
    if (batch_size <= 1024) {
        return 30;
    }
    return 15;
}


float measure_kernel(KernelKind kernel_kind, const KernelArguments& arguments, int batch_size, size_t active_state_bytes)
{
    int iterations = benchmark_iterations(batch_size);
    int warmup_iterations = 5;

    check_cuda(cudaMemset(arguments.state_pool, 0, active_state_bytes));
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        launch_kernel(kernel_kind, arguments, batch_size);
    }
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    check_cuda(cudaMemset(arguments.state_pool, 0, active_state_bytes));
    check_cuda(cudaDeviceSynchronize());

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    check_cuda(cudaEventCreate(&start_event));
    check_cuda(cudaEventCreate(&stop_event));

    check_cuda(cudaEventRecord(start_event));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        launch_kernel(kernel_kind, arguments, batch_size);
    }
    check_cuda(cudaEventRecord(stop_event));
    check_cuda(cudaEventSynchronize(stop_event));
    check_cuda(cudaGetLastError());

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    check_cuda(cudaEventDestroy(start_event));
    check_cuda(cudaEventDestroy(stop_event));

    return elapsed_ms * 1000.0f / static_cast<float>(iterations);
}


float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}


float compare_one_step_outputs(
    const KernelArguments& arguments,
    int batch_size,
    size_t active_state_bytes,
    KernelKind compared_kernel)
{
    size_t output_elements = static_cast<size_t>(batch_size) * arguments.num_heads * arguments.value_dim;
    size_t output_bytes = output_elements * sizeof(float);
    std::vector<float> block_output(output_elements);
    std::vector<float> hybrid_output(output_elements);

    check_cuda(cudaMemset(arguments.state_pool, 0, active_state_bytes));
    launch_kernel(KernelKind::BlockReduction, arguments, batch_size);
    check_cuda(cudaGetLastError());
    check_cuda(cudaMemcpy(block_output.data(), arguments.output, output_bytes, cudaMemcpyDeviceToHost));

    check_cuda(cudaMemset(arguments.state_pool, 0, active_state_bytes));
    launch_kernel(compared_kernel, arguments, batch_size);
    check_cuda(cudaGetLastError());
    check_cuda(cudaMemcpy(hybrid_output.data(), arguments.output, output_bytes, cudaMemcpyDeviceToHost));

    float max_error = 0.0f;
    for (size_t index = 0; index < output_elements; ++index) {
        max_error = std::max(max_error, std::fabs(block_output[index] - hybrid_output[index]));
    }
    return max_error;
}


int main()
{
    const int max_batch_size = 2048;
    const int num_layers = 1;
    const int num_heads = 32;
    const int key_dim = 128;
    const int value_dim = 128;
    const int repeats = 5;
    const int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

    size_t qk_elements = static_cast<size_t>(max_batch_size) * num_heads * key_dim;
    size_t value_elements = static_cast<size_t>(max_batch_size) * num_heads * value_dim;
    size_t gate_elements = static_cast<size_t>(max_batch_size) * num_heads;
    size_t state_elements = static_cast<size_t>(max_batch_size) * num_layers * num_heads * key_dim * value_dim;

    std::vector<float> h_q(qk_elements);
    std::vector<float> h_k(qk_elements);
    std::vector<float> h_v(value_elements);
    std::vector<float> h_g(gate_elements);
    std::vector<float> h_beta(gate_elements);
    std::vector<int> h_state_slot_ids(max_batch_size);

    for (size_t index = 0; index < qk_elements; ++index) {
        h_q[index] = 0.02f * std::sin(static_cast<float>(index) * 0.013f);
        h_k[index] = 0.02f * std::cos(static_cast<float>(index) * 0.017f);
    }
    for (size_t index = 0; index < value_elements; ++index) {
        h_v[index] = 0.1f * std::sin(static_cast<float>(index) * 0.007f);
    }
    for (size_t index = 0; index < gate_elements; ++index) {
        h_g[index] = -0.05f - 0.001f * static_cast<float>(index % 17);
        h_beta[index] = 0.25f + 0.01f * static_cast<float>(index % 11);
    }
    for (int index = 0; index < max_batch_size; ++index) {
        h_state_slot_ids[index] = index;
    }

    float* d_q;
    float* d_k;
    float* d_v;
    float* d_g;
    float* d_beta;
    int* d_state_slot_ids;
    float* d_state_pool;
    float* d_output;

    size_t qk_bytes = qk_elements * sizeof(float);
    size_t value_bytes = value_elements * sizeof(float);
    size_t gate_bytes = gate_elements * sizeof(float);
    size_t slot_bytes = static_cast<size_t>(max_batch_size) * sizeof(int);
    size_t state_bytes = state_elements * sizeof(float);

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

    KernelArguments arguments{
        d_q,
        d_k,
        d_v,
        d_g,
        d_beta,
        d_state_slot_ids,
        d_state_pool,
        d_output,
        num_layers,
        num_heads,
        key_dim,
        value_dim
    };

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0));

    std::printf("GPU: %s\n", properties.name);
    std::printf("Shape: H=%d, Dk=%d, Dv=%d\n", num_heads, key_dim, value_dim);
    std::printf("State layout: [B, 1, H, Dk, Dv], FP32\n");
    std::printf("Maximum one-layer state pool: %.2f GiB\n", static_cast<double>(state_bytes) / 1024.0 / 1024.0 / 1024.0);
    std::printf("Each result is the median of %d measurements.\n\n", repeats);
    std::printf("     B |  Block us | Hybrid us |   Warp us | Hybrid vs Block | Warp vs Block | errors H/W\n");
    std::printf("-------+-----------+-----------+-----------+-----------------+---------------+-------------------\n");

    for (int batch_size : batch_sizes) {
        size_t active_state_elements = static_cast<size_t>(batch_size) * num_layers * num_heads * key_dim * value_dim;
        size_t active_state_bytes = active_state_elements * sizeof(float);
        std::vector<float> block_times;
        std::vector<float> hybrid_times;
        std::vector<float> warp_times;

        for (int repeat = 0; repeat < repeats; ++repeat) {
            if (repeat % 3 == 0) {
                block_times.push_back(measure_kernel(KernelKind::BlockReduction, arguments, batch_size, active_state_bytes));
                hybrid_times.push_back(measure_kernel(KernelKind::HybridReduction, arguments, batch_size, active_state_bytes));
                warp_times.push_back(measure_kernel(KernelKind::FullWarpShuffle, arguments, batch_size, active_state_bytes));
            } else if (repeat % 3 == 1) {
                hybrid_times.push_back(measure_kernel(KernelKind::HybridReduction, arguments, batch_size, active_state_bytes));
                warp_times.push_back(measure_kernel(KernelKind::FullWarpShuffle, arguments, batch_size, active_state_bytes));
                block_times.push_back(measure_kernel(KernelKind::BlockReduction, arguments, batch_size, active_state_bytes));
            } else {
                warp_times.push_back(measure_kernel(KernelKind::FullWarpShuffle, arguments, batch_size, active_state_bytes));
                block_times.push_back(measure_kernel(KernelKind::BlockReduction, arguments, batch_size, active_state_bytes));
                hybrid_times.push_back(measure_kernel(KernelKind::HybridReduction, arguments, batch_size, active_state_bytes));
            }
        }

        float block_us = median(block_times);
        float hybrid_us = median(hybrid_times);
        float warp_us = median(warp_times);
        double hybrid_change = (static_cast<double>(hybrid_us) / block_us - 1.0) * 100.0;
        double warp_change = (static_cast<double>(warp_us) / block_us - 1.0) * 100.0;
        float hybrid_error = compare_one_step_outputs(
            arguments,
            batch_size,
            active_state_bytes,
            KernelKind::HybridReduction
        );
        float warp_error = compare_one_step_outputs(
            arguments,
            batch_size,
            active_state_bytes,
            KernelKind::FullWarpShuffle
        );

        std::printf(
            "%6d | %9.3f | %9.3f | %9.3f | %+14.2f%% | %+12.2f%% | %.1e / %.1e\n",
            batch_size,
            block_us,
            hybrid_us,
            warp_us,
            hybrid_change,
            warp_change,
            hybrid_error,
            warp_error
        );
    }

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
