#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int correctness_recurrent_steps = 8;

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
        long long state_offset =
            state_matrix_base
            + static_cast<long long>(key_index) * value_dim
            + value_index;

        float decayed_state = state_pool[state_offset] * decay;
        remembered_value += shared_key[key_index] * decayed_state;
    }

    float delta = beta_value * (v[value_offset] - remembered_value);
    float result = 0.0f;

    for (int key_index = 0; key_index < key_dim; ++key_index) {
        long long state_offset =
            state_matrix_base
            + static_cast<long long>(key_index) * value_dim
            + value_index;

        float decayed_state = state_pool[state_offset] * decay;
        float updated_state = decayed_state + shared_key[key_index] * delta;

        state_pool[state_offset] = updated_state;
        result += shared_query[key_index] * updated_state;
    }

    output[value_offset] = result;
}


/*
Full Warp 基线：

一个 Block 负责一个 (batch, head)。
128 个线程分别负责 State 的 128 个 Dv 列。
*/
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

    long long qk_base =
        static_cast<long long>(batch_index * num_heads + head_index)
        * key_dim;

    int gate_offset = batch_index * num_heads + head_index;

    float query_value = q[qk_base + thread_index];
    float key_value = k[qk_base + thread_index];

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

        if (lane_index == 0) {
            shared_query_multiplier =
                rsqrtf(query_block_sum + 1e-6f)
                * rsqrtf(static_cast<float>(key_dim));

            shared_key_multiplier =
                rsqrtf(key_block_sum + 1e-6f);

            shared_decay = expf(g[gate_offset]);
            shared_beta = beta[gate_offset];
        }
    }

    __syncthreads();

    shared_query[thread_index] = query_value * shared_query_multiplier;
    shared_key[thread_index] = key_value * shared_key_multiplier;

    __syncthreads();

    int state_slot = state_slot_ids[batch_index];

    long long state_matrix_base =
        static_cast<long long>(state_slot)
            * num_layers
            * num_heads
            * key_dim
            * value_dim
        + static_cast<long long>(gdn_index)
            * num_heads
            * key_dim
            * value_dim
        + static_cast<long long>(head_index)
            * key_dim
            * value_dim;

    long long value_offset =
        static_cast<long long>(batch_index * num_heads + head_index)
            * value_dim
        + value_index;

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


/*
Split-K + float4：

一个 Block 仍然有 128 个线程，也就是 4 个 Warp。

warp_index 决定当前 Warp 负责哪个 Dk 分块。
lane_index 决定当前线程负责哪个 float4 Dv 向量。
*/
__global__ void gdn_splitk_float4_fp32(
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
    constexpr int values_per_lane = 4;

    int batch_head_index = blockIdx.x;
    int batch_index = batch_head_index / num_heads;
    int head_index = batch_head_index % num_heads;

    int thread_index = threadIdx.x;
    int lane_index = thread_index % warp_size;
    int warp_index = thread_index / warp_size;

    if (batch_index >= batch_size) {
        return;
    }

    /*
    当前 Warp 负责 32 个 Dk 行。

    Warp 0: 0～31
    Warp 1: 32～63
    Warp 2: 64～95
    Warp 3: 96～127
    */
    int key_begin = warp_index * warp_size;
    int key_end = key_begin + warp_size;

    /*
    当前 Lane 负责 4 个连续 Dv 元素。

    lane 0:  0～3
    lane 1:  4～7
    ...
    lane 31: 124～127
    */
    int value_base = lane_index * values_per_lane;

    __shared__ float shared_query[128];
    __shared__ float shared_key[128];

    __shared__ float shared_query_warp_sums[num_warps];
    __shared__ float shared_key_warp_sums[num_warps];

    __shared__ float shared_query_multiplier;
    __shared__ float shared_key_multiplier;
    __shared__ float shared_decay;
    __shared__ float shared_beta;

    /*
    每个 Warp 都会计算一个局部 remembered：

        partial_remembered[warp][value]

    4 个 Warp 分别对应 4 个 Dk 分块。
    */
    __shared__ float shared_remembered[num_warps][128];

    /*
    合并 4 个局部 remembered 后计算出的完整 delta。
    */
    __shared__ float shared_delta[128];

    /*
    每个 Warp 计算自己的局部 output：

        partial_output[warp][value]
    */
    __shared__ float shared_output[num_warps][128];

    long long qk_base =
        static_cast<long long>(batch_index * num_heads + head_index)
        * key_dim;

    int gate_offset = batch_index * num_heads + head_index;

    /*
    128 个线程分别读取一个 Q 和 K 元素。
    */
    float query_value = q[qk_base + thread_index];
    float key_value = k[qk_base + thread_index];

    shared_query[thread_index] = query_value;
    shared_key[thread_index] = key_value;

    /*
    第一级：每个 Warp 分别归约自己的 32 个元素。
    */
    float query_squared_sum = warp_reduce_sum(query_value * query_value);
    float key_squared_sum = warp_reduce_sum(key_value * key_value);

    if (lane_index == 0) {
        shared_query_warp_sums[warp_index] = query_squared_sum;
        shared_key_warp_sums[warp_index] = key_squared_sum;
    }

    __syncthreads();

    /*
    第二级：Warp 0 将四个 Warp 的部分和归约成完整范数。
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

        if (lane_index == 0) {
            shared_query_multiplier =
                rsqrtf(query_block_sum + 1e-6f)
                * rsqrtf(static_cast<float>(key_dim));

            shared_key_multiplier =
                rsqrtf(key_block_sum + 1e-6f);

            shared_decay = expf(g[gate_offset]);
            shared_beta = beta[gate_offset];
        }
    }

    __syncthreads();

    shared_query[thread_index] = query_value * shared_query_multiplier;
    shared_key[thread_index] = key_value * shared_key_multiplier;

    __syncthreads();

    int state_slot = state_slot_ids[batch_index];

    long long state_matrix_base =
        static_cast<long long>(state_slot)
            * num_layers
            * num_heads
            * key_dim
            * value_dim
        + static_cast<long long>(gdn_index)
            * num_heads
            * key_dim
            * value_dim
        + static_cast<long long>(head_index)
            * key_dim
            * value_dim;

    long long value_matrix_base =
        static_cast<long long>(batch_index * num_heads + head_index)
        * value_dim;

    /*
    四个寄存器分别保存当前线程负责的四个 Dv 元素的
   局部 remembered。
    */
    float remembered_x = 0.0f;
    float remembered_y = 0.0f;
    float remembered_z = 0.0f;
    float remembered_w = 0.0f;

    /*
    第一遍 State 扫描。

    每个 Warp 只扫描自己负责的 32 个 Dk 行。
    每个 Lane 使用 float4 读取四个连续 Dv 元素。
    */
    for (int key_index = key_begin; key_index < key_end; ++key_index) {
        long long state_row_base =
            state_matrix_base
            + static_cast<long long>(key_index) * value_dim;

        const float4* state_row =
            reinterpret_cast<const float4*>(
                state_pool + state_row_base
            );

        float4 state_values = state_row[lane_index];
        float normalized_key = shared_key[key_index];

        state_values.x *= shared_decay;
        state_values.y *= shared_decay;
        state_values.z *= shared_decay;
        state_values.w *= shared_decay;

        remembered_x += normalized_key * state_values.x;
        remembered_y += normalized_key * state_values.y;
        remembered_z += normalized_key * state_values.z;
        remembered_w += normalized_key * state_values.w;
    }

    /*
    每个 Warp 将自己的局部 remembered 写到 Shared Memory。
    */
    shared_remembered[warp_index][value_base] = remembered_x;
    shared_remembered[warp_index][value_base + 1] = remembered_y;
    shared_remembered[warp_index][value_base + 2] = remembered_z;
    shared_remembered[warp_index][value_base + 3] = remembered_w;

    __syncthreads();

    /*
    Warp 0 合并四个 Dk 分块的 remembered，并计算 delta。

    只有 Warp 0 执行，避免四个 Warp 重复读取 V、
    重复计算相同的 delta。
    */
    if (warp_index == 0) {
        float total_remembered_x = 0.0f;
        float total_remembered_y = 0.0f;
        float total_remembered_z = 0.0f;
        float total_remembered_w = 0.0f;

        for (int source_warp = 0; source_warp < num_warps; ++source_warp) {
            total_remembered_x +=
                shared_remembered[source_warp][value_base];

            total_remembered_y +=
                shared_remembered[source_warp][value_base + 1];

            total_remembered_z +=
                shared_remembered[source_warp][value_base + 2];

            total_remembered_w +=
                shared_remembered[source_warp][value_base + 3];
        }

        const float4* value_vector =
            reinterpret_cast<const float4*>(
                v + value_matrix_base
            );

        float4 value_values = value_vector[lane_index];

        shared_delta[value_base] =
            shared_beta
            * (value_values.x - total_remembered_x);

        shared_delta[value_base + 1] =
            shared_beta
            * (value_values.y - total_remembered_y);

        shared_delta[value_base + 2] =
            shared_beta
            * (value_values.z - total_remembered_z);

        shared_delta[value_base + 3] =
            shared_beta
            * (value_values.w - total_remembered_w);
    }

    __syncthreads();

    float delta_x = shared_delta[value_base];
    float delta_y = shared_delta[value_base + 1];
    float delta_z = shared_delta[value_base + 2];
    float delta_w = shared_delta[value_base + 3];

    float output_x = 0.0f;
    float output_y = 0.0f;
    float output_z = 0.0f;
    float output_w = 0.0f;

    /*
    第二遍 State 扫描。

    每个 Warp 更新自己负责的 Dk 行。
    不同 Warp 更新不同 State 行，因此不存在写冲突。
    */
    for (int key_index = key_begin; key_index < key_end; ++key_index) {
        long long state_row_base =
            state_matrix_base
            + static_cast<long long>(key_index) * value_dim;

        float4* state_row =
            reinterpret_cast<float4*>(
                state_pool + state_row_base
            );

        float4 state_values = state_row[lane_index];

        float normalized_key = shared_key[key_index];
        float normalized_query = shared_query[key_index];

        state_values.x =
            state_values.x * shared_decay
            + normalized_key * delta_x;

        state_values.y =
            state_values.y * shared_decay
            + normalized_key * delta_y;

        state_values.z =
            state_values.z * shared_decay
            + normalized_key * delta_z;

        state_values.w =
            state_values.w * shared_decay
            + normalized_key * delta_w;

        state_row[lane_index] = state_values;

        output_x += normalized_query * state_values.x;
        output_y += normalized_query * state_values.y;
        output_z += normalized_query * state_values.z;
        output_w += normalized_query * state_values.w;
    }

    /*
    当前 Warp 得到的只是自己 32 个 Dk 行产生的局部输出。
    */
    shared_output[warp_index][value_base] = output_x;
    shared_output[warp_index][value_base + 1] = output_y;
    shared_output[warp_index][value_base + 2] = output_z;
    shared_output[warp_index][value_base + 3] = output_w;

    __syncthreads();

    /*
    Warp 0 合并四个 Dk 分块的 output，并使用 float4 写回。
    */
    if (warp_index == 0) {
        float final_output_x = 0.0f;
        float final_output_y = 0.0f;
        float final_output_z = 0.0f;
        float final_output_w = 0.0f;

        for (int source_warp = 0; source_warp < num_warps; ++source_warp) {
            final_output_x += shared_output[source_warp][value_base];
            final_output_y += shared_output[source_warp][value_base + 1];
            final_output_z += shared_output[source_warp][value_base + 2];
            final_output_w += shared_output[source_warp][value_base + 3];
        }

        float4 result = make_float4(
            final_output_x,
            final_output_y,
            final_output_z,
            final_output_w
        );

        float4* output_vector =
            reinterpret_cast<float4*>(
                output + value_matrix_base
            );

        output_vector[lane_index] = result;
    }
}


__global__ void max_absolute_difference_kernel(
    const float* first,
    const float* second,
    size_t num_elements,
    unsigned int* maximum_bits)
{
    __shared__ float shared_maximum[256];

    int thread_index = threadIdx.x;
    size_t global_index =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + thread_index;

    size_t global_stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;

    float local_maximum = 0.0f;

    for (
        size_t index = global_index;
        index < num_elements;
        index += global_stride
    ) {
        float difference = fabsf(first[index] - second[index]);
        local_maximum = fmaxf(local_maximum, difference);
    }

    shared_maximum[thread_index] = local_maximum;
    __syncthreads();

    for (int stride = 128; stride > 0; stride /= 2) {
        if (thread_index < stride) {
            shared_maximum[thread_index] = fmaxf(
                shared_maximum[thread_index],
                shared_maximum[thread_index + stride]
            );
        }

        __syncthreads();
    }

    if (thread_index == 0) {
        unsigned int value_bits =
            __float_as_uint(shared_maximum[0]);

        atomicMax(maximum_bits, value_bits);
    }
}


float compute_max_absolute_difference(
    const float* first,
    const float* second,
    size_t num_elements)
{
    unsigned int* device_maximum_bits = nullptr;

    check_cuda(cudaMalloc(
        &device_maximum_bits,
        sizeof(unsigned int)
    ));

    check_cuda(cudaMemset(
        device_maximum_bits,
        0,
        sizeof(unsigned int)
    ));

    int threads = 256;
    int blocks = 1024;

    max_absolute_difference_kernel<<<blocks, threads>>>(
        first,
        second,
        num_elements,
        device_maximum_bits
    );

    check_cuda(cudaGetLastError());

    unsigned int host_maximum_bits = 0;

    check_cuda(cudaMemcpy(
        &host_maximum_bits,
        device_maximum_bits,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    ));

    check_cuda(cudaFree(device_maximum_bits));

    union {
        unsigned int bits;
        float value;
    } converter{};

    converter.bits = host_maximum_bits;
    return converter.value;
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
    FullWarp,
    SplitKFloat4
};


void launch_kernel(
    KernelKind kernel_kind,
    const KernelArguments& arguments,
    int batch_size,
    int gdn_index)
{
    int blocks = batch_size * arguments.num_heads;
    int threads = 128;

    if (kernel_kind == KernelKind::FullWarp) {
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
            gdn_index
        );
    } else {
        gdn_splitk_float4_fp32<<<blocks, threads>>>(
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
            gdn_index
        );
    }
}


float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}


float measure_kernel(
    KernelKind kernel_kind,
    const KernelArguments& arguments,
    int batch_size,
    size_t active_state_bytes)
{
    const int warmup_iterations = 3;

    int benchmark_iterations = 15;

    if (batch_size <= 64) {
        benchmark_iterations = 200;
    } else if (batch_size <= 256) {
        benchmark_iterations = 100;
    } else if (batch_size <= 512) {
        benchmark_iterations = 50;
    } else if (batch_size <= 1024) {
        benchmark_iterations = 30;
    }

    check_cuda(cudaMemset(
        arguments.state_pool,
        0,
        active_state_bytes
    ));

    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        for (
            int gdn_index = 0;
            gdn_index < arguments.num_layers;
            ++gdn_index
        ) {
            launch_kernel(
                kernel_kind,
                arguments,
                batch_size,
                gdn_index
            );
        }
    }

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    check_cuda(cudaMemset(
        arguments.state_pool,
        0,
        active_state_bytes
    ));

    check_cuda(cudaDeviceSynchronize());

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    check_cuda(cudaEventCreate(&start_event));
    check_cuda(cudaEventCreate(&stop_event));

    check_cuda(cudaEventRecord(start_event));

    for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
        for (
            int gdn_index = 0;
            gdn_index < arguments.num_layers;
            ++gdn_index
        ) {
            launch_kernel(
                kernel_kind,
                arguments,
                batch_size,
                gdn_index
            );
        }
    }

    check_cuda(cudaEventRecord(stop_event));
    check_cuda(cudaEventSynchronize(stop_event));
    check_cuda(cudaGetLastError());

    float elapsed_ms = 0.0f;

    check_cuda(cudaEventElapsedTime(
        &elapsed_ms,
        start_event,
        stop_event
    ));

    check_cuda(cudaEventDestroy(start_event));
    check_cuda(cudaEventDestroy(stop_event));

    return elapsed_ms * 1000.0f
        / static_cast<float>(benchmark_iterations);
}


struct ErrorResult {
    float state_error;
    float output_error;
};


ErrorResult compare_results(
    const KernelArguments& full_warp_arguments,
    const KernelArguments& splitk_arguments,
    int batch_size,
    size_t active_state_elements,
    size_t active_state_bytes)
{
    size_t output_elements =
        static_cast<size_t>(batch_size)
        * full_warp_arguments.num_heads
        * full_warp_arguments.value_dim;

    check_cuda(cudaMemset(
        full_warp_arguments.state_pool,
        0,
        active_state_bytes
    ));

    check_cuda(cudaMemset(
        splitk_arguments.state_pool,
        0,
        active_state_bytes
    ));

    for (
        int decode_step = 0;
        decode_step < correctness_recurrent_steps;
        ++decode_step
    ) {
        for (
            int gdn_index = 0;
            gdn_index < full_warp_arguments.num_layers;
            ++gdn_index
        ) {
            launch_kernel(
                KernelKind::FullWarp,
                full_warp_arguments,
                batch_size,
                gdn_index
            );

            launch_kernel(
                KernelKind::SplitKFloat4,
                splitk_arguments,
                batch_size,
                gdn_index
            );
        }
    }

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    ErrorResult result{};

    result.state_error =
        compute_max_absolute_difference(
            full_warp_arguments.state_pool,
            splitk_arguments.state_pool,
            active_state_elements
        );

    result.output_error =
        compute_max_absolute_difference(
            full_warp_arguments.output,
            splitk_arguments.output,
            output_elements
        );

    return result;
}


int main()
{
    #ifdef PROFILE_BATCH_SIZE

        const int max_batch_size = PROFILE_BATCH_SIZE;
        const int repeats = 1;

        const int batch_sizes[] = {
            PROFILE_BATCH_SIZE
        };

    #else

        const int max_batch_size = 2048;
        const int repeats = 5;

        const int batch_sizes[] = {
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048
        };

    #endif

        const int num_layers = 1;
        const int num_heads = 32;
        const int key_dim = 128;
        const int value_dim = 128;
    size_t qk_elements =
        static_cast<size_t>(max_batch_size)
        * num_heads
        * key_dim;

    size_t value_elements =
        static_cast<size_t>(max_batch_size)
        * num_heads
        * value_dim;

    size_t gate_elements =
        static_cast<size_t>(max_batch_size)
        * num_heads;

    size_t state_elements =
        static_cast<size_t>(max_batch_size)
        * num_layers
        * num_heads
        * key_dim
        * value_dim;

    std::vector<float> host_q(qk_elements);
    std::vector<float> host_k(qk_elements);
    std::vector<float> host_v(value_elements);
    std::vector<float> host_g(gate_elements);
    std::vector<float> host_beta(gate_elements);
    std::vector<int> host_state_slot_ids(max_batch_size);

    for (size_t index = 0; index < qk_elements; ++index) {
        host_q[index] =
            0.02f * std::sin(static_cast<float>(index) * 0.013f);

        host_k[index] =
            0.02f * std::cos(static_cast<float>(index) * 0.017f);
    }

    for (size_t index = 0; index < value_elements; ++index) {
        host_v[index] =
            0.1f * std::sin(static_cast<float>(index) * 0.007f);
    }

    for (size_t index = 0; index < gate_elements; ++index) {
        host_g[index] =
            -0.05f
            - 0.001f * static_cast<float>(index % 17);

        host_beta[index] =
            0.25f
            + 0.01f * static_cast<float>(index % 11);
    }

    for (int index = 0; index < max_batch_size; ++index) {
        host_state_slot_ids[index] = index;
    }

    float* device_q = nullptr;
    float* device_k = nullptr;
    float* device_v = nullptr;
    float* device_g = nullptr;
    float* device_beta = nullptr;
    int* device_state_slot_ids = nullptr;

    float* device_full_warp_state = nullptr;
    float* device_splitk_state = nullptr;

    float* device_full_warp_output = nullptr;
    float* device_splitk_output = nullptr;

    size_t qk_bytes = qk_elements * sizeof(float);
    size_t value_bytes = value_elements * sizeof(float);
    size_t gate_bytes = gate_elements * sizeof(float);
    size_t slot_bytes = static_cast<size_t>(max_batch_size) * sizeof(int);
    size_t state_bytes = state_elements * sizeof(float);

    check_cuda(cudaMalloc(&device_q, qk_bytes));
    check_cuda(cudaMalloc(&device_k, qk_bytes));
    check_cuda(cudaMalloc(&device_v, value_bytes));
    check_cuda(cudaMalloc(&device_g, gate_bytes));
    check_cuda(cudaMalloc(&device_beta, gate_bytes));
    check_cuda(cudaMalloc(&device_state_slot_ids, slot_bytes));

    check_cuda(cudaMalloc(&device_full_warp_state, state_bytes));
    check_cuda(cudaMalloc(&device_splitk_state, state_bytes));

    check_cuda(cudaMalloc(&device_full_warp_output, value_bytes));
    check_cuda(cudaMalloc(&device_splitk_output, value_bytes));

    check_cuda(cudaMemcpy(
        device_q,
        host_q.data(),
        qk_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_k,
        host_k.data(),
        qk_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_v,
        host_v.data(),
        value_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_g,
        host_g.data(),
        gate_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_beta,
        host_beta.data(),
        gate_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_state_slot_ids,
        host_state_slot_ids.data(),
        slot_bytes,
        cudaMemcpyHostToDevice
    ));

    KernelArguments full_warp_arguments{
        device_q,
        device_k,
        device_v,
        device_g,
        device_beta,
        device_state_slot_ids,
        device_full_warp_state,
        device_full_warp_output,
        num_layers,
        num_heads,
        key_dim,
        value_dim
    };

    KernelArguments splitk_arguments{
        device_q,
        device_k,
        device_v,
        device_g,
        device_beta,
        device_state_slot_ids,
        device_splitk_state,
        device_splitk_output,
        num_layers,
        num_heads,
        key_dim,
        value_dim
    };

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0));

    std::printf("GPU: %s\n", properties.name);
    std::printf(
        "Shape: H=%d, Dk=%d, Dv=%d\n",
        num_heads,
        key_dim,
        value_dim
    );

    std::printf("GDN layers: %d\n", num_layers);

    std::printf(
        "Correctness recurrent steps: %d\n",
        correctness_recurrent_steps
    );

    std::printf(
        "One state pool: %.2f GiB\n",
        static_cast<double>(state_bytes)
            / 1024.0 / 1024.0 / 1024.0
    );

    std::printf(
        "Two state pools: %.2f GiB\n",
        static_cast<double>(state_bytes * 2)
            / 1024.0 / 1024.0 / 1024.0
    );

    std::printf(
        "Each result is the median of %d measurements.\n\n",
        repeats
    );

    std::printf(
        "     B | FullWarp step us | SplitK step us | FullWarp layer us | SplitK layer us |    change | speedup | state error | output error\n"
    );

    std::printf(
        "-------+------------------+----------------+-------------------+-----------------+-----------+---------+-------------+-------------\n"
    );

    for (int batch_size : batch_sizes) {
        size_t active_state_elements =
            static_cast<size_t>(batch_size)
            * num_layers
            * num_heads
            * key_dim
            * value_dim;

        size_t active_state_bytes =
            active_state_elements * sizeof(float);

        std::vector<float> full_warp_times;
        std::vector<float> splitk_times;

        for (int repeat = 0; repeat < repeats; ++repeat) {
            if (repeat % 2 == 0) {
                full_warp_times.push_back(
                    measure_kernel(
                        KernelKind::FullWarp,
                        full_warp_arguments,
                        batch_size,
                        active_state_bytes
                    )
                );

                splitk_times.push_back(
                    measure_kernel(
                        KernelKind::SplitKFloat4,
                        splitk_arguments,
                        batch_size,
                        active_state_bytes
                    )
                );
            } else {
                splitk_times.push_back(
                    measure_kernel(
                        KernelKind::SplitKFloat4,
                        splitk_arguments,
                        batch_size,
                        active_state_bytes
                    )
                );

                full_warp_times.push_back(
                    measure_kernel(
                        KernelKind::FullWarp,
                        full_warp_arguments,
                        batch_size,
                        active_state_bytes
                    )
                );
            }
        }

        float full_warp_step_us = median(full_warp_times);
        float splitk_step_us = median(splitk_times);

        float full_warp_layer_us =
            full_warp_step_us / static_cast<float>(num_layers);

        float splitk_layer_us =
            splitk_step_us / static_cast<float>(num_layers);

        double change_percent =
            (static_cast<double>(splitk_step_us) / full_warp_step_us - 1.0)
            * 100.0;

        double speedup =
            static_cast<double>(full_warp_step_us) / splitk_step_us;

        ErrorResult errors = compare_results(
            full_warp_arguments,
            splitk_arguments,
            batch_size,
            active_state_elements,
            active_state_bytes
        );

        std::printf(
            "%6d | %16.3f | %14.3f | %17.3f | %15.3f | %+8.2f%% | %7.3fx | %11.2e | %11.2e\n",
            batch_size,
            full_warp_step_us,
            splitk_step_us,
            full_warp_layer_us,
            splitk_layer_us,
            change_percent,
            speedup,
            errors.state_error,
            errors.output_error
        );
    }

    check_cuda(cudaFree(device_q));
    check_cuda(cudaFree(device_k));
    check_cuda(cudaFree(device_v));
    check_cuda(cudaFree(device_g));
    check_cuda(cudaFree(device_beta));
    check_cuda(cudaFree(device_state_slot_ids));

    check_cuda(cudaFree(device_full_warp_state));
    check_cuda(cudaFree(device_splitk_state));

    check_cuda(cudaFree(device_full_warp_output));
    check_cuda(cudaFree(device_splitk_output));

    return EXIT_SUCCESS;
}