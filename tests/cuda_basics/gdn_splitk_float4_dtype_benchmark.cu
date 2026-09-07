#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>


constexpr int BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int VALUES_PER_LANE = 4;

constexpr int NUM_HEADS = 32;
constexpr int KEY_DIM = 128;
constexpr int VALUE_DIM = 128;
constexpr int NUM_LAYERS = 1;


struct alignas(8) BFloat16x4 {
    __nv_bfloat162 xy;
    __nv_bfloat162 zw;
};


struct FloatValues4 {
    float x;
    float y;
    float z;
    float w;
};


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


__device__ __forceinline__ float load_scalar(float value)
{
    return value;
}


__device__ __forceinline__ float load_scalar(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}


template<typename InputType>
__device__ __forceinline__ FloatValues4 load_four_values(
    const InputType* pointer,
    long long element_offset)
{
    const float4* vector_pointer =
        reinterpret_cast<const float4*>(
            pointer + element_offset
        );

    float4 packed = vector_pointer[0];

    return {
        packed.x,
        packed.y,
        packed.z,
        packed.w
    };
}


template<>
__device__ __forceinline__ FloatValues4 load_four_values<__nv_bfloat16>(
    const __nv_bfloat16* pointer,
    long long element_offset)
{
    const BFloat16x4* vector_pointer =
        reinterpret_cast<const BFloat16x4*>(
            pointer + element_offset
        );

    BFloat16x4 packed = vector_pointer[0];

    float2 xy = __bfloat1622float2(packed.xy);
    float2 zw = __bfloat1622float2(packed.zw);

    return {
        xy.x,
        xy.y,
        zw.x,
        zw.y
    };
}


template<typename OutputType>
__device__ __forceinline__ void store_four_values(
    OutputType* pointer,
    long long element_offset,
    FloatValues4 values)
{
    float4* vector_pointer =
        reinterpret_cast<float4*>(
            pointer + element_offset
        );

    vector_pointer[0] = make_float4(
        values.x,
        values.y,
        values.z,
        values.w
    );
}


template<>
__device__ __forceinline__ void store_four_values<__nv_bfloat16>(
    __nv_bfloat16* pointer,
    long long element_offset,
    FloatValues4 values)
{
    BFloat16x4* vector_pointer =
        reinterpret_cast<BFloat16x4*>(
            pointer + element_offset
        );

    BFloat16x4 packed;

    packed.xy = __floats2bfloat162_rn(
        values.x,
        values.y
    );

    packed.zw = __floats2bfloat162_rn(
        values.z,
        values.w
    );

    vector_pointer[0] = packed;
}


template<typename InputType, typename OutputType>
__global__ void gdn_splitk_float4(
    const InputType* query,
    const InputType* key,
    const InputType* value,
    const float* g,
    const InputType* beta,
    const long long* state_slot_ids,
    float* state_pool,
    OutputType* output,
    int batch_size,
    int gdn_index)
{
    int batch_head_index = blockIdx.x;
    int batch_index = batch_head_index / NUM_HEADS;
    int head_index = batch_head_index % NUM_HEADS;

    int thread_index = threadIdx.x;
    int lane_index = thread_index % WARP_SIZE;
    int warp_index = thread_index / WARP_SIZE;

    if (batch_index >= batch_size) {
        return;
    }

    int key_begin = warp_index * WARP_SIZE;
    int key_end = key_begin + WARP_SIZE;
    int value_base = lane_index * VALUES_PER_LANE;

    __shared__ float shared_query[KEY_DIM];
    __shared__ float shared_key[KEY_DIM];

    __shared__ float shared_query_warp_sums[NUM_WARPS];
    __shared__ float shared_key_warp_sums[NUM_WARPS];

    __shared__ float shared_query_multiplier;
    __shared__ float shared_key_multiplier;
    __shared__ float shared_decay;
    __shared__ float shared_beta;

    __shared__ float shared_remembered[NUM_WARPS][VALUE_DIM];
    __shared__ float shared_delta[VALUE_DIM];
    __shared__ float shared_output[NUM_WARPS][VALUE_DIM];

    long long qk_base =
        static_cast<long long>(
            batch_index * NUM_HEADS + head_index
        ) * KEY_DIM;

    int gate_offset =
        batch_index * NUM_HEADS + head_index;

    float query_value =
        load_scalar(
            query[qk_base + thread_index]
        );

    float key_value =
        load_scalar(
            key[qk_base + thread_index]
        );

    shared_query[thread_index] = query_value;
    shared_key[thread_index] = key_value;

    float query_squared_sum =
        warp_reduce_sum(
            query_value * query_value
        );

    float key_squared_sum =
        warp_reduce_sum(
            key_value * key_value
        );

    if (lane_index == 0) {
        shared_query_warp_sums[warp_index] =
            query_squared_sum;

        shared_key_warp_sums[warp_index] =
            key_squared_sum;
    }

    __syncthreads();

    if (warp_index == 0) {
        float query_block_sum =
            lane_index < NUM_WARPS
                ? shared_query_warp_sums[lane_index]
                : 0.0f;

        float key_block_sum =
            lane_index < NUM_WARPS
                ? shared_key_warp_sums[lane_index]
                : 0.0f;

        query_block_sum =
            warp_reduce_sum(query_block_sum);

        key_block_sum =
            warp_reduce_sum(key_block_sum);

        if (lane_index == 0) {
            shared_query_multiplier =
                rsqrtf(query_block_sum + 1e-6f)
                * rsqrtf(
                    static_cast<float>(KEY_DIM)
                );

            shared_key_multiplier =
                rsqrtf(key_block_sum + 1e-6f);

            shared_decay =
                expf(g[gate_offset]);

            shared_beta =
                load_scalar(
                    beta[gate_offset]
                );
        }
    }

    __syncthreads();

    shared_query[thread_index] =
        query_value
        * shared_query_multiplier;

    shared_key[thread_index] =
        key_value
        * shared_key_multiplier;

    __syncthreads();

    long long state_slot =
        state_slot_ids[batch_index];

    long long state_matrix_base =
        state_slot
            * NUM_LAYERS
            * NUM_HEADS
            * KEY_DIM
            * VALUE_DIM
        + static_cast<long long>(gdn_index)
            * NUM_HEADS
            * KEY_DIM
            * VALUE_DIM
        + static_cast<long long>(head_index)
            * KEY_DIM
            * VALUE_DIM;

    long long value_matrix_base =
        static_cast<long long>(
            batch_index * NUM_HEADS + head_index
        ) * VALUE_DIM;

    float remembered_x = 0.0f;
    float remembered_y = 0.0f;
    float remembered_z = 0.0f;
    float remembered_w = 0.0f;

    for (
        int key_index = key_begin;
        key_index < key_end;
        ++key_index
    ) {
        long long state_row_base =
            state_matrix_base
            + static_cast<long long>(key_index)
                * VALUE_DIM;

        const float4* state_row =
            reinterpret_cast<const float4*>(
                state_pool + state_row_base
            );

        float4 state_values =
            state_row[lane_index];

        float normalized_key =
            shared_key[key_index];

        state_values.x *= shared_decay;
        state_values.y *= shared_decay;
        state_values.z *= shared_decay;
        state_values.w *= shared_decay;

        remembered_x +=
            normalized_key
            * state_values.x;

        remembered_y +=
            normalized_key
            * state_values.y;

        remembered_z +=
            normalized_key
            * state_values.z;

        remembered_w +=
            normalized_key
            * state_values.w;
    }

    shared_remembered[warp_index][value_base] =
        remembered_x;

    shared_remembered[warp_index][value_base + 1] =
        remembered_y;

    shared_remembered[warp_index][value_base + 2] =
        remembered_z;

    shared_remembered[warp_index][value_base + 3] =
        remembered_w;

    __syncthreads();

    if (warp_index == 0) {
        float total_remembered_x = 0.0f;
        float total_remembered_y = 0.0f;
        float total_remembered_z = 0.0f;
        float total_remembered_w = 0.0f;

        for (
            int source_warp = 0;
            source_warp < NUM_WARPS;
            ++source_warp
        ) {
            total_remembered_x +=
                shared_remembered[
                    source_warp
                ][value_base];

            total_remembered_y +=
                shared_remembered[
                    source_warp
                ][value_base + 1];

            total_remembered_z +=
                shared_remembered[
                    source_warp
                ][value_base + 2];

            total_remembered_w +=
                shared_remembered[
                    source_warp
                ][value_base + 3];
        }

        long long value_offset =
            value_matrix_base
            + value_base;

        FloatValues4 value_values =
            load_four_values(
                value,
                value_offset
            );

        shared_delta[value_base] =
            shared_beta
            * (
                value_values.x
                - total_remembered_x
            );

        shared_delta[value_base + 1] =
            shared_beta
            * (
                value_values.y
                - total_remembered_y
            );

        shared_delta[value_base + 2] =
            shared_beta
            * (
                value_values.z
                - total_remembered_z
            );

        shared_delta[value_base + 3] =
            shared_beta
            * (
                value_values.w
                - total_remembered_w
            );
    }

    __syncthreads();

    float delta_x =
        shared_delta[value_base];

    float delta_y =
        shared_delta[value_base + 1];

    float delta_z =
        shared_delta[value_base + 2];

    float delta_w =
        shared_delta[value_base + 3];

    float output_x = 0.0f;
    float output_y = 0.0f;
    float output_z = 0.0f;
    float output_w = 0.0f;

    for (
        int key_index = key_begin;
        key_index < key_end;
        ++key_index
    ) {
        long long state_row_base =
            state_matrix_base
            + static_cast<long long>(key_index)
                * VALUE_DIM;

        float4* state_row =
            reinterpret_cast<float4*>(
                state_pool + state_row_base
            );

        float4 state_values =
            state_row[lane_index];

        float normalized_key =
            shared_key[key_index];

        float normalized_query =
            shared_query[key_index];

        state_values.x =
            state_values.x
                * shared_decay
            + normalized_key
                * delta_x;

        state_values.y =
            state_values.y
                * shared_decay
            + normalized_key
                * delta_y;

        state_values.z =
            state_values.z
                * shared_decay
            + normalized_key
                * delta_z;

        state_values.w =
            state_values.w
                * shared_decay
            + normalized_key
                * delta_w;

        state_row[lane_index] =
            state_values;

        output_x +=
            normalized_query
            * state_values.x;

        output_y +=
            normalized_query
            * state_values.y;

        output_z +=
            normalized_query
            * state_values.z;

        output_w +=
            normalized_query
            * state_values.w;
    }

    shared_output[warp_index][value_base] =
        output_x;

    shared_output[warp_index][value_base + 1] =
        output_y;

    shared_output[warp_index][value_base + 2] =
        output_z;

    shared_output[warp_index][value_base + 3] =
        output_w;

    __syncthreads();

    if (warp_index == 0) {
        FloatValues4 final_output = {
            0.0f,
            0.0f,
            0.0f,
            0.0f
        };

        for (
            int source_warp = 0;
            source_warp < NUM_WARPS;
            ++source_warp
        ) {
            final_output.x +=
                shared_output[
                    source_warp
                ][value_base];

            final_output.y +=
                shared_output[
                    source_warp
                ][value_base + 1];

            final_output.z +=
                shared_output[
                    source_warp
                ][value_base + 2];

            final_output.w +=
                shared_output[
                    source_warp
                ][value_base + 3];
        }

        long long output_offset =
            value_matrix_base
            + value_base;

        store_four_values(
            output,
            output_offset,
            final_output
        );
    }
}


struct BenchmarkArguments {
    float* query_fp32;
    float* key_fp32;
    float* value_fp32;
    float* beta_fp32;
    float* output_fp32;
    float* state_fp32;

    __nv_bfloat16* query_bf16;
    __nv_bfloat16* key_bf16;
    __nv_bfloat16* value_bf16;
    __nv_bfloat16* beta_bf16;
    __nv_bfloat16* output_bf16;
    float* state_bf16;

    float* g;
    long long* state_slot_ids;
};


void launch_fp32(
    const BenchmarkArguments& arguments,
    int batch_size)
{
    int blocks = batch_size * NUM_HEADS;

    gdn_splitk_float4<float, float>
        <<<blocks, BLOCK_SIZE>>>(
            arguments.query_fp32,
            arguments.key_fp32,
            arguments.value_fp32,
            arguments.g,
            arguments.beta_fp32,
            arguments.state_slot_ids,
            arguments.state_fp32,
            arguments.output_fp32,
            batch_size,
            0
        );
}


void launch_bf16(
    const BenchmarkArguments& arguments,
    int batch_size)
{
    int blocks = batch_size * NUM_HEADS;

    gdn_splitk_float4<
        __nv_bfloat16,
        __nv_bfloat16
    ><<<blocks, BLOCK_SIZE>>>(
        arguments.query_bf16,
        arguments.key_bf16,
        arguments.value_bf16,
        arguments.g,
        arguments.beta_bf16,
        arguments.state_slot_ids,
        arguments.state_bf16,
        arguments.output_bf16,
        batch_size,
        0
    );
}


float median(std::vector<float> values)
{
    std::sort(
        values.begin(),
        values.end()
    );

    return values[
        values.size() / 2
    ];
}


int get_iterations(int batch_size)
{
    if (batch_size <= 8) {
        return 500;
    }

    if (batch_size <= 64) {
        return 200;
    }

    if (batch_size <= 256) {
        return 100;
    }

    return 50;
}


float measure_backend(
    const BenchmarkArguments& arguments,
    int batch_size,
    size_t active_state_bytes,
    bool use_bf16)
{
    const int warmup_iterations = 10;
    const int repeats = 5;

    int benchmark_iterations =
        get_iterations(batch_size);

    std::vector<float> measurements;

    for (
        int repeat = 0;
        repeat < repeats;
        ++repeat
    ) {
        float* state_pool =
            use_bf16
                ? arguments.state_bf16
                : arguments.state_fp32;

        check_cuda(cudaMemset(
            state_pool,
            0,
            active_state_bytes
        ));

        for (
            int iteration = 0;
            iteration < warmup_iterations;
            ++iteration
        ) {
            if (use_bf16) {
                launch_bf16(
                    arguments,
                    batch_size
                );
            } else {
                launch_fp32(
                    arguments,
                    batch_size
                );
            }
        }

        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());

        check_cuda(cudaMemset(
            state_pool,
            0,
            active_state_bytes
        ));

        check_cuda(cudaDeviceSynchronize());

        cudaEvent_t start_event;
        cudaEvent_t stop_event;

        check_cuda(cudaEventCreate(
            &start_event
        ));

        check_cuda(cudaEventCreate(
            &stop_event
        ));

        check_cuda(cudaEventRecord(
            start_event
        ));

        for (
            int iteration = 0;
            iteration < benchmark_iterations;
            ++iteration
        ) {
            if (use_bf16) {
                launch_bf16(
                    arguments,
                    batch_size
                );
            } else {
                launch_fp32(
                    arguments,
                    batch_size
                );
            }
        }

        check_cuda(cudaEventRecord(
            stop_event
        ));

        check_cuda(cudaEventSynchronize(
            stop_event
        ));

        check_cuda(cudaGetLastError());

        float elapsed_ms = 0.0f;

        check_cuda(cudaEventElapsedTime(
            &elapsed_ms,
            start_event,
            stop_event
        ));

        check_cuda(cudaEventDestroy(
            start_event
        ));

        check_cuda(cudaEventDestroy(
            stop_event
        ));

        float average_microseconds =
            elapsed_ms
            * 1000.0f
            / static_cast<float>(
                benchmark_iterations
            );

        measurements.push_back(
            average_microseconds
        );
    }

    return median(measurements);
}


int main()
{
    const int max_batch_size = 512;

    const int batch_sizes[] = {
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512
    };

    size_t qk_elements =
        static_cast<size_t>(max_batch_size)
        * NUM_HEADS
        * KEY_DIM;

    size_t value_elements =
        static_cast<size_t>(max_batch_size)
        * NUM_HEADS
        * VALUE_DIM;

    size_t gate_elements =
        static_cast<size_t>(max_batch_size)
        * NUM_HEADS;

    size_t state_elements =
        static_cast<size_t>(max_batch_size)
        * NUM_LAYERS
        * NUM_HEADS
        * KEY_DIM
        * VALUE_DIM;

    size_t qk_fp32_bytes =
        qk_elements
        * sizeof(float);

    size_t qk_bf16_bytes =
        qk_elements
        * sizeof(__nv_bfloat16);

    size_t value_fp32_bytes =
        value_elements
        * sizeof(float);

    size_t value_bf16_bytes =
        value_elements
        * sizeof(__nv_bfloat16);

    size_t gate_fp32_bytes =
        gate_elements
        * sizeof(float);

    size_t gate_bf16_bytes =
        gate_elements
        * sizeof(__nv_bfloat16);

    size_t state_bytes =
        state_elements
        * sizeof(float);

    size_t slot_bytes =
        static_cast<size_t>(max_batch_size)
        * sizeof(long long);

    std::vector<float> host_query_fp32(
        qk_elements
    );

    std::vector<float> host_key_fp32(
        qk_elements
    );

    std::vector<float> host_value_fp32(
        value_elements
    );

    std::vector<float> host_beta_fp32(
        gate_elements
    );

    std::vector<float> host_g(
        gate_elements
    );

    std::vector<__nv_bfloat16> host_query_bf16(
        qk_elements
    );

    std::vector<__nv_bfloat16> host_key_bf16(
        qk_elements
    );

    std::vector<__nv_bfloat16> host_value_bf16(
        value_elements
    );

    std::vector<__nv_bfloat16> host_beta_bf16(
        gate_elements
    );

    std::vector<long long> host_state_slot_ids(
        max_batch_size
    );

    for (
        size_t index = 0;
        index < qk_elements;
        ++index
    ) {
        float query_value =
            0.02f
            * std::sin(
                static_cast<float>(index)
                * 0.013f
            );

        float key_value =
            0.02f
            * std::cos(
                static_cast<float>(index)
                * 0.017f
            );

        host_query_fp32[index] =
            query_value;

        host_key_fp32[index] =
            key_value;

        host_query_bf16[index] =
            __float2bfloat16_rn(
                query_value
            );

        host_key_bf16[index] =
            __float2bfloat16_rn(
                key_value
            );
    }

    for (
        size_t index = 0;
        index < value_elements;
        ++index
    ) {
        float current_value =
            0.1f
            * std::sin(
                static_cast<float>(index)
                * 0.007f
            );

        host_value_fp32[index] =
            current_value;

        host_value_bf16[index] =
            __float2bfloat16_rn(
                current_value
            );
    }

    for (
        size_t index = 0;
        index < gate_elements;
        ++index
    ) {
        float beta_value =
            0.25f
            + 0.01f
                * static_cast<float>(
                    index % 11
                );

        host_beta_fp32[index] =
            beta_value;

        host_beta_bf16[index] =
            __float2bfloat16_rn(
                beta_value
            );

        host_g[index] =
            -0.05f
            - 0.001f
                * static_cast<float>(
                    index % 17
                );
    }

    for (
        int batch_index = 0;
        batch_index < max_batch_size;
        ++batch_index
    ) {
        host_state_slot_ids[batch_index] =
            batch_index;
    }

    BenchmarkArguments arguments{};

    check_cuda(cudaMalloc(
        &arguments.query_fp32,
        qk_fp32_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.key_fp32,
        qk_fp32_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.value_fp32,
        value_fp32_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.beta_fp32,
        gate_fp32_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.output_fp32,
        value_fp32_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.state_fp32,
        state_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.query_bf16,
        qk_bf16_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.key_bf16,
        qk_bf16_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.value_bf16,
        value_bf16_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.beta_bf16,
        gate_bf16_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.output_bf16,
        value_bf16_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.state_bf16,
        state_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.g,
        gate_fp32_bytes
    ));

    check_cuda(cudaMalloc(
        &arguments.state_slot_ids,
        slot_bytes
    ));

    check_cuda(cudaMemcpy(
        arguments.query_fp32,
        host_query_fp32.data(),
        qk_fp32_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        arguments.key_fp32,
        host_key_fp32.data(),
        qk_fp32_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        arguments.value_fp32,
        host_value_fp32.data(),
        value_fp32_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        arguments.beta_fp32,
        host_beta_fp32.data(),
        gate_fp32_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        arguments.query_bf16,
        host_query_bf16.data(),
        qk_bf16_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        arguments.key_bf16,
        host_key_bf16.data(),
        qk_bf16_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        arguments.value_bf16,
        host_value_bf16.data(),
        value_bf16_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        arguments.beta_bf16,
        host_beta_bf16.data(),
        gate_bf16_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        arguments.g,
        host_g.data(),
        gate_fp32_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        arguments.state_slot_ids,
        host_state_slot_ids.data(),
        slot_bytes,
        cudaMemcpyHostToDevice
    ));

    cudaDeviceProp properties;

    check_cuda(cudaGetDeviceProperties(
        &properties,
        0
    ));

    std::printf(
        "GPU: %s\n",
        properties.name
    );

    std::printf(
        "Shape: H=%d, Dk=%d, Dv=%d\n",
        NUM_HEADS,
        KEY_DIM,
        VALUE_DIM
    );

    std::printf(
        "State and accumulation: FP32\n"
    );

    std::printf(
        "Maximum state pool per backend: %.2f GiB\n",
        static_cast<double>(state_bytes)
            / 1024.0
            / 1024.0
            / 1024.0
    );

    std::printf(
        "Each result is the median of 5 measurements.\n\n"
    );

    std::printf(
        "%6s | %14s | %14s | %10s | %10s\n",
        "B",
        "FP32 us",
        "BF16 us",
        "change",
        "speedup"
    );

    std::printf(
        "-------+----------------+----------------+------------+------------\n"
    );

    for (int batch_size : batch_sizes) {
        size_t active_state_elements =
            static_cast<size_t>(batch_size)
            * NUM_LAYERS
            * NUM_HEADS
            * KEY_DIM
            * VALUE_DIM;

        size_t active_state_bytes =
            active_state_elements
            * sizeof(float);

        float fp32_microseconds =
            measure_backend(
                arguments,
                batch_size,
                active_state_bytes,
                false
            );

        float bf16_microseconds =
            measure_backend(
                arguments,
                batch_size,
                active_state_bytes,
                true
            );

        float change_percent =
            (
                bf16_microseconds
                / fp32_microseconds
                - 1.0f
            ) * 100.0f;

        float speedup =
            fp32_microseconds
            / bf16_microseconds;

        std::printf(
            "%6d | %14.3f | %14.3f | %+9.2f%% | %9.3fx\n",
            batch_size,
            fp32_microseconds,
            bf16_microseconds,
            change_percent,
            speedup
        );
    }

    check_cuda(cudaFree(
        arguments.query_fp32
    ));

    check_cuda(cudaFree(
        arguments.key_fp32
    ));

    check_cuda(cudaFree(
        arguments.value_fp32
    ));

    check_cuda(cudaFree(
        arguments.beta_fp32
    ));

    check_cuda(cudaFree(
        arguments.output_fp32
    ));

    check_cuda(cudaFree(
        arguments.state_fp32
    ));

    check_cuda(cudaFree(
        arguments.query_bf16
    ));

    check_cuda(cudaFree(
        arguments.key_bf16
    ));

    check_cuda(cudaFree(
        arguments.value_bf16
    ));

    check_cuda(cudaFree(
        arguments.beta_bf16
    ));

    check_cuda(cudaFree(
        arguments.output_bf16
    ));

    check_cuda(cudaFree(
        arguments.state_bf16
    ));

    check_cuda(cudaFree(
        arguments.g
    ));

    check_cuda(cudaFree(
        arguments.state_slot_ids
    ));

    return EXIT_SUCCESS;
}