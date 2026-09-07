#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>


constexpr int block_size = 128;
constexpr int warp_size = 32;
constexpr int num_warps = 4;
constexpr int values_per_lane = 4;

constexpr int num_decode_steps = 8;


/*
四个连续BF16元素一共占：

    4 × 2 bytes = 8 bytes

alignas(8)保证结构起始地址按8字节对齐。
*/
struct alignas(8) BFloat16x4 {
    __nv_bfloat162 xy;
    __nv_bfloat162 zw;
};


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
真实BF16输入、FP32 State版本。

输入：
    query/key: [B,H,Dk] BF16
    value:     [B,H,Dv] BF16
    g:         [B,H] FP32
    beta:      [B,H] BF16
    slot_ids:  [B] int64
    state:     [slots,layers,H,Dk,Dv] FP32

输出：
    output:    [B,H,Dv] BF16

实际Runtime多一个长度为1的Sequence维：
    [B,1,H,D]

因为L=1且Tensor连续，去掉该维度不会改变底层元素排列。
*/
__global__ void gdn_splitk_float4_bf16(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float* g,
    const __nv_bfloat16* beta,
    const long long* state_slot_ids,
    float* state_pool,
    __nv_bfloat16* output,
    int batch_size,
    int num_layers,
    int num_heads,
    int key_dim,
    int value_dim,
    int gdn_index)
{
    int batch_head_index = blockIdx.x;
    int batch_index = batch_head_index / num_heads;
    int head_index = batch_head_index % num_heads;

    int thread_index = threadIdx.x;
    int lane_index = thread_index % warp_size;
    int warp_index = thread_index / warp_size;

    if (batch_index >= batch_size) {
        return;
    }

    int key_begin = warp_index * warp_size;
    int key_end = key_begin + warp_size;
    int value_base = lane_index * values_per_lane;

    __shared__ float shared_query[128];
    __shared__ float shared_key[128];

    __shared__ float shared_query_warp_sums[num_warps];
    __shared__ float shared_key_warp_sums[num_warps];

    __shared__ float shared_query_multiplier;
    __shared__ float shared_key_multiplier;
    __shared__ float shared_decay;
    __shared__ float shared_beta;

    __shared__ float shared_remembered[num_warps][128];
    __shared__ float shared_delta[128];
    __shared__ float shared_output[num_warps][128];

    long long qk_base =
        static_cast<long long>(
            batch_index * num_heads + head_index
        ) * key_dim;

    int gate_offset =
        batch_index * num_heads + head_index;

    /*
    BF16输入在进入计算时立即转换成FP32。

    从这一行开始：
        Q/K归一化
        State计算
        所有乘加和归约

    都使用FP32。
    */
    float query_value =
        __bfloat162float(
            query[qk_base + thread_index]
        );

    float key_value =
        __bfloat162float(
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
            lane_index < num_warps
                ? shared_query_warp_sums[lane_index]
                : 0.0f;

        float key_block_sum =
            lane_index < num_warps
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
                    static_cast<float>(key_dim)
                );

            shared_key_multiplier =
                rsqrtf(key_block_sum + 1e-6f);

            shared_decay =
                expf(g[gate_offset]);

            /*
            Beta输入是BF16，但参与delta计算前转为FP32。
            */
            shared_beta =
                __bfloat162float(
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
        static_cast<long long>(
            batch_index * num_heads + head_index
        ) * value_dim;

    float remembered_x = 0.0f;
    float remembered_y = 0.0f;
    float remembered_z = 0.0f;
    float remembered_w = 0.0f;

    /*
    State仍然是FP32，所以State访问继续使用float4：

        4 × FP32 = 16 bytes
    */
    for (
        int key_index = key_begin;
        key_index < key_end;
        ++key_index
    ) {
        long long state_row_base =
            state_matrix_base
            + static_cast<long long>(key_index)
                * value_dim;

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

    /*
    Warp 0合并四个Dk分块，并读取BF16 Value。

    BFloat16x4一次读取4个连续BF16：
        4 × 2 bytes = 8 bytes
    */
    if (warp_index == 0) {
        float total_remembered_x = 0.0f;
        float total_remembered_y = 0.0f;
        float total_remembered_z = 0.0f;
        float total_remembered_w = 0.0f;

        for (
            int source_warp = 0;
            source_warp < num_warps;
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

        const BFloat16x4* value_vector =
            reinterpret_cast<
                const BFloat16x4*
            >(
                value + value_matrix_base
            );

        BFloat16x4 packed_value =
            value_vector[lane_index];

        float2 value_xy =
            __bfloat1622float2(
                packed_value.xy
            );

        float2 value_zw =
            __bfloat1622float2(
                packed_value.zw
            );

        shared_delta[value_base] =
            shared_beta
            * (
                value_xy.x
                - total_remembered_x
            );

        shared_delta[value_base + 1] =
            shared_beta
            * (
                value_xy.y
                - total_remembered_y
            );

        shared_delta[value_base + 2] =
            shared_beta
            * (
                value_zw.x
                - total_remembered_z
            );

        shared_delta[value_base + 3] =
            shared_beta
            * (
                value_zw.y
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
                * value_dim;

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

    /*
    Warp 0合并FP32 partial output，
    最后才量化为BF16输出。
    */
    if (warp_index == 0) {
        float final_output_x = 0.0f;
        float final_output_y = 0.0f;
        float final_output_z = 0.0f;
        float final_output_w = 0.0f;

        for (
            int source_warp = 0;
            source_warp < num_warps;
            ++source_warp
        ) {
            final_output_x +=
                shared_output[
                    source_warp
                ][value_base];

            final_output_y +=
                shared_output[
                    source_warp
                ][value_base + 1];

            final_output_z +=
                shared_output[
                    source_warp
                ][value_base + 2];

            final_output_w +=
                shared_output[
                    source_warp
                ][value_base + 3];
        }

        BFloat16x4 packed_output{};

        packed_output.xy =
            __floats2bfloat162_rn(
                final_output_x,
                final_output_y
            );

        packed_output.zw =
            __floats2bfloat162_rn(
                final_output_z,
                final_output_w
            );

        BFloat16x4* output_vector =
            reinterpret_cast<BFloat16x4*>(
                output + value_matrix_base
            );

        output_vector[lane_index] =
            packed_output;
    }
}


void cpu_reference_step(
    const std::vector<__nv_bfloat16>& query,
    const std::vector<__nv_bfloat16>& key,
    const std::vector<__nv_bfloat16>& value,
    const std::vector<float>& g,
    const std::vector<__nv_bfloat16>& beta,
    const std::vector<long long>& state_slot_ids,
    std::vector<float>& state_pool,
    std::vector<__nv_bfloat16>& output,
    int batch_size,
    int num_layers,
    int num_heads,
    int key_dim,
    int value_dim,
    int gdn_index)
{
    std::vector<float> normalized_query(key_dim);
    std::vector<float> normalized_key(key_dim);

    for (
        int batch_index = 0;
        batch_index < batch_size;
        ++batch_index
    ) {
        long long state_slot =
            state_slot_ids[batch_index];

        for (
            int head_index = 0;
            head_index < num_heads;
            ++head_index
        ) {
            long long qk_base =
                static_cast<long long>(
                    batch_index
                    * num_heads
                    + head_index
                ) * key_dim;

            long long value_base =
                static_cast<long long>(
                    batch_index
                    * num_heads
                    + head_index
                ) * value_dim;

            int gate_offset =
                batch_index
                    * num_heads
                + head_index;

            float query_squared_sum = 0.0f;
            float key_squared_sum = 0.0f;

            for (
                int key_index = 0;
                key_index < key_dim;
                ++key_index
            ) {
                float query_value =
                    __bfloat162float(
                        query[
                            qk_base
                            + key_index
                        ]
                    );

                float key_value =
                    __bfloat162float(
                        key[
                            qk_base
                            + key_index
                        ]
                    );

                normalized_query[key_index] =
                    query_value;

                normalized_key[key_index] =
                    key_value;

                query_squared_sum +=
                    query_value
                    * query_value;

                key_squared_sum +=
                    key_value
                    * key_value;
            }

            float query_multiplier =
                1.0f
                / std::sqrt(
                    query_squared_sum
                    + 1e-6f
                )
                / std::sqrt(
                    static_cast<float>(
                        key_dim
                    )
                );

            float key_multiplier =
                1.0f
                / std::sqrt(
                    key_squared_sum
                    + 1e-6f
                );

            for (
                int key_index = 0;
                key_index < key_dim;
                ++key_index
            ) {
                normalized_query[key_index] *=
                    query_multiplier;

                normalized_key[key_index] *=
                    key_multiplier;
            }

            float decay =
                std::exp(
                    g[gate_offset]
                );

            float beta_value =
                __bfloat162float(
                    beta[gate_offset]
                );

            long long state_matrix_base =
                state_slot
                    * num_layers
                    * num_heads
                    * key_dim
                    * value_dim
                + static_cast<long long>(
                    gdn_index
                )
                    * num_heads
                    * key_dim
                    * value_dim
                + static_cast<long long>(
                    head_index
                )
                    * key_dim
                    * value_dim;

            for (
                int value_index = 0;
                value_index < value_dim;
                ++value_index
            ) {
                float remembered_value = 0.0f;

                for (
                    int key_index = 0;
                    key_index < key_dim;
                    ++key_index
                ) {
                    long long state_offset =
                        state_matrix_base
                        + static_cast<long long>(
                            key_index
                        ) * value_dim
                        + value_index;

                    float decayed_state =
                        state_pool[state_offset]
                        * decay;

                    remembered_value +=
                        normalized_key[key_index]
                        * decayed_state;
                }

                float value_fp32 =
                    __bfloat162float(
                        value[
                            value_base
                            + value_index
                        ]
                    );

                float delta =
                    beta_value
                    * (
                        value_fp32
                        - remembered_value
                    );

                float result = 0.0f;

                for (
                    int key_index = 0;
                    key_index < key_dim;
                    ++key_index
                ) {
                    long long state_offset =
                        state_matrix_base
                        + static_cast<long long>(
                            key_index
                        ) * value_dim
                        + value_index;

                    float decayed_state =
                        state_pool[state_offset]
                        * decay;

                    float updated_state =
                        decayed_state
                        + normalized_key[key_index]
                            * delta;

                    state_pool[state_offset] =
                        updated_state;

                    result +=
                        normalized_query[key_index]
                        * updated_state;
                }

                output[
                    value_base
                    + value_index
                ] = __float2bfloat16_rn(
                    result
                );
            }
        }
    }
}


int main()
{
    const int batch_size = 2;
    const int num_slots = 4;
    const int num_layers = 2;
    const int num_heads = 32;
    const int key_dim = 128;
    const int value_dim = 128;
    const int gdn_index = 1;

    std::vector<long long> host_state_slot_ids = {
        3,
        1
    };

    size_t qk_elements =
        static_cast<size_t>(batch_size)
        * num_heads
        * key_dim;

    size_t value_elements =
        static_cast<size_t>(batch_size)
        * num_heads
        * value_dim;

    size_t gate_elements =
        static_cast<size_t>(batch_size)
        * num_heads;

    size_t state_elements =
        static_cast<size_t>(num_slots)
        * num_layers
        * num_heads
        * key_dim
        * value_dim;

    std::vector<__nv_bfloat16> host_query(
        qk_elements
    );

    std::vector<__nv_bfloat16> host_key(
        qk_elements
    );

    std::vector<__nv_bfloat16> host_value(
        value_elements
    );

    std::vector<float> host_g(
        gate_elements
    );

    std::vector<__nv_bfloat16> host_beta(
        gate_elements
    );

    std::vector<float> initial_state(
        state_elements
    );

    std::vector<float> reference_state(
        state_elements
    );

    std::vector<float> cuda_state(
        state_elements
    );

    std::vector<__nv_bfloat16> reference_output(
        value_elements
    );

    std::vector<__nv_bfloat16> cuda_output(
        value_elements
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

        host_query[index] =
            __float2bfloat16_rn(
                query_value
            );

        host_key[index] =
            __float2bfloat16_rn(
                key_value
            );
    }

    for (
        size_t index = 0;
        index < value_elements;
        ++index
    ) {
        float value_value =
            0.1f
            * std::sin(
                static_cast<float>(index)
                * 0.007f
            );

        host_value[index] =
            __float2bfloat16_rn(
                value_value
            );
    }

    for (
        size_t index = 0;
        index < gate_elements;
        ++index
    ) {
        host_g[index] =
            -0.05f
            - 0.001f
                * static_cast<float>(
                    index % 17
                );

        float beta_value =
            0.25f
            + 0.01f
                * static_cast<float>(
                    index % 11
                );

        host_beta[index] =
            __float2bfloat16_rn(
                beta_value
            );
    }

    /*
    使用非零初始State。

    这样第一步就会真正计算：
        remembered = k^T S
    而不是因为S=0得到平凡结果。
    */
    for (
        size_t index = 0;
        index < state_elements;
        ++index
    ) {
        initial_state[index] =
            0.001f
            * std::sin(
                static_cast<float>(
                    index % 100003
                )
                * 0.0011f
            );
    }

    reference_state = initial_state;

    for (
        int decode_step = 0;
        decode_step < num_decode_steps;
        ++decode_step
    ) {
        cpu_reference_step(
            host_query,
            host_key,
            host_value,
            host_g,
            host_beta,
            host_state_slot_ids,
            reference_state,
            reference_output,
            batch_size,
            num_layers,
            num_heads,
            key_dim,
            value_dim,
            gdn_index
        );
    }

    __nv_bfloat16* device_query = nullptr;
    __nv_bfloat16* device_key = nullptr;
    __nv_bfloat16* device_value = nullptr;
    float* device_g = nullptr;
    __nv_bfloat16* device_beta = nullptr;
    long long* device_state_slot_ids = nullptr;
    float* device_state_pool = nullptr;
    __nv_bfloat16* device_output = nullptr;

    size_t qk_bytes =
        qk_elements
        * sizeof(__nv_bfloat16);

    size_t value_bytes =
        value_elements
        * sizeof(__nv_bfloat16);

    size_t gate_fp32_bytes =
        gate_elements
        * sizeof(float);

    size_t gate_bf16_bytes =
        gate_elements
        * sizeof(__nv_bfloat16);

    size_t slot_bytes =
        host_state_slot_ids.size()
        * sizeof(long long);

    size_t state_bytes =
        state_elements
        * sizeof(float);

    check_cuda(cudaMalloc(
        &device_query,
        qk_bytes
    ));

    check_cuda(cudaMalloc(
        &device_key,
        qk_bytes
    ));

    check_cuda(cudaMalloc(
        &device_value,
        value_bytes
    ));

    check_cuda(cudaMalloc(
        &device_g,
        gate_fp32_bytes
    ));

    check_cuda(cudaMalloc(
        &device_beta,
        gate_bf16_bytes
    ));

    check_cuda(cudaMalloc(
        &device_state_slot_ids,
        slot_bytes
    ));

    check_cuda(cudaMalloc(
        &device_state_pool,
        state_bytes
    ));

    check_cuda(cudaMalloc(
        &device_output,
        value_bytes
    ));

    check_cuda(cudaMemcpy(
        device_query,
        host_query.data(),
        qk_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_key,
        host_key.data(),
        qk_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_value,
        host_value.data(),
        value_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_g,
        host_g.data(),
        gate_fp32_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_beta,
        host_beta.data(),
        gate_bf16_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_state_slot_ids,
        host_state_slot_ids.data(),
        slot_bytes,
        cudaMemcpyHostToDevice
    ));

    check_cuda(cudaMemcpy(
        device_state_pool,
        initial_state.data(),
        state_bytes,
        cudaMemcpyHostToDevice
    ));

    int blocks =
        batch_size
        * num_heads;

    int threads =
        block_size;

    for (
        int decode_step = 0;
        decode_step < num_decode_steps;
        ++decode_step
    ) {
        gdn_splitk_float4_bf16<<<
            blocks,
            threads
        >>>(
            device_query,
            device_key,
            device_value,
            device_g,
            device_beta,
            device_state_slot_ids,
            device_state_pool,
            device_output,
            batch_size,
            num_layers,
            num_heads,
            key_dim,
            value_dim,
            gdn_index
        );
    }

    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    check_cuda(cudaMemcpy(
        cuda_state.data(),
        device_state_pool,
        state_bytes,
        cudaMemcpyDeviceToHost
    ));

    check_cuda(cudaMemcpy(
        cuda_output.data(),
        device_output,
        value_bytes,
        cudaMemcpyDeviceToHost
    ));

    float max_state_error = 0.0f;
    float max_output_error = 0.0f;
    float max_untouched_state_error = 0.0f;

    for (
        size_t index = 0;
        index < state_elements;
        ++index
    ) {
        float state_error =
            std::fabs(
                reference_state[index]
                - cuda_state[index]
            );

        max_state_error =
            std::max(
                max_state_error,
                state_error
            );
    }

    for (
        size_t index = 0;
        index < value_elements;
        ++index
    ) {
        float reference_value =
            __bfloat162float(
                reference_output[index]
            );

        float cuda_value =
            __bfloat162float(
                cuda_output[index]
            );

        float output_error =
            std::fabs(
                reference_value
                - cuda_value
            );

        max_output_error =
            std::max(
                max_output_error,
                output_error
            );
    }

    /*
    检查未被slot_ids选中的State，以及被选slot中的其他层。

    这些位置必须保持与initial_state完全一致。
    */
    size_t layer_stride =
        static_cast<size_t>(num_heads)
        * key_dim
        * value_dim;

    for (
        int slot_index = 0;
        slot_index < num_slots;
        ++slot_index
    ) {
        for (
            int layer_index = 0;
            layer_index < num_layers;
            ++layer_index
        ) {
            bool selected_state =
                (
                    slot_index == 3
                    || slot_index == 1
                )
                && layer_index == gdn_index;

            if (selected_state) {
                continue;
            }

            size_t state_begin =
                (
                    static_cast<size_t>(
                        slot_index
                    ) * num_layers
                    + layer_index
                ) * layer_stride;

            size_t state_end =
                state_begin
                + layer_stride;

            for (
                size_t index = state_begin;
                index < state_end;
                ++index
            ) {
                float error =
                    std::fabs(
                        cuda_state[index]
                        - initial_state[index]
                    );

                max_untouched_state_error =
                    std::max(
                        max_untouched_state_error,
                        error
                    );
            }
        }
    }

    std::printf(
        "Shape: B=%d, H=%d, Dk=%d, Dv=%d\n",
        batch_size,
        num_heads,
        key_dim,
        value_dim
    );

    std::printf(
        "Input: Q/K/V/Beta=BF16, G=FP32\n"
    );

    std::printf(
        "State/accumulation=FP32, Output=BF16\n"
    );

    std::printf(
        "state_slot_ids: [%lld, %lld]\n",
        host_state_slot_ids[0],
        host_state_slot_ids[1]
    );

    std::printf(
        "gdn_index: %d\n",
        gdn_index
    );

    std::printf(
        "decode steps: %d\n",
        num_decode_steps
    );

    std::printf(
        "max state error: %.9e\n",
        max_state_error
    );

    std::printf(
        "max output error: %.9e\n",
        max_output_error
    );

    std::printf(
        "max untouched state error: %.9e\n",
        max_untouched_state_error
    );

    /*
    State保持FP32，因此要求比BF16 Output更严格。

    Output已经量化成BF16，误差阈值相应更宽。
    */
    bool passed =
        max_state_error <= 1e-5f
        && max_output_error <= 2e-3f
        && max_untouched_state_error == 0.0f;

    if (passed) {
        std::printf(
            "Part BF16-1 PASSED\n"
        );
    } else {
        std::printf(
            "Part BF16-1 FAILED\n"
        );
    }

    check_cuda(cudaFree(device_query));
    check_cuda(cudaFree(device_key));
    check_cuda(cudaFree(device_value));
    check_cuda(cudaFree(device_g));
    check_cuda(cudaFree(device_beta));
    check_cuda(cudaFree(device_state_slot_ids));
    check_cuda(cudaFree(device_state_pool));
    check_cuda(cudaFree(device_output));

    return passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}