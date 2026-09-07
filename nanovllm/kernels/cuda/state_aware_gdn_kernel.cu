#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>


constexpr int BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int VALUES_PER_LANE = 4;

constexpr int NUM_HEADS = 32;
constexpr int KEY_DIM = 128;
constexpr int VALUE_DIM = 128;


/*
4 个 BF16 元素：

    4 × 2 bytes = 8 bytes
*/
struct alignas(8) BFloat16x4 {
    __nv_bfloat162 xy;
    __nv_bfloat162 zw;
};


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
一个 Block 处理一个：

    batch × head

一个 Block 有 128 个线程，也就是 4 个 Warp。

warp_index 决定当前 Warp 负责的 Dk 范围：

    Warp 0：Dk 0～31
    Warp 1：Dk 32～63
    Warp 2：Dk 64～95
    Warp 3：Dk 96～127

lane_index 决定当前线程负责的 4 个 Dv：

    Lane 0：Dv 0～3
    Lane 1：Dv 4～7
    ...
    Lane 31：Dv 124～127
*/
__global__ void state_aware_gdn_bf16_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float* g,
    const __nv_bfloat16* beta,
    const int64_t* state_slot_ids,
    float* recurrent_state_pool,
    __nv_bfloat16* output,
    int batch_size,
    int num_gdn_layers,
    int gdn_index,
    float scale,
    int64_t query_batch_stride,
    int64_t query_head_stride,
    int64_t key_batch_stride,
    int64_t key_head_stride,
    int64_t value_batch_stride,
    int64_t value_head_stride,
    int64_t g_batch_stride,
    int64_t g_head_stride,
    int64_t beta_batch_stride,
    int64_t beta_head_stride,
    int64_t output_batch_stride,
    int64_t output_head_stride)
{
    int batch_head_index = blockIdx.x;
    int batch_index = batch_head_index / NUM_HEADS;
    int head_index = batch_head_index % NUM_HEADS;

    int thread_index = threadIdx.x;
    int lane_index = thread_index % WARP_SIZE;
    int warp_index = thread_index / WARP_SIZE;

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

    /*
    原始 Shape 是 [B, 1, H, Dk]。

    L=1 且 Tensor 连续，所以底层可以按 [B, H, Dk] 访问。
    */
    int64_t query_head_base = static_cast<int64_t>(batch_index) * query_batch_stride + static_cast<int64_t>(head_index) * query_head_stride;
    int64_t key_head_base = static_cast<int64_t>(batch_index) * key_batch_stride + static_cast<int64_t>(head_index) * key_head_stride;

    int64_t g_offset = static_cast<int64_t>(batch_index) * g_batch_stride + static_cast<int64_t>(head_index) * g_head_stride;
    int64_t beta_offset = static_cast<int64_t>(batch_index) * beta_batch_stride + static_cast<int64_t>(head_index) * beta_head_stride;

    float query_value = __bfloat162float(query[query_head_base + thread_index]);
    float key_value = __bfloat162float(key[key_head_base + thread_index]);

    shared_query[thread_index] = query_value;
    shared_key[thread_index] = key_value;

    /*
    第一层归约：每个 Warp 归约自己负责的 32 个 Q/K 元素。
    */
    float query_squared_sum = warp_reduce_sum(query_value * query_value);
    float key_squared_sum = warp_reduce_sum(key_value * key_value);

    if (lane_index == 0) {
        shared_query_warp_sums[warp_index] = query_squared_sum;
        shared_key_warp_sums[warp_index] = key_squared_sum;
    }

    __syncthreads();

    /*
    第二层归约：Warp 0 合并四个 Warp 的结果。
    */
    if (warp_index == 0) {
        float query_block_sum = lane_index < NUM_WARPS ? shared_query_warp_sums[lane_index] : 0.0f;
        float key_block_sum = lane_index < NUM_WARPS ? shared_key_warp_sums[lane_index] : 0.0f;

        query_block_sum = warp_reduce_sum(query_block_sum);
        key_block_sum = warp_reduce_sum(key_block_sum);

        if (lane_index == 0) {
            shared_query_multiplier = rsqrtf(query_block_sum + 1e-6f) * scale;
            shared_key_multiplier = rsqrtf(key_block_sum + 1e-6f);
            shared_decay = expf(g[g_offset]);
            shared_beta = __bfloat162float(beta[beta_offset]);
        }
    }

    __syncthreads();

    shared_query[thread_index] = query_value * shared_query_multiplier;
    shared_key[thread_index] = key_value * shared_key_multiplier;

    __syncthreads();

    /*
    根据 batch row 找到长期 State Pool 中的物理 slot。
    */
    int64_t state_slot = state_slot_ids[batch_index];

    /*
    定位：

        recurrent_state_pool[
            state_slot,
            gdn_index,
            head_index,
            0,
            0
        ]
    */
    int64_t state_matrix_base =
        state_slot * num_gdn_layers * NUM_HEADS * KEY_DIM * VALUE_DIM
        + static_cast<int64_t>(gdn_index) * NUM_HEADS * KEY_DIM * VALUE_DIM
        + static_cast<int64_t>(head_index) * KEY_DIM * VALUE_DIM;

    int64_t value_head_base = static_cast<int64_t>(batch_index) * value_batch_stride + static_cast<int64_t>(head_index) * value_head_stride;
    int64_t output_head_base = static_cast<int64_t>(batch_index) * output_batch_stride + static_cast<int64_t>(head_index) * output_head_stride;
    float remembered_x = 0.0f;
    float remembered_y = 0.0f;
    float remembered_z = 0.0f;
    float remembered_w = 0.0f;

    /*
    第一遍扫描 State：

        S_decay = exp(g) × S_old
        remembered = K^T × S_decay
    */
    for (int key_index = key_begin; key_index < key_end; ++key_index) {
        int64_t state_row_base = state_matrix_base + static_cast<int64_t>(key_index) * VALUE_DIM;

        const float4* state_row = reinterpret_cast<const float4*>(recurrent_state_pool + state_row_base);
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

    shared_remembered[warp_index][value_base] = remembered_x;
    shared_remembered[warp_index][value_base + 1] = remembered_y;
    shared_remembered[warp_index][value_base + 2] = remembered_z;
    shared_remembered[warp_index][value_base + 3] = remembered_w;

    __syncthreads();

    /*
    Warp 0 合并四个 Dk 分块，并计算：

        delta = beta × (value - remembered)
    */
    if (warp_index == 0) {
        float total_remembered_x = 0.0f;
        float total_remembered_y = 0.0f;
        float total_remembered_z = 0.0f;
        float total_remembered_w = 0.0f;

        for (int source_warp = 0; source_warp < NUM_WARPS; ++source_warp) {
            total_remembered_x += shared_remembered[source_warp][value_base];
            total_remembered_y += shared_remembered[source_warp][value_base + 1];
            total_remembered_z += shared_remembered[source_warp][value_base + 2];
            total_remembered_w += shared_remembered[source_warp][value_base + 3];
        }

        const BFloat16x4* value_vector = reinterpret_cast<const BFloat16x4*>(value + value_head_base);
        BFloat16x4 packed_value = value_vector[lane_index];

        float2 value_xy = __bfloat1622float2(packed_value.xy);
        float2 value_zw = __bfloat1622float2(packed_value.zw);

        shared_delta[value_base] = shared_beta * (value_xy.x - total_remembered_x);
        shared_delta[value_base + 1] = shared_beta * (value_xy.y - total_remembered_y);
        shared_delta[value_base + 2] = shared_beta * (value_zw.x - total_remembered_z);
        shared_delta[value_base + 3] = shared_beta * (value_zw.y - total_remembered_w);
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
    第二遍扫描 State：

        S_new = exp(g) × S_old + K × delta
        output = Q^T × S_new
    */
    for (int key_index = key_begin; key_index < key_end; ++key_index) {
        int64_t state_row_base = state_matrix_base + static_cast<int64_t>(key_index) * VALUE_DIM;

        float4* state_row = reinterpret_cast<float4*>(recurrent_state_pool + state_row_base);
        float4 state_values = state_row[lane_index];

        float normalized_key = shared_key[key_index];
        float normalized_query = shared_query[key_index];

        state_values.x = state_values.x * shared_decay + normalized_key * delta_x;
        state_values.y = state_values.y * shared_decay + normalized_key * delta_y;
        state_values.z = state_values.z * shared_decay + normalized_key * delta_z;
        state_values.w = state_values.w * shared_decay + normalized_key * delta_w;

        state_row[lane_index] = state_values;

        output_x += normalized_query * state_values.x;
        output_y += normalized_query * state_values.y;
        output_z += normalized_query * state_values.z;
        output_w += normalized_query * state_values.w;
    }

    shared_output[warp_index][value_base] = output_x;
    shared_output[warp_index][value_base + 1] = output_y;
    shared_output[warp_index][value_base + 2] = output_z;
    shared_output[warp_index][value_base + 3] = output_w;

    __syncthreads();

    /*
    Warp 0 合并四个局部 Output，并转换成 BF16。
    */
    if (warp_index == 0) {
        float final_output_x = 0.0f;
        float final_output_y = 0.0f;
        float final_output_z = 0.0f;
        float final_output_w = 0.0f;

        for (int source_warp = 0; source_warp < NUM_WARPS; ++source_warp) {
            final_output_x += shared_output[source_warp][value_base];
            final_output_y += shared_output[source_warp][value_base + 1];
            final_output_z += shared_output[source_warp][value_base + 2];
            final_output_w += shared_output[source_warp][value_base + 3];
        }

        BFloat16x4* output_vector = reinterpret_cast<BFloat16x4*>(output + output_head_base);
        BFloat16x4 packed_output;
        packed_output.xy = __floats2bfloat162_rn(final_output_x, final_output_y);
        packed_output.zw = __floats2bfloat162_rn(final_output_z, final_output_w);

        output_vector[lane_index] = packed_output;
    }
}


/*
PyTorch 调用的 CUDA Launcher。
*/
torch::Tensor state_aware_gdn_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    torch::Tensor recurrent_state_pool,
    const torch::Tensor& state_slot_ids,
    int64_t gdn_index,
    torch::Tensor output,
    double scale)
{
    /*
    只保留五组必要检查：

        1. CUDA Device
        2. dtype
        3. contiguous
        4. Q/K/V/Gate Shape
        5. State/Output Shape
    */

    TORCH_CHECK(
        query.is_cuda() && key.is_cuda() && value.is_cuda() && g.is_cuda() && beta.is_cuda()
        && recurrent_state_pool.is_cuda() && state_slot_ids.is_cuda() && output.is_cuda(),
        "all inputs must be CUDA tensors"
    );

    int device_index = query.get_device();

    TORCH_CHECK(
        key.get_device() == device_index && value.get_device() == device_index
        && g.get_device() == device_index && beta.get_device() == device_index
        && recurrent_state_pool.get_device() == device_index
        && state_slot_ids.get_device() == device_index && output.get_device() == device_index,
        "all tensors must be on the same CUDA device"
    );

    TORCH_CHECK(
        query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16
        && value.scalar_type() == at::kBFloat16 && beta.scalar_type() == at::kBFloat16
        && output.scalar_type() == at::kBFloat16 && g.scalar_type() == at::kFloat
        && recurrent_state_pool.scalar_type() == at::kFloat
        && state_slot_ids.scalar_type() == at::kLong,
        "expected Q/K/V/Beta/Output=BF16, G/State=FP32 and state_slot_ids=INT64"
    );

    int64_t batch_size = query.size(0);

    TORCH_CHECK(
        query.dim() == 4 && batch_size > 0 && query.size(1) == 1
        && query.size(2) == NUM_HEADS && query.size(3) == KEY_DIM
        && key.sizes() == query.sizes()
        && value.dim() == 4 && value.size(0) == batch_size && value.size(1) == 1
        && value.size(2) == NUM_HEADS && value.size(3) == VALUE_DIM
        && g.dim() == 3 && g.size(0) == batch_size && g.size(1) == 1
        && g.size(2) == NUM_HEADS && beta.sizes() == g.sizes(),
        "expected Q/K=[B,1,32,128], V=[B,1,32,128] and G/Beta=[B,1,32]"
    );

    int64_t num_gdn_layers = recurrent_state_pool.size(1);

    TORCH_CHECK(
        recurrent_state_pool.dim() == 5 && recurrent_state_pool.size(0) > 0
        && num_gdn_layers > 0 && recurrent_state_pool.size(2) == NUM_HEADS
        && recurrent_state_pool.size(3) == KEY_DIM
        && recurrent_state_pool.size(4) == VALUE_DIM
        && state_slot_ids.dim() == 1 && state_slot_ids.size(0) == batch_size
        && output.sizes() == value.sizes()
        && gdn_index >= 0 && gdn_index < num_gdn_layers,
        "invalid State Pool, state_slot_ids, output or gdn_index"
    );

    TORCH_CHECK(
        recurrent_state_pool.is_contiguous() && state_slot_ids.is_contiguous(),
        "recurrent_state_pool and state_slot_ids must be contiguous"
    );

    /*
    切换到 query 所在 GPU。

    不检查 state_slot_ids 的具体值，因为复制到 CPU 会导致同步。
    Scheduler 负责保证 slot 合法且互不重复。
    */
    c10::cuda::CUDAGuard device_guard(query.device());

    /*
    PyTorch 使用 at::BFloat16，
    CUDA Kernel 使用 __nv_bfloat16。

    reinterpret_cast 不复制数据，只改变指针类型。
    */
    const __nv_bfloat16* query_pointer =
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>());

    const __nv_bfloat16* key_pointer =
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>());

    const __nv_bfloat16* value_pointer =
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>());

    const float* g_pointer = g.data_ptr<float>();

    const __nv_bfloat16* beta_pointer =
        reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>());

    const int64_t* state_slot_ids_pointer = state_slot_ids.data_ptr<int64_t>();
    float* recurrent_state_pointer = recurrent_state_pool.data_ptr<float>();

    __nv_bfloat16* output_pointer =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());

    /*
    必须使用 PyTorch 当前 Stream，不能擅自使用默认 Stream。
    */
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream(device_index);

    int blocks = static_cast<int>(batch_size) * NUM_HEADS;

    state_aware_gdn_bf16_kernel<<<blocks, BLOCK_SIZE, 0, current_stream>>>(
        query_pointer,
        key_pointer,
        value_pointer,
        g_pointer,
        beta_pointer,
        state_slot_ids_pointer,
        recurrent_state_pointer,
        output_pointer,
        static_cast<int>(batch_size),
        static_cast<int>(num_gdn_layers),
        static_cast<int>(gdn_index),
        static_cast<float>(scale),
        query.stride(0),
        query.stride(2),
        key.stride(0),
        key.stride(2),
        value.stride(0),
        value.stride(2),
        g.stride(0),
        g.stride(2),
        beta.stride(0),
        beta.stride(2),
        output.stride(0),
        output.stride(2)
    );

    /*
    只检查 Launch 错误，不进行 cudaDeviceSynchronize()。
    */
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}