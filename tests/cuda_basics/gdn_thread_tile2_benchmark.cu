#include <cuda_runtime.h>

#include <algorithm>
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
Full Warp 基线：

    一个 CUDA Block 负责一个 (batch, head)
    一个 Block 有 128 个线程
    每个线程负责一个 value_index

因为 Dv = 128，所以：

    thread 0   -> value_index 0
    thread 1   -> value_index 1
    ...
    thread 127 -> value_index 127
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
        static_cast<long long>(
            batch_index * num_heads + head_index
        ) * key_dim;

    int gate_offset =
        batch_index * num_heads + head_index;

    float query_value = q[qk_base + thread_index];
    float key_value = k[qk_base + thread_index];

    shared_query[thread_index] = query_value;
    shared_key[thread_index] = key_value;

    float query_squared_sum =
        warp_reduce_sum(query_value * query_value);

    float key_squared_sum =
        warp_reduce_sum(key_value * key_value);

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

            shared_beta =
                beta[gate_offset];
        }
    }

    __syncthreads();

    shared_query[thread_index] *=
        shared_query_multiplier;

    shared_key[thread_index] *=
        shared_key_multiplier;

    __syncthreads();

    int state_slot =
        state_slot_ids[batch_index];

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
        static_cast<long long>(
            batch_index * num_heads + head_index
        ) * value_dim
        + value_index;

    float remembered_value = 0.0f;

    /*
    第一遍扫描 State：

        S_decay = exp(g) * S_old
        remembered = k^T * S_decay

    当前线程只处理 value_index 对应的一列。
    */
    for (
        int key_index = 0;
        key_index < key_dim;
        ++key_index
    ) {
        long long state_offset =
            state_matrix_base
            + static_cast<long long>(key_index)
                * value_dim
            + value_index;

        float decayed_state =
            state_pool[state_offset]
            * shared_decay;

        remembered_value +=
            shared_key[key_index]
            * decayed_state;
    }

    float delta =
        shared_beta
        * (
            v[value_offset]
            - remembered_value
        );

    float result = 0.0f;

    /*
    第二遍扫描 State：

        S_new = S_decay + k * delta
        output = q^T * S_new
    */
    for (
        int key_index = 0;
        key_index < key_dim;
        ++key_index
    ) {
        long long state_offset =
            state_matrix_base
            + static_cast<long long>(key_index)
                * value_dim
            + value_index;

        float decayed_state =
            state_pool[state_offset]
            * shared_decay;

        float updated_state =
            decayed_state
            + shared_key[key_index]
                * delta;

        state_pool[state_offset] =
            updated_state;

        result +=
            shared_query[key_index]
            * updated_state;
    }

    output[value_offset] = result;
}


/*
Thread Tile 2：

    一个 CUDA Block 仍然负责一个 (batch, head)
    但是一个 Block 只有 64 个线程
    每个线程负责两个 Dv 元素

映射关系：

    thread 0  -> value 0  和 value 64
    thread 1  -> value 1  和 value 65
    ...
    thread 63 -> value 63 和 value 127

所以：

    64 threads * 2 values/thread = 128 values
*/
__global__ void gdn_thread_tile2_fp32(
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
    constexpr int block_size = 64;
    constexpr int warp_size = 32;
    constexpr int num_warps = block_size / warp_size;

    int batch_head_index = blockIdx.x;
    int batch_index = batch_head_index / num_heads;
    int head_index = batch_head_index % num_heads;

    int thread_index = threadIdx.x;

    int first_value_index = thread_index;
    int second_value_index =
        thread_index + block_size;

    int first_key_index = thread_index;
    int second_key_index =
        thread_index + block_size;

    int lane_index =
        thread_index % warp_size;

    int warp_index =
        thread_index / warp_size;

    if (batch_index >= batch_size) {
        return;
    }

    /*
    Q 和 K 的完整长度仍然是 128。

    现在只有 64 个线程，因此每个线程负责加载两个
    Q 元素和两个 K 元素。
    */
    __shared__ float shared_query[128];
    __shared__ float shared_key[128];

    /*
    64 个线程一共有两个 Warp，因此这里只需要保存
    两个 Warp 的局部归约结果。
    */
    __shared__ float shared_query_warp_sums[num_warps];
    __shared__ float shared_key_warp_sums[num_warps];

    __shared__ float shared_query_multiplier;
    __shared__ float shared_key_multiplier;
    __shared__ float shared_decay;
    __shared__ float shared_beta;

    long long qk_base =
        static_cast<long long>(
            batch_index * num_heads + head_index
        ) * key_dim;

    int gate_offset =
        batch_index * num_heads + head_index;

    float first_query_value =
        q[qk_base + first_key_index];

    float second_query_value =
        q[qk_base + second_key_index];

    float first_key_value =
        k[qk_base + first_key_index];

    float second_key_value =
        k[qk_base + second_key_index];

    shared_query[first_key_index] =
        first_query_value;

    shared_query[second_key_index] =
        second_query_value;

    shared_key[first_key_index] =
        first_key_value;

    shared_key[second_key_index] =
        second_key_value;

    /*
    每个线程先在寄存器里合并自己负责的两个元素：

        local_sum = x0^2 + x1^2

    然后每个 Warp 对 32 个 local_sum 做 Shuffle Reduction。
    */
    float query_squared_sum =
        first_query_value
            * first_query_value
        + second_query_value
            * second_query_value;

    float key_squared_sum =
        first_key_value
            * first_key_value
        + second_key_value
            * second_key_value;

    query_squared_sum =
        warp_reduce_sum(query_squared_sum);

    key_squared_sum =
        warp_reduce_sum(key_squared_sum);

    /*
    两个 Warp 的 lane 0 分别写入一个局部结果。
    */
    if (lane_index == 0) {
        shared_query_warp_sums[warp_index] =
            query_squared_sum;

        shared_key_warp_sums[warp_index] =
            key_squared_sum;
    }

    __syncthreads();

    /*
    Warp 0 再将两个 Warp 的局部结果合并。

    这里只有 lane 0 和 lane 1 读取有效值，
    其他 lane 使用 0。
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

            shared_beta =
                beta[gate_offset];
        }
    }

    __syncthreads();

    /*
    每个线程归一化两个 Q/K 元素。
    */
    shared_query[first_key_index] =
        first_query_value
        * shared_query_multiplier;

    shared_query[second_key_index] =
        second_query_value
        * shared_query_multiplier;

    shared_key[first_key_index] =
        first_key_value
        * shared_key_multiplier;

    shared_key[second_key_index] =
        second_key_value
        * shared_key_multiplier;

    __syncthreads();

    int state_slot =
        state_slot_ids[batch_index];

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

    long long output_matrix_base =
        static_cast<long long>(
            batch_index * num_heads + head_index
        ) * value_dim;

    long long first_value_offset =
        output_matrix_base
        + first_value_index;

    long long second_value_offset =
        output_matrix_base
        + second_value_index;

    /*
    一个线程现在需要维护两套中间结果。

    它们一般会进入线程私有寄存器：
        first_remembered_value
        second_remembered_value
    */
    float first_remembered_value = 0.0f;
    float second_remembered_value = 0.0f;

    /*
    第一遍扫描 State。

    对每个 key_index，当前线程读取同一行里的两个元素：

        S[key_index, first_value_index]
        S[key_index, second_value_index]
    */
    for (
        int key_index = 0;
        key_index < key_dim;
        ++key_index
    ) {
        long long state_row_base =
            state_matrix_base
            + static_cast<long long>(key_index)
                * value_dim;

        long long first_state_offset =
            state_row_base
            + first_value_index;

        long long second_state_offset =
            state_row_base
            + second_value_index;

        float first_decayed_state =
            state_pool[first_state_offset]
            * shared_decay;

        float second_decayed_state =
            state_pool[second_state_offset]
            * shared_decay;

        float normalized_key =
            shared_key[key_index];

        first_remembered_value +=
            normalized_key
            * first_decayed_state;

        second_remembered_value +=
            normalized_key
            * second_decayed_state;
    }

    float first_delta =
        shared_beta
        * (
            v[first_value_offset]
            - first_remembered_value
        );

    float second_delta =
        shared_beta
        * (
            v[second_value_offset]
            - second_remembered_value
        );

    float first_result = 0.0f;
    float second_result = 0.0f;

    /*
    第二遍扫描 State，分别更新两列并计算两个输出。
    */
    for (
        int key_index = 0;
        key_index < key_dim;
        ++key_index
    ) {
        long long state_row_base =
            state_matrix_base
            + static_cast<long long>(key_index)
                * value_dim;

        long long first_state_offset =
            state_row_base
            + first_value_index;

        long long second_state_offset =
            state_row_base
            + second_value_index;

        float normalized_key =
            shared_key[key_index];

        float normalized_query =
            shared_query[key_index];

        float first_decayed_state =
            state_pool[first_state_offset]
            * shared_decay;

        float second_decayed_state =
            state_pool[second_state_offset]
            * shared_decay;

        float first_updated_state =
            first_decayed_state
            + normalized_key
                * first_delta;

        float second_updated_state =
            second_decayed_state
            + normalized_key
                * second_delta;

        state_pool[first_state_offset] =
            first_updated_state;

        state_pool[second_state_offset] =
            second_updated_state;

        first_result +=
            normalized_query
            * first_updated_state;

        second_result +=
            normalized_query
            * second_updated_state;
    }

    output[first_value_offset] =
        first_result;

    output[second_value_offset] =
        second_result;
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
    ThreadTile2
};


void launch_kernel(
    KernelKind kernel_kind,
    const KernelArguments& arguments,
    int batch_size,
    int gdn_index)
{
    int blocks =
        batch_size
        * arguments.num_heads;

    if (kernel_kind == KernelKind::FullWarp) {
        int threads = 128;

        gdn_full_warp_shuffle_fp32<<<
            blocks,
            threads
        >>>(
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
        int threads = 64;

        gdn_thread_tile2_fp32<<<
            blocks,
            threads
        >>>(
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
    std::sort(
        values.begin(),
        values.end()
    );

    return values[values.size() / 2];
}


float measure_kernel(
    KernelKind kernel_kind,
    const KernelArguments& arguments,
    int batch_size,
    size_t active_state_bytes)
{
    const int warmup_iterations = 3;

    int benchmark_iterations = 10;

    if (batch_size <= 512) {
        benchmark_iterations = 30;
    } else if (batch_size <= 1024) {
        benchmark_iterations = 20;
    }

    check_cuda(cudaMemset(
        arguments.state_pool,
        0,
        active_state_bytes
    ));

    /*
    这个文件专门测超大 Batch，因此只分配一个 GDN 层的
    State Pool，避免 B=2048 时 24 层 State 超出显存容量。
    */
    for (
        int iteration = 0;
        iteration < warmup_iterations;
        ++iteration
    ) {
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

    return
        elapsed_ms
        * 1000.0f
        / static_cast<float>(
            benchmark_iterations
        );
}


float compare_outputs(
    const KernelArguments& arguments,
    int batch_size,
    size_t active_state_bytes)
{
    size_t output_elements =
        static_cast<size_t>(batch_size)
        * arguments.num_heads
        * arguments.value_dim;

    size_t output_bytes =
        output_elements
        * sizeof(float);

    std::vector<float> full_warp_output(
        output_elements
    );

    std::vector<float> tile2_output(
        output_elements
    );

    check_cuda(cudaMemset(
        arguments.state_pool,
        0,
        active_state_bytes
    ));

    for (
        int gdn_index = 0;
        gdn_index < arguments.num_layers;
        ++gdn_index
    ) {
        launch_kernel(
            KernelKind::FullWarp,
            arguments,
            batch_size,
            gdn_index
        );
    }

    check_cuda(cudaGetLastError());

    check_cuda(cudaMemcpy(
        full_warp_output.data(),
        arguments.output,
        output_bytes,
        cudaMemcpyDeviceToHost
    ));

    check_cuda(cudaMemset(
        arguments.state_pool,
        0,
        active_state_bytes
    ));

    for (
        int gdn_index = 0;
        gdn_index < arguments.num_layers;
        ++gdn_index
    ) {
        launch_kernel(
            KernelKind::ThreadTile2,
            arguments,
            batch_size,
            gdn_index
        );
    }

    check_cuda(cudaGetLastError());

    check_cuda(cudaMemcpy(
        tile2_output.data(),
        arguments.output,
        output_bytes,
        cudaMemcpyDeviceToHost
    ));

    float max_error = 0.0f;

    for (
        size_t index = 0;
        index < output_elements;
        ++index
    ) {
        float error = std::fabs(
            full_warp_output[index]
            - tile2_output[index]
        );

        max_error =
            std::max(max_error, error);
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

    /*
    当前 Thread Tile 2 内核专门针对：

        Dk = 128
        Dv = 128

    写这个检查是为了避免换了模型 Shape 后静默算错。
    */
    if (key_dim != 128 || value_dim != 128) {
        std::fprintf(
            stderr,
            "This experiment requires Dk=128 and Dv=128.\n"
        );

        return EXIT_FAILURE;
    }

    const int batch_sizes[] = {
        512,
        1024,
        2048
    };

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

    std::vector<float> host_q(
        qk_elements
    );

    std::vector<float> host_k(
        qk_elements
    );

    std::vector<float> host_v(
        value_elements
    );

    std::vector<float> host_g(
        gate_elements
    );

    std::vector<float> host_beta(
        gate_elements
    );

    std::vector<int> host_state_slot_ids(
        max_batch_size
    );

    for (
        size_t index = 0;
        index < qk_elements;
        ++index
    ) {
        host_q[index] =
            0.02f
            * std::sin(
                static_cast<float>(index)
                * 0.013f
            );

        host_k[index] =
            0.02f
            * std::cos(
                static_cast<float>(index)
                * 0.017f
            );
    }

    for (
        size_t index = 0;
        index < value_elements;
        ++index
    ) {
        host_v[index] =
            0.1f
            * std::sin(
                static_cast<float>(index)
                * 0.007f
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

        host_beta[index] =
            0.25f
            + 0.01f
                * static_cast<float>(
                    index % 11
                );
    }

    for (
        int index = 0;
        index < max_batch_size;
        ++index
    ) {
        host_state_slot_ids[index] =
            index;
    }

    float* device_q = nullptr;
    float* device_k = nullptr;
    float* device_v = nullptr;
    float* device_g = nullptr;
    float* device_beta = nullptr;
    int* device_state_slot_ids = nullptr;
    float* device_state_pool = nullptr;
    float* device_output = nullptr;

    size_t qk_bytes =
        qk_elements
        * sizeof(float);

    size_t value_bytes =
        value_elements
        * sizeof(float);

    size_t gate_bytes =
        gate_elements
        * sizeof(float);

    size_t slot_bytes =
        static_cast<size_t>(max_batch_size)
        * sizeof(int);

    size_t state_bytes =
        state_elements
        * sizeof(float);

    check_cuda(cudaMalloc(
        &device_q,
        qk_bytes
    ));

    check_cuda(cudaMalloc(
        &device_k,
        qk_bytes
    ));

    check_cuda(cudaMalloc(
        &device_v,
        value_bytes
    ));

    check_cuda(cudaMalloc(
        &device_g,
        gate_bytes
    ));

    check_cuda(cudaMalloc(
        &device_beta,
        gate_bytes
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

    KernelArguments arguments{
        device_q,
        device_k,
        device_v,
        device_g,
        device_beta,
        device_state_slot_ids,
        device_state_pool,
        device_output,
        num_layers,
        num_heads,
        key_dim,
        value_dim
    };

    cudaDeviceProp properties{};

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
        num_heads,
        key_dim,
        value_dim
    );

    std::printf(
        "GDN layers: %d\n",
        num_layers
    );

    std::printf(
        "Maximum state pool: %.2f GiB\n",
        static_cast<double>(state_bytes)
        / 1024.0
        / 1024.0
        / 1024.0
    );

    std::printf(
        "Full Warp: 128 threads, 1 value/thread\n"
    );

    std::printf(
        "Tile 2:    64 threads, 2 values/thread\n"
    );

    std::printf(
        "Each result is the median of %d measurements.\n\n",
        repeats
    );

    std::printf(
        "     B | FullWarp step us | Tile2 step us | FullWarp layer us | Tile2 layer us |    change | speedup | max error\n"
    );

    std::printf(
        "-------+------------------+---------------+-------------------+----------------+-----------+---------+----------\n"
    );

    for (int batch_size : batch_sizes) {
        size_t active_state_elements =
            static_cast<size_t>(batch_size)
            * num_layers
            * num_heads
            * key_dim
            * value_dim;

        size_t active_state_bytes =
            active_state_elements
            * sizeof(float);

        std::vector<float> full_warp_times;
        std::vector<float> tile2_times;

        for (
            int repeat = 0;
            repeat < repeats;
            ++repeat
        ) {
            /*
            交替测试顺序，降低 GPU 温度和频率变化造成的偏差。
            */
            if (repeat % 2 == 0) {
                full_warp_times.push_back(
                    measure_kernel(
                        KernelKind::FullWarp,
                        arguments,
                        batch_size,
                        active_state_bytes
                    )
                );

                tile2_times.push_back(
                    measure_kernel(
                        KernelKind::ThreadTile2,
                        arguments,
                        batch_size,
                        active_state_bytes
                    )
                );
            } else {
                tile2_times.push_back(
                    measure_kernel(
                        KernelKind::ThreadTile2,
                        arguments,
                        batch_size,
                        active_state_bytes
                    )
                );

                full_warp_times.push_back(
                    measure_kernel(
                        KernelKind::FullWarp,
                        arguments,
                        batch_size,
                        active_state_bytes
                    )
                );
            }
        }

        float full_warp_step_us =
            median(full_warp_times);

        float tile2_step_us =
            median(tile2_times);

        float full_warp_layer_us =
            full_warp_step_us
            / static_cast<float>(
                num_layers
            );

        float tile2_layer_us =
            tile2_step_us
            / static_cast<float>(
                num_layers
            );

        double change_percent =
            (
                static_cast<double>(
                    tile2_step_us
                )
                / full_warp_step_us
                - 1.0
            )
            * 100.0;

        double speedup =
            static_cast<double>(
                full_warp_step_us
            )
            / tile2_step_us;

        float max_error =
            compare_outputs(
                arguments,
                batch_size,
                active_state_bytes
            );

        std::printf(
            "%6d | %16.3f | %13.3f | %17.3f | %14.3f | %+8.2f%% | %7.3fx | %.2e\n",
            batch_size,
            full_warp_step_us,
            tile2_step_us,
            full_warp_layer_us,
            tile2_layer_us,
            change_percent,
            speedup,
            max_error
        );
    }

    check_cuda(cudaFree(
        device_q
    ));

    check_cuda(cudaFree(
        device_k
    ));

    check_cuda(cudaFree(
        device_v
    ));

    check_cuda(cudaFree(
        device_g
    ));

    check_cuda(cudaFree(
        device_beta
    ));

    check_cuda(cudaFree(
        device_state_slot_ids
    ));

    check_cuda(cudaFree(
        device_state_pool
    ));

    check_cuda(cudaFree(
        device_output
    ));

    return EXIT_SUCCESS;
}
