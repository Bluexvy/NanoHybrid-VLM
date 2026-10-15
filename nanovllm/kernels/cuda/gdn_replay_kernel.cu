#include "gdn_replay_cuda.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace nanovllm_replay {
constexpr int WARP = 32;
constexpr int WARPS = 4;
constexpr int VALUE_TILE = 4;
constexpr int KEY_ROWS = 4;  // Dk <= 128, four rows per lane.

template <typename T> __device__ __forceinline__ float as_float(T x) { return float(x); }
template <> __device__ __forceinline__ float as_float(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float as_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int step = 16; step; step >>= 1) x += __shfl_xor_sync(0xffffffffu, x, step);
    return x;
}

template <typename Input, typename Gate>
__device__ __forceinline__ void recurrent_body(ReplayParams p, const Input* query,
                                              Stride4 query_stride, float* token_output) {
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;
    const int b = blockIdx.z, h = blockIdx.y;
    const int v0 = (blockIdx.x * WARPS + warp) * VALUE_TILE;
    if (v0 >= p.value_dim) return;  // Uniform within a whole warp.
    const Input* key = static_cast<const Input*>(p.key);
    const Input* value = static_cast<const Input*>(p.value);
    const Gate* beta = static_cast<const Gate*>(p.beta);
    const float* base = p.state + b * p.state_stride.s0 + h * p.state_stride.s1;
    float state[KEY_ROWS][VALUE_TILE];
    const bool vectorized = p.state_stride.s3 == 1 && p.state_stride.s2 % 4 == 0
                            && v0 + VALUE_TILE <= p.value_dim
                            && (reinterpret_cast<unsigned long long>(base + v0) & 15) == 0;
    #pragma unroll
    for (int r = 0; r < KEY_ROWS; ++r) {
        const int k = lane + WARP * r;
        if (k < p.key_dim && vectorized) {
            const float4 x = *reinterpret_cast<const float4*>(base + k * p.state_stride.s2 + v0);
            state[r][0] = x.x; state[r][1] = x.y;
            state[r][2] = x.z; state[r][3] = x.w;
        } else {
            #pragma unroll
            for (int c = 0; c < VALUE_TILE; ++c)
                state[r][c] = k < p.key_dim && v0 + c < p.value_dim
                    ? base[k * p.state_stride.s2 + (v0 + c) * p.state_stride.s3] : 0.f;
        }
    }
    // State stays in registers across the replay prefix: one checkpoint read,
    // one final write. Every warp owns complete K reductions for four V columns.
    for (int t = 0; t < p.tokens; ++t) {
        float keys[KEY_ROWS], square_sum = 0.f;
        #pragma unroll
        for (int r = 0; r < KEY_ROWS; ++r) {
            const int k = lane + WARP * r;
            keys[r] = k < p.key_dim ? as_float(key[b * p.key_stride.s0 + t * p.key_stride.s1
                         + h * p.key_stride.s2 + k * p.key_stride.s3]) : 0.f;
            square_sum += keys[r] * keys[r];
        }
        const float inverse_norm = rsqrtf(warp_sum(square_sum) + 1e-6f);
        const float decay = expf(p.g[b * p.g_stride.s0 + t * p.g_stride.s1 + h * p.g_stride.s2]);
        const float update_gate = as_float(beta[b * p.beta_stride.s0 + t * p.beta_stride.s1
                                                + h * p.beta_stride.s2]);
        float remembered[VALUE_TILE] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int r = 0; r < KEY_ROWS; ++r) {
            keys[r] *= inverse_norm;
            #pragma unroll
            for (int c = 0; c < VALUE_TILE; ++c) {
                state[r][c] *= decay;
                remembered[c] += state[r][c] * keys[r];
            }
        }
        float delta[VALUE_TILE];
        #pragma unroll
        for (int c = 0; c < VALUE_TILE; ++c) {
            const float prediction = warp_sum(remembered[c]);
            const float v = v0 + c < p.value_dim
                ? as_float(value[b * p.value_stride.s0 + t * p.value_stride.s1
                         + h * p.value_stride.s2 + (v0 + c) * p.value_stride.s3]) : 0.f;
            delta[c] = (v - prediction) * update_gate;
        }
        #pragma unroll
        for (int r = 0; r < KEY_ROWS; ++r)
            #pragma unroll
            for (int c = 0; c < VALUE_TILE; ++c) state[r][c] += keys[r] * delta[c];
        if (query) {
            float queries[KEY_ROWS], query_square = 0.f;
            #pragma unroll
            for (int r = 0; r < KEY_ROWS; ++r) {
                const int k = lane + WARP * r;
                queries[r] = k < p.key_dim ? as_float(query[b * query_stride.s0 + t * query_stride.s1
                    + h * query_stride.s2 + k * query_stride.s3]) : 0.f;
                query_square += queries[r] * queries[r];
            }
            const float norm = rsqrtf(warp_sum(query_square) + 1e-6f) * rsqrtf(float(p.key_dim));
            float out[VALUE_TILE] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int r = 0; r < KEY_ROWS; ++r)
                #pragma unroll
                for (int c = 0; c < VALUE_TILE; ++c) out[c] += queries[r] * norm * state[r][c];
            #pragma unroll
            for (int c = 0; c < VALUE_TILE; ++c) {
                const float total = warp_sum(out[c]);
                if (lane == 0 && v0 + c < p.value_dim)
                    token_output[((index_t(b) * p.tokens + t) * p.heads + h) * p.value_dim + v0 + c] = total;
            }
        }
    }
    float* output = p.state_output + (index_t(b) * p.heads + h) * p.key_dim * p.value_dim;
    #pragma unroll
    for (int r = 0; r < KEY_ROWS; ++r) {
        const int k = lane + WARP * r;
        if (k >= p.key_dim) continue;
        float* row = output + index_t(k) * p.value_dim;
        if (v0 + VALUE_TILE <= p.value_dim && p.value_dim % 4 == 0) {
            *reinterpret_cast<float4*>(row + v0) = make_float4(
                state[r][0], state[r][1], state[r][2], state[r][3]);
        } else {
            #pragma unroll
            for (int c = 0; c < VALUE_TILE; ++c)
                if (v0 + c < p.value_dim) row[v0 + c] = state[r][c];
        }
    }
}

template <typename Input, typename Gate>
__global__ void replay_recurrent(ReplayParams p) {
    recurrent_body<Input, Gate>(p, nullptr, Stride4{}, nullptr);
}

template <typename Input, typename Gate>
__global__ void verify_recurrent(VerifyParams p) {
    recurrent_body<Input, Gate>(p.replay, static_cast<const Input*>(p.query), p.query_stride, p.token_output);
}

template <typename Input>
__global__ void verify_conv(VerifyParams vp) {
    const ReplayParams p = vp.replay;
    const index_t row = index_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= index_t(p.batch) * p.channels) return;
    const int b = row / p.channels, c = row % p.channels;
    const Input* history = static_cast<const Input*>(p.conv_state) + b * p.conv_stride.s0 + c * p.conv_stride.s1;
    const Input* raw = static_cast<const Input*>(p.conv_input) + b * p.input_stride.s0 + c * p.input_stride.s1;
    const Input* weights = static_cast<const Input*>(vp.conv_weight) + index_t(c) * 4;
    const Input* bias = static_cast<const Input*>(vp.conv_bias);
    Input* output = static_cast<Input*>(vp.conv_token_output) + row * p.tokens;
    float state[4], weight[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        state[i] = as_float(history[i * p.conv_stride.s2]);
        weight[i] = as_float(weights[i]);
    }
    for (int t = 0; t < p.tokens; ++t) {
        state[0] = state[1]; state[1] = state[2]; state[2] = state[3];
        state[3] = as_float(raw[t * p.input_stride.s2]);
        float result = bias ? as_float(bias[c]) : 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) result += state[i] * weight[i];
        if (vp.activation) result = result / (1.f + expf(-result));
        output[t] = Input(result);
    }
    Input* updated = static_cast<Input*>(p.conv_output) + row * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) updated[i] = Input(state[i]);
}

template <typename Input>
__global__ void replay_conv(ReplayParams p) {
    const index_t row = index_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= index_t(p.batch) * p.channels) return;
    const int b = row / p.channels, c = row % p.channels;
    const Input* initial = static_cast<const Input*>(p.conv_state)
                           + b * p.conv_stride.s0 + c * p.conv_stride.s1;
    const Input* input = static_cast<const Input*>(p.conv_input)
                         + b * p.input_stride.s0 + c * p.input_stride.s1;
    Input* output = static_cast<Input*>(p.conv_output) + row * p.window;
    // Common BF16/FP16 4-tap, one-token path: move four values as 64-bit bits.
    if constexpr (sizeof(Input) == 2) {
        if (p.window == 4 && p.tokens == 1 && p.conv_stride.s2 == 1
                && (reinterpret_cast<unsigned long long>(initial) & 7) == 0) {
            const unsigned long long history = *reinterpret_cast<const unsigned long long*>(initial);
            const unsigned short latest = *reinterpret_cast<const unsigned short*>(input);
            *reinterpret_cast<unsigned long long*>(output) = (history >> 16)
                                       | (static_cast<unsigned long long>(latest) << 48);
            return;
        }
    } else {
        if (p.window == 4 && p.tokens == 1 && p.conv_stride.s2 == 1
                && (reinterpret_cast<unsigned long long>(initial) & 15) == 0) {
            const float4 x = *reinterpret_cast<const float4*>(initial);
            *reinterpret_cast<float4*>(output) = make_float4(x.y, x.z, x.w, *input);
            return;
        }
    }
    for (int i = 0; i < p.window; ++i) {
        const int source = p.tokens + i;
        output[i] = source < p.window ? initial[source * p.conv_stride.s2]
                                     : input[(source - p.window) * p.input_stride.s2];
    }
}

#ifndef GDN_REPLAY_NVRTC
template <typename Input>
void launch_input(const ReplayParams& p, int gate, cudaStream_t stream) {
    const dim3 grid((index_t(p.value_dim) + WARPS * VALUE_TILE - 1) / (WARPS * VALUE_TILE), p.heads, p.batch);
    switch (gate) {
        case 0: replay_recurrent<Input, float><<<grid, WARPS * WARP, 0, stream>>>(p); break;
        case 1: replay_recurrent<Input, __half><<<grid, WARPS * WARP, 0, stream>>>(p); break;
        case 2: replay_recurrent<Input, __nv_bfloat16><<<grid, WARPS * WARP, 0, stream>>>(p); break;
    }
}
void launch_gdn_replay(const ReplayParams& p, int input, int gate, int conv, cudaStream_t stream) {
    switch (input) {
        case 0: launch_input<float>(p, gate, stream); break;
        case 1: launch_input<__half>(p, gate, stream); break;
        case 2: launch_input<__nv_bfloat16>(p, gate, stream); break;
    }
    const int blocks = (index_t(p.batch) * p.channels + 255) / 256;
    switch (conv) {
        case 0: replay_conv<float><<<blocks, 256, 0, stream>>>(p); break;
        case 1: replay_conv<__half><<<blocks, 256, 0, stream>>>(p); break;
        case 2: replay_conv<__nv_bfloat16><<<blocks, 256, 0, stream>>>(p); break;
    }
}

template <typename Input>
void launch_verify_input(const VerifyParams& p, int gate, cudaStream_t stream) {
    const dim3 grid((index_t(p.replay.value_dim) + 15) / 16, p.replay.heads, p.replay.batch);
    switch (gate) {
        case 0: verify_recurrent<Input, float><<<grid, 128, 0, stream>>>(p); break;
        case 1: verify_recurrent<Input, __half><<<grid, 128, 0, stream>>>(p); break;
        case 2: verify_recurrent<Input, __nv_bfloat16><<<grid, 128, 0, stream>>>(p); break;
    }
}
void launch_gdn_recurrent_verify(const VerifyParams& p, int input, int gate, cudaStream_t stream) {
    switch (input) {
        case 0: launch_verify_input<float>(p, gate, stream); break;
        case 1: launch_verify_input<__half>(p, gate, stream); break;
        case 2: launch_verify_input<__nv_bfloat16>(p, gate, stream); break;
    }
}
void launch_gdn_conv_verify(const VerifyParams& p, int input, cudaStream_t stream) {
    const int blocks = (index_t(p.replay.batch) * p.replay.channels + 255) / 256;
    switch (input) {
        case 0: verify_conv<float><<<blocks, 256, 0, stream>>>(p); break;
        case 1: verify_conv<__half><<<blocks, 256, 0, stream>>>(p); break;
        case 2: verify_conv<__nv_bfloat16><<<blocks, 256, 0, stream>>>(p); break;
    }
}
#endif
}  // namespace nanovllm_replay
