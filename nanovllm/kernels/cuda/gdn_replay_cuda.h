#pragma once

#ifndef GDN_REPLAY_NVRTC
#include <cuda_runtime.h>
#endif

namespace nanovllm_replay {
using index_t = long long;
struct Stride3 { index_t s0, s1, s2; };
struct Stride4 { index_t s0, s1, s2, s3; };

// All strides are in elements. Checkpoints may be State Pool views.
struct ReplayParams {
    const void *key, *value, *beta, *conv_input, *conv_state;
    const float *g, *state;
    void *conv_output;
    float *state_output;
    Stride4 key_stride, value_stride, state_stride;
    Stride3 g_stride, beta_stride, input_stride, conv_stride;
    int batch, tokens, heads, key_dim, value_dim, channels, window;
};

struct VerifyParams {
    ReplayParams replay;
    const void* query;
    Stride4 query_stride;
    float* token_output;
    const void *conv_weight, *conv_bias;
    void* conv_token_output;
    int activation;
};

#ifndef GDN_REPLAY_NVRTC
// dtype code: 0=FP32, 1=FP16, 2=BF16.
void launch_gdn_replay(const ReplayParams& params, int input_dtype,
                       int beta_dtype, int conv_dtype, cudaStream_t stream);
void launch_gdn_recurrent_verify(const VerifyParams&, int input_dtype, int beta_dtype, cudaStream_t);
void launch_gdn_conv_verify(const VerifyParams&, int input_dtype, cudaStream_t);
#endif
}  // namespace nanovllm_replay
