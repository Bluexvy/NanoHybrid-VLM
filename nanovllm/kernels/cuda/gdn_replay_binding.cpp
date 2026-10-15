#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <optional>
#include <pybind11/stl.h>
#include "gdn_replay_cuda.h"

namespace {
int dtype_code(const torch::Tensor& t) {
    switch (t.scalar_type()) {
        case at::kFloat: return 0;
        case at::kHalf: return 1;
        case at::kBFloat16: return 2;
        default: TORCH_CHECK(false, "Replay supports FP32/FP16/BF16 inputs only");
    }
    return -1;
}
nanovllm_replay::Stride3 stride3(const torch::Tensor& x) {
    return {x.stride(0), x.stride(1), x.stride(2)};
}
nanovllm_replay::Stride4 stride4(const torch::Tensor& x) {
    return {x.stride(0), x.stride(1), x.stride(2), x.stride(3)};
}
}

std::vector<torch::Tensor> gdn_replay(
    const torch::Tensor& key, const torch::Tensor& value, const torch::Tensor& g,
    const torch::Tensor& beta, const torch::Tensor& raw_conv,
    const torch::Tensor& conv, const torch::Tensor& state, int64_t tokens) {
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat && state.dim() == 4,
                "Checkpoint must be a CUDA FP32 [B,H,Dk,Dv] tensor");
    for (const auto& t : {key, value, g, beta, raw_conv, conv}) {
        TORCH_CHECK(t.is_cuda() && t.device() == state.device(), "All replay tensors must share a CUDA device");
        for (const auto s : t.strides()) TORCH_CHECK(s >= 0, "Negative strides are unsupported");
    }
    TORCH_CHECK(key.dim() == 4 && value.dim() == 4 && g.dim() == 3 && beta.dim() == 3,
                "Invalid K/V/g/beta ranks");
    TORCH_CHECK(raw_conv.dim() == 3 && conv.dim() == 3, "Invalid convolution ranks");
    const int64_t B = key.size(0), T = key.size(1), H = key.size(2), K = key.size(3);
    const int64_t V = value.size(3), C = conv.size(1), W = conv.size(2);
    TORCH_CHECK(B > 0 && B <= 65535 && H > 0 && H <= 65535 && K > 0 && K <= 128,
                "Replay requires B/H in [1,65535] and Dk in [1,128]");
    TORCH_CHECK(V > 0 && V <= std::numeric_limits<int>::max() && C > 0
                && C <= std::numeric_limits<int>::max() && W > 0 && W <= std::numeric_limits<int>::max(),
                "Invalid value/channel/window dimensions");
    TORCH_CHECK(tokens >= 1 && tokens <= T && tokens <= 16, "Replay prefix must contain 1..16 captured tokens");
    TORCH_CHECK(value.size(0) == B && value.size(1) == T && value.size(2) == H, "K/V mismatch");
    TORCH_CHECK(g.size(0) == B && g.size(1) == T && g.size(2) == H && g.sizes() == beta.sizes(), "Gate mismatch");
    TORCH_CHECK(state.size(0) == B && state.size(1) == H && state.size(2) == K && state.size(3) == V,
                "Checkpoint shape mismatch");
    TORCH_CHECK(raw_conv.size(0) == B && raw_conv.size(1) == C && raw_conv.size(2) == T
                && conv.size(0) == B, "Convolution shape mismatch");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(), "K/V must share a dtype");
    TORCH_CHECK(conv.scalar_type() == raw_conv.scalar_type(), "Conv input/checkpoint must share a dtype");
    TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be FP32 log decay");
    const int input_type = dtype_code(key), beta_type = dtype_code(beta), conv_type = dtype_code(conv);
    c10::cuda::CUDAGuard guard(state.device());
    auto output_state = torch::empty({B, H, K, V}, state.options());
    auto output_conv = torch::empty({B, C, W}, conv.options());
    nanovllm_replay::ReplayParams p{};
    p.key = key.data_ptr(); p.value = value.data_ptr(); p.beta = beta.data_ptr();
    p.conv_input = raw_conv.data_ptr(); p.conv_state = conv.data_ptr();
    p.g = g.data_ptr<float>(); p.state = state.data_ptr<float>();
    p.conv_output = output_conv.data_ptr(); p.state_output = output_state.data_ptr<float>();
    p.key_stride = stride4(key); p.value_stride = stride4(value); p.state_stride = stride4(state);
    p.g_stride = stride3(g); p.beta_stride = stride3(beta);
    p.input_stride = stride3(raw_conv); p.conv_stride = stride3(conv);
    p.batch = B; p.tokens = tokens; p.heads = H; p.key_dim = K; p.value_dim = V;
    p.channels = C; p.window = W;
    nanovllm_replay::launch_gdn_replay(p, input_type, beta_type, conv_type,
                                     c10::cuda::getCurrentCUDAStream(state.get_device()).stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_conv, output_state};
}

std::vector<torch::Tensor> gdn_recurrent_verify(
    const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& g, const torch::Tensor& beta, const torch::Tensor& state) {
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat && state.dim() == 4,
                "Verification checkpoint must be CUDA FP32 [B,H,K,V]");
    TORCH_CHECK(query.dim() == 4 && key.sizes() == query.sizes() && value.dim() == 4,
                "Invalid verification Q/K/V shape");
    const int64_t B = key.size(0), T = key.size(1), H = key.size(2), K = key.size(3), V = value.size(3);
    TORCH_CHECK(B >= 1 && B <= 65535 && T >= 1 && T <= 16 && H >= 1 && H <= 65535
                && K >= 1 && K <= 128 && V >= 1 && V <= std::numeric_limits<int>::max(),
                "Unsupported verification dimensions");
    TORCH_CHECK(value.size(0) == B && value.size(1) == T && value.size(2) == H
                && state.size(0) == B && state.size(1) == H && state.size(2) == K && state.size(3) == V,
                "Verification value/checkpoint dimensions must align");
    TORCH_CHECK(g.dim() == 3 && g.size(0) == B && g.size(1) == T && g.size(2) == H
                && beta.sizes() == g.sizes() && g.scalar_type() == at::kFloat, "Invalid verification gates");
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && value.scalar_type() == key.scalar_type(),
                "Verification Q/K/V must share a dtype");
    for (const auto& tensor : {query, key, value, g, beta}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == state.device(), "Verification tensors must share a CUDA device");
        for (auto stride : tensor.strides()) TORCH_CHECK(stride >= 0, "Negative strides are unsupported");
    }
    c10::cuda::CUDAGuard guard(state.device());
    auto output = torch::empty({B, T, H, V}, state.options());
    auto updated = torch::empty({B, H, K, V}, state.options());
    nanovllm_replay::VerifyParams vp{};
    auto& p = vp.replay;
    p.key = key.data_ptr(); p.value = value.data_ptr(); p.beta = beta.data_ptr();
    p.g = g.data_ptr<float>(); p.state = state.data_ptr<float>(); p.state_output = updated.data_ptr<float>();
    p.key_stride = stride4(key); p.value_stride = stride4(value); p.state_stride = stride4(state);
    p.g_stride = stride3(g); p.beta_stride = stride3(beta);
    p.batch = B; p.tokens = T; p.heads = H; p.key_dim = K; p.value_dim = V;
    vp.query = query.data_ptr(); vp.query_stride = stride4(query); vp.token_output = output.data_ptr<float>();
    nanovllm_replay::launch_gdn_recurrent_verify(vp, dtype_code(key), dtype_code(beta),
        c10::cuda::getCurrentCUDAStream(state.get_device()).stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, updated};
}

std::vector<torch::Tensor> gdn_conv_verify(
    const torch::Tensor& input, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias, const torch::Tensor& state, bool activation) {
    TORCH_CHECK(input.is_cuda() && input.dim() == 3 && state.dim() == 3, "Invalid verification convolution input");
    const int64_t B = input.size(0), C = input.size(1), T = input.size(2);
    TORCH_CHECK(B >= 1 && B <= 65535 && C >= 1 && C <= std::numeric_limits<int>::max() && T >= 1 && T <= 16,
                "Unsupported verification convolution dimensions");
    TORCH_CHECK(state.size(0) == B && state.size(1) == C && state.size(2) == 4
                && weight.dim() == 2 && weight.size(0) == C && weight.size(1) == 4 && weight.is_contiguous(),
                "Verification convolution requires [C,4] weights and [B,C,4] history");
    for (const auto& tensor : {weight, state})
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == input.device()
                    && tensor.scalar_type() == input.scalar_type(), "Convolution tensors must share device/dtype");
    if (bias.has_value()) TORCH_CHECK(bias->is_cuda() && bias->device() == input.device()
        && bias->scalar_type() == input.scalar_type() && bias->dim() == 1
        && bias->size(0) == C && bias->is_contiguous(), "Invalid convolution bias");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty({B, C, T}, input.options());
    auto updated = torch::empty({B, C, 4}, state.options());
    nanovllm_replay::VerifyParams vp{};
    auto& p = vp.replay;
    p.conv_input = input.data_ptr(); p.conv_state = state.data_ptr(); p.conv_output = updated.data_ptr();
    p.input_stride = stride3(input); p.conv_stride = stride3(state);
    p.batch = B; p.channels = C; p.tokens = T; p.window = 4;
    vp.conv_weight = weight.data_ptr(); vp.conv_bias = bias.has_value() ? bias->data_ptr() : nullptr;
    vp.conv_token_output = output.data_ptr(); vp.activation = activation;
    nanovllm_replay::launch_gdn_conv_verify(vp, dtype_code(input),
        c10::cuda::getCurrentCUDAStream(input.get_device()).stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, updated};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_replay", &gdn_replay, "Read-only checkpoint GDN state replay (CUDA)");
    m.def("gdn_recurrent_verify", &gdn_recurrent_verify, "Fused short-prefix GDN verification (CUDA)");
    m.def("gdn_conv_verify", &gdn_conv_verify, "Fused 4-tap convolution verification (CUDA)");
}
