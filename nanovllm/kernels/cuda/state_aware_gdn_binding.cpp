#include <torch/extension.h>


/*
这个函数的具体实现放在下一步的 .cu 文件中。

.cpp 文件只需要知道：

    函数叫什么
    接收哪些参数
    返回什么类型

这叫函数声明。
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
    double scale
);


torch::Tensor state_aware_causal_conv1d_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    torch::Tensor conv_state_pool,
    const torch::Tensor& state_slot_ids,
    int64_t gdn_index,
    torch::Tensor output,
    int64_t block_size
);

/*
TORCH_EXTENSION_NAME 由 PyTorch 编译系统自动替换成
Extension 的模块名。

后面 Python 执行：

    extension.state_aware_gdn(...)

最终就会调用上面声明的：

    state_aware_gdn_cuda(...)
*/
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    module.def(
        "state_aware_gdn",
        &state_aware_gdn_cuda,
        R"doc(
State-aware GDN Decode CUDA operator.

Inputs:
    query:
        [B, 1, H, Dk], BF16, CUDA

    key:
        [B, 1, H, Dk], BF16, CUDA

    value:
        [B, 1, H, Dv], BF16, CUDA

    g:
        [B, 1, H], FP32, CUDA

    beta:
        [B, 1, H], BF16, CUDA

    recurrent_state_pool:
        [num_slots, num_gdn_layers, H, Dk, Dv],
        FP32, CUDA

    state_slot_ids:
        [B], INT64, CUDA

    gdn_index:
        Index of the GDN layer in recurrent_state_pool.

    output:
        [B, 1, H, Dv], BF16, CUDA

    scale:
        Query scale, normally Dk ** -0.5.

Side effects:
    recurrent_state_pool is updated in place.
    output is written in place.

Returns:
    output
)doc"
    );
        module.def(
        "state_aware_causal_conv1d",
        &state_aware_causal_conv1d_cuda,
        R"doc(
State-aware causal convolution Decode CUDA operator.

Inputs:
    x:
        [B, C, 1], BF16

    weight:
        [C, 4], BF16

    conv_state_pool:
        [num_slots, num_gdn_layers, C, 4], BF16

    state_slot_ids:
        [B], INT64

    gdn_index:
        Compact GDN layer index.

    output:
        [B, C, 1], BF16

Side effects:
    The selected Conv State Pool rows are updated in place.

Returns:
    output
)doc"
    );
}