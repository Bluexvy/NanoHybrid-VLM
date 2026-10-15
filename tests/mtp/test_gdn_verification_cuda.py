"""NVRTC/driver numerical and CUDA Graph checks for fused verification kernels."""
import ctypes as ct
import sys
import torch
from test_gdn_replay_cuda import NVRTCReplay, ReplayParams, Stride3, Stride4, make_case, load

reference = load("verification_reference_device", "nanovllm/kernels/gdn_verification.py")


class VerifyParams(ct.Structure):
    _fields_ = [("replay", ReplayParams), ("query", ct.c_void_p), ("query_stride", Stride4),
                ("token_output", ct.c_void_p), ("conv_weight", ct.c_void_p), ("conv_bias", ct.c_void_p),
                ("conv_token_output", ct.c_void_p), ("activation", ct.c_int)]


def run_case(driver, inputs, initial_conv, initial_state, query, weight, bias, graph_mode):
    allocated, checkpoints = [], []
    stream, graph, executable = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    def allocate(size):
        pointer = ct.c_uint64()
        driver.call("cuMemAlloc_v2", ct.byref(pointer), size)
        allocated.append(pointer.value)
        return pointer.value
    def upload(tensor, checkpoint=False):
        storage = tensor.untyped_storage()
        pointer = allocate(storage.nbytes())
        driver.call("cuMemcpyHtoD_v2", pointer, storage.data_ptr(), storage.nbytes())
        if checkpoint:
            checkpoints.append((pointer, ct.string_at(storage.data_ptr(), storage.nbytes())))
        return pointer + tensor.data_ptr() - storage.data_ptr()
    batch, tokens, heads, key_dim = query.shape
    value_dim, channels = inputs.value.shape[-1], initial_conv.shape[1]
    conv_out = torch.empty(batch, channels, tokens, dtype=initial_conv.dtype)
    conv_state = torch.empty(initial_conv.shape, dtype=initial_conv.dtype)
    recurrent_out = torch.empty(batch, tokens, heads, value_dim)
    recurrent_state = torch.empty(initial_state.shape)
    try:
        vp = VerifyParams()
        p = vp.replay
        for name, tensor in (("key", inputs.key), ("value", inputs.value), ("g", inputs.g),
            ("beta", inputs.beta), ("conv_input", inputs.conv_input), ("conv_state", initial_conv),
            ("state", initial_state)):
            setattr(p, name, upload(tensor, name in {"conv_state", "state"}))
        p.batch, p.tokens, p.heads, p.key_dim, p.value_dim = batch, tokens, heads, key_dim, value_dim
        p.channels, p.window = channels, 4
        for name, tensor in (("key_stride", inputs.key), ("value_stride", inputs.value), ("state_stride", initial_state)):
            setattr(p, name, Stride4(*tensor.stride()))
        for name, tensor in (("g_stride", inputs.g), ("beta_stride", inputs.beta),
            ("input_stride", inputs.conv_input), ("conv_stride", initial_conv)):
            setattr(p, name, Stride3(*tensor.stride()))
        p.state_output = allocate(recurrent_state.numel() * 4)
        p.conv_output = allocate(conv_state.numel() * conv_state.element_size())
        vp.query, vp.query_stride = upload(query), Stride4(*query.stride())
        vp.token_output = allocate(recurrent_out.numel() * 4)
        vp.conv_token_output = allocate(conv_out.numel() * conv_out.element_size())
        vp.conv_weight, vp.conv_bias, vp.activation = upload(weight), upload(bias) if bias is not None else None, 1
        types = {torch.float32: "float", torch.float16: "__half", torch.bfloat16: "__nv_bfloat16"}
        recurrent = f"&nanovllm_replay::verify_recurrent<{types[query.dtype]},{types[inputs.beta.dtype]}>"
        convolution = f"&nanovllm_replay::verify_conv<{types[initial_conv.dtype]}>"
        parameters = (ct.c_void_p * 1)(ct.addressof(vp))
        if graph_mode:
            driver.call("cuStreamCreate", ct.byref(stream), 1)
            driver.call("cuStreamBeginCapture", stream, 0)
        driver.call("cuLaunchKernel", driver.functions[convolution], (batch * channels + 255) // 256,
                    1, 1, 256, 1, 1, 0, stream, parameters, None)
        driver.call("cuLaunchKernel", driver.functions[recurrent], (value_dim + 15) // 16,
                    heads, batch, 128, 1, 1, 0, stream, parameters, None)
        if graph_mode:
            driver.call("cuStreamEndCapture", stream, ct.byref(graph))
            driver.call("cuGraphInstantiate_v2", ct.byref(executable), graph, None, None, 0)
            for _ in range(3):
                driver.call("cuGraphLaunch", executable, stream)
        driver.call("cuCtxSynchronize")
        for tensor, pointer in ((conv_out, vp.conv_token_output), (conv_state, p.conv_output),
                                (recurrent_out, vp.token_output), (recurrent_state, p.state_output)):
            driver.call("cuMemcpyDtoH_v2", tensor.data_ptr(), pointer, tensor.numel() * tensor.element_size())
        for pointer, original in checkpoints:
            actual = ct.create_string_buffer(len(original))
            driver.call("cuMemcpyDtoH_v2", actual, pointer, len(original))
            assert actual.raw == original, "Verification mutated its read-only checkpoint"
        expected_conv, expected_conv_state = reference.verification_conv(inputs.conv_input, weight, bias, initial_conv)
        expected_out, expected_state = reference.verification_recurrent(query, inputs.key, inputs.value,
            inputs.g, inputs.beta, initial_state)
        tolerance = (3e-5, 3e-6) if query.dtype == torch.float32 else (.01, .003)
        torch.testing.assert_close(conv_out, expected_conv, rtol=tolerance[0], atol=tolerance[1])
        torch.testing.assert_close(conv_state, expected_conv_state, rtol=0, atol=0)
        torch.testing.assert_close(recurrent_out.to(query.dtype), expected_out, rtol=tolerance[0], atol=tolerance[1])
        torch.testing.assert_close(recurrent_state, expected_state, rtol=3e-5, atol=3e-6)
    finally:
        if executable.value: driver.call("cuGraphExecDestroy", executable)
        if graph.value: driver.call("cuGraphDestroy", graph)
        if stream.value: driver.call("cuStreamDestroy_v2", stream)
        for pointer in reversed(allocated): driver.call("cuMemFree_v2", pointer)


def main():
    torch.manual_seed(137)
    driver = NVRTCReplay()
    cases = [(dtype, 128, 128, 3, False, None, 4) for dtype in
             (torch.float32, torch.float16, torch.bfloat16)]
    cases += [(torch.bfloat16, 128, 128, 16, False, torch.float32, 4),
              (torch.float16, 17, 19, 3, True, torch.float32, 4),
              (torch.bfloat16, 65, 7, 16, True, None, 4)]
    try:
        for graph_mode in (False, True):
            for case in cases:
                inputs, conv, state = make_case(*case)
                query = torch.randn_like(inputs.key)
                weight = torch.randn(conv.shape[1], 4, dtype=conv.dtype)
                bias = None if graph_mode else torch.randn(conv.shape[1], dtype=conv.dtype)
                run_case(driver, inputs, conv, state, query, weight, bias, graph_mode)
                print(f"PASS verification dtype={case[0]}, prefix={case[3]}, strided={case[4]}, graph={graph_mode}")
        print("12 fused verification device checks passed; PyTorch binding not checked.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
