"""Small CUDA replay checks, without model weights or throughput benchmarks.

Normal CUDA PyTorch: python tests/mtp/test_gdn_replay_cuda.py
CPU PyTorch + CUDA toolkit/driver: add --nvrtc (checks device kernels only).
"""
import argparse
import ctypes as ct
import importlib.util
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = load("gdn_replay_reference_smoke", "nanovllm/engine/gdn_replay.py")
cuda = load("gdn_replay_cuda_smoke", "nanovllm/kernels/gdn_replay_cuda.py")


class Stride3(ct.Structure):
    _fields_ = [(f"s{i}", ct.c_longlong) for i in range(3)]


class Stride4(ct.Structure):
    _fields_ = [(f"s{i}", ct.c_longlong) for i in range(4)]


class ReplayParams(ct.Structure):
    # Keep in sync with gdn_replay_cuda.h. All strides are element strides.
    _fields_ = ([(name, ct.c_void_p) for name in
                 ("key", "value", "beta", "conv_input", "conv_state", "g", "state",
                  "conv_output", "state_output")]
                + [(name, Stride4) for name in ("key_stride", "value_stride", "state_stride")]
                + [(name, Stride3) for name in ("g_stride", "beta_stride", "input_stride", "conv_stride")]
                + [(name, ct.c_int) for name in
                   ("batch", "tokens", "heads", "key_dim", "value_dim", "channels", "window")])


class NVRTCReplay:
    """Driver-only device check: deliberately does not test the PyTorch binding."""
    def __init__(self):
        toolkit = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
        if os.name == "nt":
            self.dll_dir = os.add_dll_directory(str(toolkit / "bin"))
            candidates = sorted((toolkit / "bin").glob("nvrtc64_*.dll"))
            candidates = [x for x in candidates if ".alt." not in x.name]
            if not candidates:
                raise RuntimeError("NVRTC DLL not found; set CUDA_PATH to the installed toolkit")
            self.nvrtc = ct.CDLL(str(candidates[0]))
            self.driver = ct.WinDLL("nvcuda.dll")
        else:
            self.nvrtc = ct.CDLL(str(toolkit / "lib64" / "libnvrtc.so"))
            self.driver = ct.CDLL("libcuda.so.1")
        self.functions = {}
        self.bind_driver()
        self.call("cuInit", 0)
        device = ct.c_int()
        self.call("cuDeviceGet", ct.byref(device), 0)
        major, minor = ct.c_int(), ct.c_int()
        self.call("cuDeviceComputeCapability", ct.byref(major), ct.byref(minor), device)
        self.context = ct.c_void_p()
        self.call("cuCtxCreate_v2", ct.byref(self.context), 0, device)
        self.module = ct.c_void_p()
        try:
            ptx, names = self.compile(toolkit, major.value * 10 + minor.value)
            self.call("cuModuleLoadData", ct.byref(self.module), ct.cast(ptx, ct.c_void_p))
            for expression, lowered in names.items():
                fn = ct.c_void_p()
                self.call("cuModuleGetFunction", ct.byref(fn), self.module, lowered)
                self.functions[expression] = fn
        except BaseException:
            self.close()
            raise

    def bind_driver(self):
        P = ct.c_void_p
        U = ct.c_uint
        self.signatures = {
            "cuInit": [U], "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
            "cuDeviceComputeCapability": [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.c_int],
            "cuCtxCreate_v2": [ct.POINTER(P), U, ct.c_int], "cuCtxDestroy_v2": [P],
            "cuModuleLoadData": [ct.POINTER(P), P], "cuModuleUnload": [P],
            "cuModuleGetFunction": [ct.POINTER(P), P, ct.c_char_p],
            "cuMemAlloc_v2": [ct.POINTER(ct.c_uint64), ct.c_size_t],
            "cuMemFree_v2": [ct.c_uint64],
            "cuMemcpyHtoD_v2": [ct.c_uint64, P, ct.c_size_t],
            "cuMemcpyDtoH_v2": [P, ct.c_uint64, ct.c_size_t],
            "cuCtxSynchronize": [],
            "cuStreamCreate": [ct.POINTER(P), U], "cuStreamDestroy_v2": [P],
            "cuStreamBeginCapture": [P, ct.c_int], "cuStreamEndCapture": [P, ct.POINTER(P)],
            "cuGraphInstantiate_v2": [ct.POINTER(P), P, P, P, ct.c_size_t],
            "cuGraphLaunch": [P, P], "cuGraphDestroy": [P], "cuGraphExecDestroy": [P],
            "cuLaunchKernel": [P, U, U, U, U, U, U, U, P, ct.POINTER(P), ct.POINTER(P)],
        }
        for name, args in self.signatures.items():
            getattr(self.driver, name).argtypes = args
            getattr(self.driver, name).restype = ct.c_int

    def call(self, name, *args):
        status = getattr(self.driver, name)(*args)
        if status:
            raise RuntimeError(f"CUDA Driver {name} failed: {status}")

    def compile(self, toolkit, arch):
        nv = self.nvrtc
        P = ct.c_void_p
        signatures = {
            "nvrtcCreateProgram": [ct.POINTER(P), ct.c_char_p, ct.c_char_p, ct.c_int, P, P],
            "nvrtcAddNameExpression": [P, ct.c_char_p],
            "nvrtcCompileProgram": [P, ct.c_int, ct.POINTER(ct.c_char_p)],
            "nvrtcGetProgramLogSize": [P, ct.POINTER(ct.c_size_t)], "nvrtcGetProgramLog": [P, P],
            "nvrtcGetPTXSize": [P, ct.POINTER(ct.c_size_t)], "nvrtcGetPTX": [P, P],
            "nvrtcGetLoweredName": [P, ct.c_char_p, ct.POINTER(ct.c_char_p)],
            "nvrtcDestroyProgram": [ct.POINTER(P)],
        }
        for name, args in signatures.items():
            getattr(nv, name).argtypes = args
            getattr(nv, name).restype = ct.c_int

        def check(name, *args):
            status = getattr(nv, name)(*args)
            if status:
                raise RuntimeError(f"{name} failed: {status}")

        source = ROOT / "nanovllm/kernels/cuda/gdn_replay_kernel.cu"
        program = P()
        check("nvrtcCreateProgram", ct.byref(program), source.read_bytes(), source.name.encode(), 0, None, None)
        try:
            types = ("float", "__half", "__nv_bfloat16")
            expressions = [f"&nanovllm_replay::replay_recurrent<{a},{b}>" for a in types for b in types]
            expressions += [f"&nanovllm_replay::replay_conv<{a}>" for a in types]
            expressions += [f"&nanovllm_replay::verify_recurrent<{a},{b}>" for a in types for b in types]
            expressions += [f"&nanovllm_replay::verify_conv<{a}>" for a in types]
            for expression in expressions:
                check("nvrtcAddNameExpression", program, expression.encode())
            options = ["--std=c++17", f"--gpu-architecture=compute_{arch}", "--fmad=false",
                       # Match PyTorch cpp_extension: catch disabled implicit conversions here too.
                       "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                       "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
                       "-DGDN_REPLAY_NVRTC", f"--include-path={toolkit / 'include'}",
                       f"--include-path={source.parent}"]
            options_array = (ct.c_char_p * len(options))(*(x.encode() for x in options))
            status = nv.nvrtcCompileProgram(program, len(options), options_array)
            if status:
                size = ct.c_size_t()
                check("nvrtcGetProgramLogSize", program, ct.byref(size))
                log = ct.create_string_buffer(size.value)
                check("nvrtcGetProgramLog", program, log)
                raise RuntimeError(log.value.decode(errors="replace"))
            names = {}
            for expression in expressions:
                lowered = ct.c_char_p()
                check("nvrtcGetLoweredName", program, expression.encode(), ct.byref(lowered))
                names[expression] = lowered.value
            size = ct.c_size_t()
            check("nvrtcGetPTXSize", program, ct.byref(size))
            ptx = ct.create_string_buffer(size.value)
            check("nvrtcGetPTX", program, ptx)
            return ptx, names
        finally:
            check("nvrtcDestroyProgram", ct.byref(program))

    def run(self, inputs, conv, state, tokens, use_graph=False):
        allocations = []
        checkpoints = []
        stream, graph, executable = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()

        def allocate(size):
            pointer = ct.c_uint64()
            self.call("cuMemAlloc_v2", ct.byref(pointer), size)
            allocations.append(pointer.value)
            return pointer.value

        def upload(x, checkpoint=False):
            storage = x.untyped_storage()
            pointer = allocate(storage.nbytes())
            self.call("cuMemcpyHtoD_v2", pointer, storage.data_ptr(), storage.nbytes())
            if checkpoint:
                checkpoints.append((pointer, ct.string_at(storage.data_ptr(), storage.nbytes())))
            return pointer + x.data_ptr() - storage.data_ptr()

        types = {torch.float32: "float", torch.float16: "__half", torch.bfloat16: "__nv_bfloat16"}
        output_conv, output_state = torch.empty(conv.shape, dtype=conv.dtype), torch.empty(state.shape)
        try:
            p = ReplayParams()
            for name, x in (("key", inputs.key), ("value", inputs.value), ("beta", inputs.beta),
                            ("g", inputs.g), ("conv_input", inputs.conv_input),
                            ("conv_state", conv), ("state", state)):
                setattr(p, name, upload(x, name in ("conv_state", "state")))
            p.conv_output = allocate(output_conv.numel() * output_conv.element_size())
            p.state_output = allocate(output_state.numel() * output_state.element_size())
            for name, x in (("key_stride", inputs.key), ("value_stride", inputs.value), ("state_stride", state)):
                setattr(p, name, Stride4(*x.stride()))
            for name, x in (("g_stride", inputs.g), ("beta_stride", inputs.beta),
                            ("input_stride", inputs.conv_input), ("conv_stride", conv)):
                setattr(p, name, Stride3(*x.stride()))
            p.batch, _, p.heads, p.key_dim = inputs.key.shape
            p.tokens, p.value_dim, p.channels, p.window = tokens, inputs.value.shape[-1], conv.shape[1], conv.shape[-1]
            arguments = (ct.c_void_p * 1)(ct.addressof(p))
            recurrent = f"&nanovllm_replay::replay_recurrent<{types[inputs.key.dtype]},{types[inputs.beta.dtype]}>"
            if use_graph:
                self.call("cuStreamCreate", ct.byref(stream), 1)
                self.call("cuStreamBeginCapture", stream, 0)
            self.call("cuLaunchKernel", self.functions[recurrent], (p.value_dim + 15) // 16,
                      p.heads, p.batch, 128, 1, 1, 0, stream, arguments, None)
            convolution = f"&nanovllm_replay::replay_conv<{types[conv.dtype]}>"
            self.call("cuLaunchKernel", self.functions[convolution], (p.batch * p.channels + 255) // 256,
                      1, 1, 256, 1, 1, 0, stream, arguments, None)
            if use_graph:
                self.call("cuStreamEndCapture", stream, ct.byref(graph))
                self.call("cuGraphInstantiate_v2", ct.byref(executable), graph, None, None, 0)
                self.call("cuGraphLaunch", executable, stream)
                self.call("cuCtxSynchronize")
                self.call("cuMemcpyDtoH_v2", output_state.data_ptr(), p.state_output,
                          output_state.numel() * output_state.element_size())
                _, expected_first = reference.replay_gdn_state(inputs, conv, state, tokens)
                torch.testing.assert_close(output_state, expected_first, rtol=3e-5, atol=3e-6)
                # Change values at the SAME address, replay, and check new math.
                original_storage = inputs.value.untyped_storage()
                backing = torch.empty(0, dtype=inputs.value.dtype).set_(original_storage,
                    0, (original_storage.nbytes() // inputs.value.element_size(),), (1,)).clone()
                changed = backing.as_strided(inputs.value.shape, inputs.value.stride(),
                                             inputs.value.storage_offset())
                changed.mul_(.5)
                base_pointer = p.value - inputs.value.storage_offset() * inputs.value.element_size()
                self.call("cuMemcpyHtoD_v2", base_pointer, backing.data_ptr(), original_storage.nbytes())
                # Pageable HtoD uses the default stream; captured kernels use a
                # nonblocking stream. Establish the cross-stream dependency.
                self.call("cuCtxSynchronize")
                self.call("cuGraphLaunch", executable, stream)
                self.call("cuCtxSynchronize")
                alternative = reference.GDNReplayInputs(inputs.conv_input, inputs.key, changed, inputs.g, inputs.beta)
                _, expected_changed = reference.replay_gdn_state(alternative, conv, state, tokens)
                self.call("cuMemcpyDtoH_v2", output_state.data_ptr(), p.state_output,
                          output_state.numel() * output_state.element_size())
                torch.testing.assert_close(output_state, expected_changed, rtol=3e-5, atol=3e-6)
                # Restore contents, not pointers, for the caller's original check.
                self.call("cuMemcpyHtoD_v2", base_pointer, original_storage.data_ptr(), original_storage.nbytes())
                self.call("cuCtxSynchronize")
                self.call("cuGraphLaunch", executable, stream)
            self.call("cuCtxSynchronize")
            for output, pointer in ((output_conv, p.conv_output), (output_state, p.state_output)):
                self.call("cuMemcpyDtoH_v2", output.data_ptr(), pointer, output.numel() * output.element_size())
            for pointer, before in checkpoints:
                after = ct.create_string_buffer(len(before))
                self.call("cuMemcpyDtoH_v2", after, pointer, len(before))
                assert after.raw == before, "Device kernel mutated its checkpoint"
            return output_conv, output_state
        finally:
            if executable.value:
                self.call("cuGraphExecDestroy", executable)
            if graph.value:
                self.call("cuGraphDestroy", graph)
            if stream.value:
                self.call("cuStreamDestroy_v2", stream)
            for pointer in reversed(allocations):
                self.call("cuMemFree_v2", pointer)

    def close(self):
        if self.module.value:
            self.call("cuModuleUnload", self.module)
        if self.context.value:
            self.call("cuCtxDestroy_v2", self.context)


def make_case(dtype, k_dim, v_dim, tokens, strided, beta_dtype=None, window=4):
    batch, heads, channels = 2, 3, 17

    def data(shape, dtype, scale=1):
        if strided:
            backing = torch.randn(*(tuple(shape[:-1]) + (shape[-1] * 2 + 2,)), dtype=dtype) * scale
            return backing[..., 1:1 + shape[-1] * 2:2]
        return torch.randn(shape, dtype=dtype) * scale

    key = data((batch, tokens, heads, k_dim), dtype)
    value = data((batch, tokens, heads, v_dim), dtype, .2)
    g = -data((batch, tokens, heads), torch.float32).abs()
    beta = data((batch, tokens, heads), beta_dtype or dtype).sigmoid()
    raw = data((batch, channels, tokens), dtype)
    conv = data((batch, channels, window), dtype)
    state = data((batch, heads, k_dim, v_dim), torch.float32, .1)
    return reference.GDNReplayInputs(raw, key, value, g, beta), conv, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvrtc", action="store_true")
    parser.add_argument("--graph", action="store_true", help="Driver graph replay with changing input contents")
    args = parser.parse_args()
    if args.graph and not args.nvrtc:
        parser.error("--graph currently checks the driver path; also pass --nvrtc")
    if not args.nvrtc and not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch unavailable; --nvrtc can check device code with CPU PyTorch")
    torch.manual_seed(2026)
    driver = NVRTCReplay() if args.nvrtc else None
    cases = [(dtype, 128, 128, 1, False, None, 4) for dtype in
             (torch.float32, torch.float16, torch.bfloat16)]
    cases += [(torch.bfloat16, 128, 128, 3, False, torch.float32, 4),
              (torch.float16, 17, 19, 3, True, torch.float32, 4),
              (torch.bfloat16, 65, 7, 6, True, None, 3),
              (torch.float32, 1, 1, 1, True, None, 1),
              (torch.bfloat16, 128, 128, 16, False, torch.float32, 4)]
    try:
        for case in cases:
            inputs, conv, state = make_case(*case)
            expected_conv, expected_state = reference.replay_gdn_state(inputs, conv, state, case[3])
            if driver:
                actual_conv, actual_state = driver.run(inputs, conv, state, case[3], use_graph=args.graph)
            else:
                # Preserve the noncontiguous view strides across transfer.
                def transfer_view(x):
                    output = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="cuda")
                    output.copy_(x)
                    return output
                inputs = reference.GDNReplayInputs(*(transfer_view(x) for x in
                       (inputs.conv_input, inputs.key, inputs.value, inputs.g, inputs.beta)))
                conv, state = transfer_view(conv), transfer_view(state)
                before_conv, before_state = conv.clone(), state.clone()
                with torch.cuda.stream(torch.cuda.current_stream()):
                    actual_conv, actual_state = cuda.replay_gdn_state_cuda(inputs, conv, state, case[3])
                torch.testing.assert_close(conv, before_conv, rtol=0, atol=0)
                torch.testing.assert_close(state, before_state, rtol=0, atol=0)
                actual_conv, actual_state = actual_conv.cpu(), actual_state.cpu()
            torch.testing.assert_close(actual_conv, expected_conv, rtol=0, atol=0)
            torch.testing.assert_close(actual_state, expected_state, rtol=3e-5, atol=3e-6)
            print(f"PASS dtype={case[0]}, K/V={case[1:3]}, prefix={case[3]}, strided={case[4]}, window={case[6]}")
        print(f"{len(cases)} device checks passed ({'NVRTC, binding not tested' if driver else 'PyTorch extension'}).")
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()
