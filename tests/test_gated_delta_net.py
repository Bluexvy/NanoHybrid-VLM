import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from nanovllm.layers.gated_delta_net import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNormGated,
    torch_causal_conv1d_reference,
    torch_recurrent_gated_delta_rule,
)

class TestTorchCausalConv1dReference(
    unittest.TestCase
):

    def test_known_causal_convolution(self):
        hidden_states = torch.tensor([[
            [1.0, 2.0, 3.0, 4.0],
        ]])

        weight = torch.tensor([
            [1.0, 10.0, 100.0],
        ])

        output, state = (
            torch_causal_conv1d_reference(
                hidden_states,
                weight,
                activation=None,
            )
        )

        expected_output = torch.tensor([[
            [
                100.0,
                210.0,
                321.0,
                432.0,
            ]
        ]])

        expected_state = torch.tensor([[
            [2.0, 3.0, 4.0]
        ]])

        torch.testing.assert_close(
            output,
            expected_output,
        )

        torch.testing.assert_close(
            state,
            expected_state,
        )
        
    def test_full_sequence_matches_two_chunks(self):
        torch.manual_seed(2026)

        hidden_states = torch.randn(
            2,
            3,
            7,
        )

        weight = torch.randn(
            3,
            4,
        )

        bias = torch.randn(3)

        full_output, full_state = (
            torch_causal_conv1d_reference(
                hidden_states,
                weight,
                bias=bias,
                activation="silu",
            )
        )

        split_position = 3

        first_output, first_state = (
            torch_causal_conv1d_reference(
                hidden_states[:, :, :split_position],
                weight,
                bias=bias,
                activation="silu",
            )
        )

        second_output, second_state = (
            torch_causal_conv1d_reference(
                hidden_states[:, :, split_position:],
                weight,
                bias=bias,
                initial_state=first_state,
                activation="silu",
            )
        )

        chunked_output = torch.cat(
            [first_output, second_output],
            dim=-1,
        )

        torch.testing.assert_close(
            chunked_output,
            full_output,
            rtol=1e-5,
            atol=1e-6,
        )

        torch.testing.assert_close(
            second_state,
            full_state,
        )
        
    def test_full_sequence_matches_token_by_token(self):
        torch.manual_seed(2026)

        hidden_states = torch.randn(
            2,
            3,
            7,
        )

        weight = torch.randn(
            3,
            4,
        )

        bias = torch.randn(3)

        full_output, full_state = (
            torch_causal_conv1d_reference(
                hidden_states,
                weight,
                bias=bias,
                activation="silu",
            )
        )

        state = None
        token_outputs = []

        for token_idx in range(
            hidden_states.shape[-1]
        ):
            token_output, state = (
                torch_causal_conv1d_reference(
                    hidden_states[
                        :,
                        :,
                        token_idx:token_idx + 1,
                    ],
                    weight,
                    bias=bias,
                    initial_state=state,
                    activation="silu",
                )
            )

            token_outputs.append(token_output)

        recurrent_output = torch.cat(
            token_outputs,
            dim=-1,
        )

        torch.testing.assert_close(
            recurrent_output,
            full_output,
            rtol=1e-5,
            atol=1e-6,
        )

        torch.testing.assert_close(
            state,
            full_state,
        )

class TestTorchRecurrentGatedDeltaRule(
    unittest.TestCase
):

    def make_inputs(self):
        torch.manual_seed(2026)

        batch_size = 2
        sequence_length = 7
        num_heads = 3
        key_dim = 4
        value_dim = 5

        query = torch.randn(
            batch_size,
            sequence_length,
            num_heads,
            key_dim,
        )

        key = torch.randn(
            batch_size,
            sequence_length,
            num_heads,
            key_dim,
        )

        value = torch.randn(
            batch_size,
            sequence_length,
            num_heads,
            value_dim,
        )

        # g 必须是负数，使 exp(g) 位于 0 和 1 之间。
        g = -F.softplus(torch.randn(
            batch_size,
            sequence_length,
            num_heads,
        ))

        # beta 位于 0 和 1 之间。
        beta = torch.sigmoid(torch.randn(
            batch_size,
            sequence_length,
            num_heads,
        ))

        return query, key, value, g, beta
    
    def test_full_sequence_matches_token_by_token(self):
        query, key, value, g, beta = (
            self.make_inputs()
        )

        full_output, full_state = (
            torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        )

        state = None
        token_outputs = []

        for token_idx in range(query.shape[1]):
            token_output, state = (
                torch_recurrent_gated_delta_rule(
                    query[:, token_idx:token_idx + 1],
                    key[:, token_idx:token_idx + 1],
                    value[:, token_idx:token_idx + 1],
                    g[:, token_idx:token_idx + 1],
                    beta[:, token_idx:token_idx + 1],
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            )

            token_outputs.append(token_output)

        recurrent_output = torch.cat(
            token_outputs,
            dim=1,
        )

        torch.testing.assert_close(
            recurrent_output,
            full_output,
            rtol=1e-5,
            atol=1e-6,
        )

        torch.testing.assert_close(
            state,
            full_state,
            rtol=1e-5,
            atol=1e-6,
        )
    def test_full_sequence_matches_two_chunks(self):
        query, key, value, g, beta = (
            self.make_inputs()
        )

        full_output, full_state = (
            torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        )

        split_position = 3

        first_output, first_state = (
            torch_recurrent_gated_delta_rule(
                query[:, :split_position],
                key[:, :split_position],
                value[:, :split_position],
                g[:, :split_position],
                beta[:, :split_position],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        )

        second_output, second_state = (
            torch_recurrent_gated_delta_rule(
                query[:, split_position:],
                key[:, split_position:],
                value[:, split_position:],
                g[:, split_position:],
                beta[:, split_position:],
                initial_state=first_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        )

        chunked_output = torch.cat(
            [first_output, second_output],
            dim=1,
        )

        torch.testing.assert_close(
            chunked_output,
            full_output,
            rtol=1e-5,
            atol=1e-6,
        )

        torch.testing.assert_close(
            second_state,
            full_state,
            rtol=1e-5,
            atol=1e-6,
        )
        
    def test_delta_update_corrects_existing_memory(self):
        query = torch.tensor([[
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]])

        key = query.clone()

        value = torch.tensor([[
            [[3.0, 5.0]],
            [[4.0, 1.0]],
        ]])

        # exp(0) = 1，不衰减。
        g = torch.zeros(1, 2, 1)

        # 完全写入当前误差。
        beta = torch.ones(1, 2, 1)

        _, final_state = (
            torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
        )

        expected_state = torch.tensor([[
            [
                [4.0, 1.0],
                [0.0, 0.0],
            ]
        ]])

        torch.testing.assert_close(
            final_state,
            expected_state,
        )
        
    def test_output_and_state_dtypes(self):
        query, key, value, g, beta = (
            self.make_inputs()
        )

        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

        output, state = (
            torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        )

        self.assertEqual(
            output.shape,
            (
                2,
                7,
                3,
                5,
            ),
        )

        self.assertEqual(
            state.shape,
            (
                2,
                3,
                4,
                5,
            ),
        )

        self.assertEqual(
            output.dtype,
            torch.bfloat16,
        )

        self.assertEqual(
            state.dtype,
            torch.float32,
        )
class TestQwen35RMSNormGated(
    unittest.TestCase
):

    def test_matches_manual_formula(self):
        torch.manual_seed(2026)

        norm = Qwen3_5RMSNormGated(
            hidden_size=3,
            eps=1e-6,
        )

        hidden_states = torch.randn(
            2,
            4,
            3,
        )

        gate = torch.randn_like(
            hidden_states
        )

        output = norm(
            hidden_states,
            gate,
        )

        variance = (
            hidden_states.float()
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        expected = (
            hidden_states.float()
            * torch.rsqrt(
                variance + 1e-6
            )
        )

        expected = (
            norm.weight
            * expected
        )

        expected = (
            expected
            * F.silu(gate.float())
        )

        torch.testing.assert_close(
            output,
            expected,
        )

    def test_zero_gate_zeros_output(self):
        norm = Qwen3_5RMSNormGated(
            hidden_size=3
        )

        hidden_states = torch.randn(
            2,
            4,
            3,
        )

        gate = torch.zeros_like(
            hidden_states
        )

        output = norm(
            hidden_states,
            gate,
        )

        torch.testing.assert_close(
            output,
            torch.zeros_like(output),
        )
        
class TestQwen35GatedDeltaNet(
    unittest.TestCase
):

    def make_config(self):
        return SimpleNamespace(
            hidden_size=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=3,
            linear_value_head_dim=3,
            linear_conv_kernel_dim=3,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            layer_types=[
                "linear_attention",
            ],
        )

    def test_parameter_shapes(self):
        layer = Qwen3_5GatedDeltaNet(
            self.make_config(),
            layer_idx=0,
        )

        self.assertEqual(
            layer.in_proj_qkv.weight.shape,
            (18, 8),
        )

        self.assertEqual(
            layer.in_proj_z.weight.shape,
            (6, 8),
        )

        self.assertEqual(
            layer.in_proj_b.weight.shape,
            (2, 8),
        )

        self.assertEqual(
            layer.in_proj_a.weight.shape,
            (2, 8),
        )

        self.assertEqual(
            layer.conv1d.weight.shape,
            (18, 1, 3),
        )

        self.assertEqual(
            layer.norm.weight.shape,
            (3,),
        )

        self.assertEqual(
            layer.out_proj.weight.shape,
            (8, 6),
        )

    def test_full_sequence_matches_two_chunks(self):
        torch.manual_seed(2026)

        layer = Qwen3_5GatedDeltaNet(
            self.make_config(),
            layer_idx=0,
        )

        layer.eval()

        hidden_states = torch.randn(
            2,
            7,
            8,
        )

        (
            full_output,
            full_conv_state,
            full_recurrent_state,
        ) = layer(hidden_states)

        split_position = 3

        (
            first_output,
            first_conv_state,
            first_recurrent_state,
        ) = layer(
            hidden_states[
                :,
                :split_position,
            ]
        )

        (
            second_output,
            second_conv_state,
            second_recurrent_state,
        ) = layer(
            hidden_states[
                :,
                split_position:,
            ],
            conv_state=first_conv_state,
            recurrent_state=first_recurrent_state,
        )

        chunked_output = torch.cat(
            [
                first_output,
                second_output,
            ],
            dim=1,
        )

        self.assertEqual(
            full_output.shape,
            (2, 7, 8),
        )

        self.assertEqual(
            full_conv_state.shape,
            (2, 18, 3),
        )

        self.assertEqual(
            full_recurrent_state.shape,
            (2, 2, 3, 3),
        )

        torch.testing.assert_close(
            chunked_output,
            full_output,
            rtol=1e-5,
            atol=1e-6,
        )

        torch.testing.assert_close(
            second_conv_state,
            full_conv_state,
            rtol=1e-5,
            atol=1e-6,
        )

        torch.testing.assert_close(
            second_recurrent_state,
            full_recurrent_state,
            rtol=1e-5,
            atol=1e-6,
        )