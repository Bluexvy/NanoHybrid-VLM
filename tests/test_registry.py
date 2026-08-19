import unittest
from types import SimpleNamespace

from nanovllm.models.registry import (
    ModelRegistry,
    get_model_class,
)


class TestModelRegistry(unittest.TestCase):

    def test_resolve_qwen3(self):
        root_config = SimpleNamespace(
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
        )

        model_class = get_model_class(root_config)

        from nanovllm.models.qwen3 import Qwen3ForCausalLM

        self.assertIs(
            model_class,
            Qwen3ForCausalLM,
        )

    def test_rejects_unknown_model_type(self):
        root_config = SimpleNamespace(
            model_type="unknown_model",
            architectures=["UnknownModel"],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported model_type 'unknown_model'",
        ):
            get_model_class(root_config)

    def test_rejects_architecture_mismatch(self):
        root_config = SimpleNamespace(
            model_type="qwen3",
            architectures=[
                "Qwen3ForSequenceClassification"
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not support checkpoint architectures",
        ):
            get_model_class(root_config)

    def test_rejects_duplicate_registration(self):
        registry = ModelRegistry()

        registration = {
            "model_type": "dummy",
            "architectures": ("DummyModel",),
            "module": "builtins",
            "class_name": "object",
        }

        registry.register(**registration)

        with self.assertRaisesRegex(
            ValueError,
            "already registered",
        ):
            registry.register(**registration)


if __name__ == "__main__":
    unittest.main()