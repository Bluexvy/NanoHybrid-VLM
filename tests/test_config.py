import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config


class TextOnlyConfig:
    """模拟纯文本模型的 Hugging Face 根配置。"""

    model_type = "qwen3"
    max_position_embeddings = 32768

    def get_text_config(self, decoder=False):
        assert decoder is True
        return self


class MultimodalConfig:
    """模拟具有文本和视觉子配置的根配置。"""

    model_type = "qwen3_5"

    def __init__(self):
        self.text_config = SimpleNamespace(
            model_type="qwen3_5_text",
            max_position_embeddings=262144,
        )

        self.vision_config = SimpleNamespace(
            model_type="qwen3_5_vision",
        )

    def get_text_config(self, decoder=False):
        assert decoder is True
        return self.text_config


class TestConfigViews(unittest.TestCase):

    def test_text_only_model_uses_root_as_text_config(self):
        root_config = TextOnlyConfig()

        with tempfile.TemporaryDirectory() as model_dir:
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=root_config,
            ) as from_pretrained:
                config = Config(
                    model=model_dir,
                    max_model_len=4096,
                )

        self.assertIs(config.hf_config, root_config)
        self.assertIs(config.text_config, root_config)
        self.assertIsNone(config.vision_config)
        self.assertEqual(config.max_model_len, 4096)

        from_pretrained.assert_called_once_with(model_dir)

    def test_multimodal_model_exposes_text_and_vision_configs(self):
        root_config = MultimodalConfig()

        with tempfile.TemporaryDirectory() as model_dir:
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=root_config,
            ):
                config = Config(
                    model=model_dir,
                    max_model_len=300000,
                )

        self.assertIs(config.hf_config, root_config)
        self.assertIs(
            config.text_config,
            root_config.text_config,
        )
        self.assertIs(
            config.vision_config,
            root_config.vision_config,
        )

        # 引擎请求 300000，但模型最多支持 262144。
        self.assertEqual(config.max_model_len, 262144)


if __name__ == "__main__":
    unittest.main()