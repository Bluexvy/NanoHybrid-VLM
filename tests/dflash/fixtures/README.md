# Public configuration fixtures

Downloaded on 2026-10-15 from:

- https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
- https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/main/config.json
- https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/main/model.safetensors

`qwen38_27b_dflash2_shapes.json` contains only names, shapes and dtypes extracted
from the **8,928-byte safetensors JSON header** (81 tensors). No tensor payload
was downloaded or included. These offline fixtures let tests detect incorrect
layer counts, TP dimensions, feature indices and selector parameter names
without allocating the full model. Real runs validate local checkpoint headers
again and the downloader records pinned repository revisions in manifests.
