# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

#from .model_lighting import Transformer, TransformerModelArgs
from .parallelize_llama import parallelize_llama
from .pipeline_llama import pipeline_llama
from .model import Transformer
from .model import TransformerModelArgs
#from .model_mamba2 import Transformer as Mamba2Transformer
#from .model_mamba2 import TransformerModelArgs as Mamba2TransformerModelArgs
from .model_moba import TransformerMoBA
from .model_moba import MoBATransformerModelArgs

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "Transformer",
    "TransformerModelArgs",
    "llama3_configs",
    "Mamba2Transformer",
    "Mamba2TransformerModelArgs",
    "TransformerMoBA",
    "MoBATransformerModelArgs",
]


llama3_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, rope_theta=500000
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "1.5B": TransformerModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        # n_kv_heads=8,
        n_kv_heads=1,
        ffn_dim_multiplier=1.3,
        multiple_of=512,
        rope_theta=500000,
        use_flex_attn=False,
    ),
    "3B": TransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}

# Create Moba configs by converting base configs to MoBATransformerModelArgs
llama3_moba_configs = {
    name: MoBATransformerModelArgs(**config.__dict__)
    for name, config in llama3_configs.items()
}

register_train_spec(
    TrainSpec(
        name="llama3",
        cls=Transformer,
        config=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)

# Register Moba model
register_train_spec(
    TrainSpec(
        name="llama3_moba",
        cls=TransformerMoBA,
        config=llama3_moba_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)

from ..moba.init import register_moba
from ..moba.config import MoBAConfig
from .model_moba import MoBATransformerModelArgs

# 从 MoBATransformerModelArgs 中拿到默认的 chunk_size 和 topk
_default_moba_cfg = MoBAConfig(
    chunk_size=MoBATransformerModelArgs().moba_chunk_size,
    topk=      MoBATransformerModelArgs().moba_topk,
)

# 只要 import 了这个包，就会把 "moba" 和 "moba_naive" 注册到 ALL_ATTENTION_FUNCTIONS
register_moba(_default_moba_cfg)