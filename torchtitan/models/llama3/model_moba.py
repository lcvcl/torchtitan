# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.models.attention import build_attention, init_attention_mask
from .model import (
    TransformerModelArgs,
    Transformer,
    Attention,
    FeedForward,
    TransformerBlock,
    repeat_kv,
    apply_rotary_emb,
    precompute_freqs_cis,
    reshape_for_broadcast
)
from .moba.config import MoBAConfig
from .moba.wrapper import moba_layer
from .moba.moba_efficient import moba_attn_varlen

@dataclass
class MoBATransformerModelArgs(TransformerModelArgs):
    # Moba specific parameters
    attention_type = "moba"
    moba_chunk_size: int = 64  # Size of each chunk for Moba attention
    moba_topk: int = 8  # Number of top chunks to attend to
    moba_alpha_init: float = 1.0  # Initial value for alpha parameter
    moba_beta_init: float = 1.0  # Initial value for beta parameter
    moba_gamma_init: float = 1.0  # Initial value for gamma parameter

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        super().update_from_config(job_config, tokenizer)
        # Update Moba parameters from config if they exist
        if hasattr(job_config.model, "moba_chunk_size"):
            self.moba_chunk_size = job_config.model.moba_chunk_size
        if hasattr(job_config.model, "moba_topk"):
            self.moba_topk = job_config.model.moba_topk
        if hasattr(job_config.model, "moba_alpha_init"):
            self.moba_alpha_init = job_config.model.moba_alpha_init
        if hasattr(job_config.model, "moba_beta_init"):
            self.moba_beta_init = job_config.model.moba_beta_init
        if hasattr(job_config.model, "moba_gamma_init"):
            self.moba_gamma_init = job_config.model.moba_gamma_init

class MoBAAttention(Attention):
    def __init__(self, model_args: MoBATransformerModelArgs):
        super().__init__(model_args)
        self.moba_config = MoBAConfig(
            moba_chunk_size=model_args.moba_chunk_size,
            moba_topk=model_args.moba_topk
        )
        self.is_causal = True

    def forward(
        self,
        x: torch.Tensor,  # [bs, seqlen, dim]
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 使用-1支持Tensor Parallel，自动推断head数量
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 使用moba_layer处理，它会自动处理KV repeat和格式转换
        output, _ = moba_layer(
            moba_impl=moba_attn_varlen,
            moba_config=self.moba_config,
            module=self,
            query=xq.transpose(1, 2),  # [bs, n_heads, seqlen, head_dim]
            key=xk.transpose(1, 2),    # [bs, n_kv_heads, seqlen, head_dim]
            value=xv.transpose(1, 2),  # [bs, n_kv_heads, seqlen, head_dim]
        )

        return self.wo(output.view(bs, seqlen, -1))

class TransformerMoBA(Transformer):
    def __init__(self, model_args: MoBATransformerModelArgs):
        super().__init__(model_args)
        # Override the attention module with MoBA attention
        for layer in self.layers.values():
            layer.attention = MoBAAttention(model_args)

    @classmethod
    def from_model_args(cls, model_args: MoBATransformerModelArgs) -> "TransformerMoBA":
        return cls(model_args) 