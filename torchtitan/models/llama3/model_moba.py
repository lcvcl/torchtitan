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

        # 首先将输入重塑为正确的形状
        xq = xq.view(bs, seqlen, self.n_heads, self.head_dim)  # [bs, seqlen, n_heads, head_dim]
        xk = xk.view(bs, seqlen, self.n_kv_heads, self.head_dim)  # [bs, seqlen, n_kv_heads, head_dim]
        xv = xv.view(bs, seqlen, self.n_kv_heads, self.head_dim)  # [bs, seqlen, n_kv_heads, head_dim]

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 扩展 key 和 value 的 head 数量
        keys = repeat_kv(xk, self.n_rep)  # [bs, seqlen, n_heads, head_dim]
        values = repeat_kv(xv, self.n_rep)  # [bs, seqlen, n_heads, head_dim]

        # 调整维度顺序
        xq = xq.transpose(1, 2)  # [bs, n_heads, seqlen, head_dim]
        keys = keys.transpose(1, 2)  # [bs, n_heads, seqlen, head_dim]
        values = values.transpose(1, 2)  # [bs, n_heads, seqlen, head_dim]

        # 处理每个batch
        outputs = []
        for i in range(bs):
            # 取当前batch的数据并调整形状
            curr_q = xq[i].permute(1, 0, 2).contiguous()  # [seqlen, n_heads, head_dim]
            curr_k = keys[i].permute(1, 0, 2).contiguous()  # [seqlen, n_heads, head_dim]
            curr_v = values[i].permute(1, 0, 2).contiguous()  # [seqlen, n_heads, head_dim]

            # 当前batch的序列长度
            curr_cu_seqlens = torch.tensor([0, seqlen], device=x.device, dtype=torch.int32)

            # 对当前batch使用Moba attention
            curr_output = moba_attn_varlen(
                q=curr_q,  # [seqlen, n_heads, head_dim]
                k=curr_k,  # [seqlen, n_heads, head_dim]
                v=curr_v,  # [seqlen, n_heads, head_dim]
                cu_seqlens=curr_cu_seqlens,  # [2]
                max_seqlen=seqlen,  # 2048
                moba_chunk_size=self.moba_config.moba_chunk_size,  # 64
                moba_topk=self.moba_config.moba_topk,  # 8
            )

            # 调整输出形状
            curr_output = curr_output.permute(1, 0, 2)  # [n_heads, seqlen, head_dim]
            outputs.append(curr_output)

        # 合并所有batch的输出
        output = torch.stack(outputs, dim=0)  # [bs, n_heads, seqlen, head_dim]
        output = output.transpose(1, 2)  # [bs, seqlen, n_heads, head_dim]
        output = output.reshape(bs, seqlen, -1)  # [bs, seqlen, dim]
        return self.wo(output)

class TransformerMoBA(Transformer):
    def __init__(self, model_args: MoBATransformerModelArgs):
        super().__init__(model_args)
        # Override the attention module with MoBA attention
        for layer in self.layers.values():
            layer.attention = MoBAAttention(model_args)

    @classmethod
    def from_model_args(cls, model_args: MoBATransformerModelArgs) -> "TransformerMoBA":
        return cls(model_args) 