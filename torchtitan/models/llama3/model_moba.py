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
        
        # 🚀 优化：预计算QKV分割点，避免运行时计算
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

    def forward(
        self,
        x: torch.Tensor,  # [bs, seqlen, dim]
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        
        # 🚀 优化方案1：如果可能，使用融合QKV投影
        if hasattr(self, '_use_fused_qkv') and self._use_fused_qkv:
            # 融合版本：一次矩阵乘法代替三次
            qkv = self._fused_qkv_proj(x)
            xq, xk, xv = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            # 标准版本：保持兼容性
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        output, _ = moba_layer(
            moba_impl=moba_attn_varlen,
            moba_config=self.moba_config,
            module=self,
            query=xq.transpose(1, 2),  # [bs, n_heads, seqlen, head_dim]
            key=xk.transpose(1, 2),    # [bs, n_kv_heads, seqlen, head_dim]
            value=xv.transpose(1, 2),  # [bs, n_kv_heads, seqlen, head_dim]
        )

        return self.wo(output.view(bs, seqlen, -1))
    
    def _setup_fused_qkv(self):
        """🚀 设置融合QKV投影（可选优化）"""
        try:
            # 创建融合权重矩阵
            fused_weight = torch.cat([
                self.wq.weight,  # [q_dim, model_dim]
                self.wk.weight,  # [kv_dim, model_dim] 
                self.wv.weight   # [kv_dim, model_dim]
            ], dim=0)  # [q_dim + 2*kv_dim, model_dim]
            
            self.register_buffer('_fused_qkv_weight', fused_weight)
            self._use_fused_qkv = True
            
            def _fused_qkv_proj(x):
                return torch.nn.functional.linear(x, self._fused_qkv_weight)
            self._fused_qkv_proj = _fused_qkv_proj
            
        except Exception:
            # 如果融合失败，回退到标准版本
            self._use_fused_qkv = False

class TransformerMoBA(Transformer):
    def __init__(self, model_args: MoBATransformerModelArgs):
        super().__init__(model_args)
        # Override the attention module with MoBA attention
        for layer in self.layers.values():
            layer.attention = MoBAAttention(model_args)
            
        # 🚀 可选：启用融合QKV优化（实验性）
        self._enable_fused_qkv_optimization()

    def _enable_fused_qkv_optimization(self):
        """启用融合QKV矩阵乘法优化"""
        for layer in self.layers.values():
            if hasattr(layer.attention, '_setup_fused_qkv'):
                layer.attention._setup_fused_qkv()

    @classmethod
    def from_model_args(cls, model_args: MoBATransformerModelArgs) -> "TransformerMoBA":
        return cls(model_args) 