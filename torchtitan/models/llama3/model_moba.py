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
from . import model
from .moba.config import MoBAConfig
from .moba.wrapper import moba_layer
from .moba.moba_efficient import moba_attn_varlen

@dataclass
class MoBATransformerModelArgs(model.TransformerModelArgs):
    # Moba specific parameters
    attention_type = "moba"
    moba_chunk_size: int = 64  # Size of each chunk for Moba attention
    moba_topk: int = 8  # Number of top chunks to attend to
    moba_alpha_init: float = 1.0  # Initial value for alpha parameter
    moba_beta_init: float = 1.0  # Initial value for beta parameter
    moba_gamma_init: float = 1.0  # Initial value for gamma parameter
    
    # 🚀 性能优化：添加编译相关参数
    use_compile: bool = False  # 是否使用torch.compile优化

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
        if hasattr(job_config.model, "use_compile"):
            self.use_compile = job_config.model.use_compile

class MoBAAttention(model.Attention):
    def __init__(self, model_args: MoBATransformerModelArgs, shared_moba_config: MoBAConfig = None):
        super().__init__(model_args)
        # 🚀 性能优化：共享MoBA配置，避免重复创建
        if shared_moba_config is not None:
            self.moba_config = shared_moba_config
        else:
            self.moba_config = MoBAConfig(
                moba_chunk_size=model_args.moba_chunk_size,
                moba_topk=model_args.moba_topk
            )
        self.is_causal = True
        
        # 🚀 性能优化：缓存常用值，避免重复计算
        self._chunk_size = model_args.moba_chunk_size
        self._topk = model_args.moba_topk

    def forward(
        self,
        x: torch.Tensor,  # [bs, seqlen, dim]
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        
        # 🚀 性能优化：合并线性变换，确保内存连续性
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 🚀 性能优化：批量reshape + contiguous，减少函数调用开销
        xq = xq.view(bs, seqlen, -1, self.head_dim).contiguous()
        xk = xk.view(bs, seqlen, -1, self.head_dim).contiguous()
        xv = xv.view(bs, seqlen, -1, self.head_dim).contiguous()
        
        # 🚀 性能优化：RoPE应用在连续内存上更高效
        xq, xk = model.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 🚀 性能优化：直接传递给moba_layer，一次性transpose + contiguous
        output, _ = moba_layer(
            moba_impl=moba_attn_varlen,
            moba_config=self.moba_config,
            module=self,
            query=xq.transpose(1, 2).contiguous(),  # 内存连续性对Flash Attention很重要
            key=xk.transpose(1, 2).contiguous(),
            value=xv.transpose(1, 2).contiguous(),
        )

        # 🚀 性能优化：直接reshape输出，避免中间变量
        return self.wo(output.view(bs, seqlen, -1))

class TransformerMoBA(model.Transformer):
    def __init__(self, model_args: MoBATransformerModelArgs):
        super().__init__(model_args)
        
        # 🚀 性能优化：创建共享的MoBA配置，所有layer复用同一个对象
        shared_moba_config = MoBAConfig(
            moba_chunk_size=model_args.moba_chunk_size,
            moba_topk=model_args.moba_topk
        )
        
        # 🚀 性能优化：批量替换attention模块，使用共享配置
        for layer in self.layers.values():
            layer.attention = MoBAAttention(model_args, shared_moba_config)
            
        # 🚀 性能优化：如果启用编译，对attention模块进行torch.compile优化
        if model_args.use_compile:
            for layer in self.layers.values():
                if hasattr(torch, 'compile'):  # 确保torch.compile可用
                    layer.attention.forward = torch.compile(layer.attention.forward)

    @classmethod
    def from_model_args(cls, model_args: MoBATransformerModelArgs) -> "TransformerMoBA":
        return cls(model_args) 