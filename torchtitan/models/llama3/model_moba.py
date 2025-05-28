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
    TransformerModelArgs as BaseTransformerModelArgs,
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
from .moba.moba_efficient import moba_efficient
from .moba.attention import MoBAAttention

@dataclass
class TransformerModelArgs(BaseTransformerModelArgs):
    # Override Moba specific parameters
    moba_chunk_size: int = 64
    moba_topk: int = 8

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        super().update_from_config(job_config, tokenizer)

class MoBAAttention(Attention):
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__(model_args)
        self.moba_config = MoBAConfig(
            moba_chunk_size=model_args.moba_chunk_size,
            moba_topk=model_args.moba_topk
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        # Use Moba attention
        output, _ = moba_layer(
            moba_impl=moba_efficient,
            moba_config=self.moba_config,
            module=self,
            query=xq,
            key=xk,
            value=xv,
        )

        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

class TransformerMoBA(Transformer):
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__(model_args)
        # Override the attention module with MoBA attention
        for layer in self.layers.values():
            layer.attention = MoBAAttention(model_args)

    @classmethod
    def from_model_args(cls, model_args: TransformerModelArgs) -> "TransformerMoBA":
        return cls(model_args) 