# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#使用前需要安装flash-linear-attention 
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from .model import (
    TransformerModelArgs,
    Transformer,
    TransformerBlock,
    FeedForward,
    RMSNorm,
)

from fla.layers.gated_deltanet import GatedDeltaNet

@dataclass
class GatedDeltaTransformerModelArgs(TransformerModelArgs):
    """Extended args for GatedDelta attention mechanism"""
    # GatedDelta specific parameters
    expand_v: float = 2.0
    head_dim: int = 256
    mode: str = 'chunk'
    use_gate: bool = True
    use_short_conv: bool = True
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1e-5

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        super().update_from_config(job_config, tokenizer)
        
        # Update GatedDelta-specific parameters from config if they exist
        if hasattr(job_config.model, 'expand_v'):
            self.expand_v = job_config.model.expand_v
        if hasattr(job_config.model, 'mode'):
            self.mode = job_config.model.mode
        if hasattr(job_config.model, 'use_gate'):
            self.use_gate = job_config.model.use_gate
        if hasattr(job_config.model, 'use_short_conv'):
            self.use_short_conv = job_config.model.use_short_conv
        if hasattr(job_config.model, 'conv_size'):
            self.conv_size = job_config.model.conv_size

class GatedDeltaAttention(nn.Module):
    
    def __init__(self, model_args: GatedDeltaTransformerModelArgs):
        super().__init__()
        
        # Calculate num_heads based on head_dim and existing n_heads
        # Keep the same total parameter allocation
        total_qk_dim = model_args.n_heads * model_args.head_dim
        self.head_dim = model_args.head_dim
        self.num_heads = total_qk_dim // self.head_dim
        
        self.gated_delta = GatedDeltaNet(
            hidden_size=model_args.dim,
            expand_v=model_args.expand_v,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            mode=model_args.mode,
            use_gate=model_args.use_gate,
            use_short_conv=model_args.use_short_conv,
            conv_size=model_args.conv_size,
            conv_bias=model_args.conv_bias,
            norm_eps=model_args.norm_eps,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass compatible with Llama3 interface"""
        
        # Convert mask format if needed
        attention_mask = None
        if mask is not None:
            # Convert causal mask to padding mask format expected by GatedDelta
            # GatedDelta expects [batch_size, seq_len] with 0s for padding
            batch_size, seq_len = x.shape[:2]
            attention_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)
        
        # Call GatedDeltaNet
        output, _, _ = self.gated_delta(
            hidden_states=x,
            attention_mask=attention_mask,
        )
        
        return output

class TransformerGatedDelta(Transformer):
    """Transformer with GatedDelta attention layers"""
    
    def __init__(self, model_args: GatedDeltaTransformerModelArgs):
        # Initialize parent with the base args
        super().__init__(model_args)
        
        # Replace all attention layers with GatedDelta
        for layer in self.layers.values():
            layer.attention = GatedDeltaAttention(model_args)
    
    @classmethod
    def from_model_args(cls, model_args: GatedDeltaTransformerModelArgs) -> "TransformerGatedDelta":
        """Factory method to create TransformerGatedDelta from args"""
        return cls(model_args)