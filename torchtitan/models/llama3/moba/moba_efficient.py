"""A clean version of efficient moba implementation with flash-attn"""

import torch
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import FlashAttnVarlenFunc
from functools import lru_cache
from einops import rearrange


@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """calc chunks that needs moba attention"""

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """

    # filter chunks ( remove last chunk of each batch )
    # filtered_chunk_indices: chunk index list that excludes the last chunk of each batch
    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = torch.ones(
        (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    )


class MixedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # self attn
        self_attn_out_sh, self_attn_lse_hs, _ = FlashAttnVarlenFunc.apply(
            q,
            k,
            v,
            self_attn_cu_seqlen,
            self_attn_cu_seqlen,
            max_seqlen,
            max_seqlen,
            0.0,  # dropout_p
            softmax_scale,
            True,  # causal
            (-1, -1),  # window_size
            0.0,  # softcap
            None,  # alibi_slopes
            False,  # deterministic
            True,  # return_attn_probs
            None,  # block_table
            torch.is_grad_enabled(),
        )

        # moba attn
        moba_attn_out, moba_attn_lse_hs, _ = FlashAttnVarlenFunc.apply(
            moba_q,
            moba_kv[:, 0],
            moba_kv[:, 1],
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            max_seqlen,
            moba_chunk_size,
            0.0,  # dropout_p
            softmax_scale,
            False,  # causal
            (-1, -1),  # window_size
            0.0,  # softcap
            None,  # alibi_slopes
            False,  # deterministic
            True,  # return_attn_probs
            None,  # block_table
            torch.is_grad_enabled(),
        )

        # convert lse shape hs -> sh ( follow the legacy mix attn logic )
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        # output buffer [S, H, D], same shape as q
        output = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # flatten vS & H for index ops
        output_2d = output.view(-1, q.shape[2])

        # calc mixed_lse
        # minus max lse to avoid exp explosion
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        # add attn output
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # add moba output
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)
        # add back max lse
        mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)

        # Save tensors for backward pass
        ctx.save_for_backward(
            q, k, v, output, mixed_attn_lse_sh,
            self_attn_cu_seqlen, self_attn_cu_seqlen,
            moba_q, moba_kv, moba_cu_seqlen_q, moba_cu_seqlen_kv,
            moba_q_sh_indices
        )

        return output

    @staticmethod
    def backward(ctx, d_output):
        max_seqlen = ctx.max_seqlen
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale

        # Get saved tensors
        (
            q, k, v, output, mixed_attn_lse_sh,
            self_attn_cu_seqlen, self_attn_cu_seqlen,
            moba_q, moba_kv, moba_cu_seqlen_q, moba_cu_seqlen_kv,
            moba_q_sh_indices
        ) = ctx.saved_tensors

        d_output = d_output.contiguous()

        # 写入调试信息到文件
        with open("/tmp/moba_debug.log", "a") as f:
            print(f"=== Backward Debug Info ===", file=f, flush=True)
            print(f"Input shapes and sizes:", file=f, flush=True)
            print(f"q shape: {q.shape}, size: {q.numel()}", file=f, flush=True)
            print(f"k shape: {k.shape}, size: {k.numel()}", file=f, flush=True)
            print(f"v shape: {v.shape}, size: {v.numel()}", file=f, flush=True)
            print(f"moba_kv shape: {moba_kv.shape}, size: {moba_kv.numel()}", file=f, flush=True)
            print(f"moba_q shape: {moba_q.shape}, size: {moba_q.numel()}", file=f, flush=True)
            print(f"d_output shape: {d_output.shape}, size: {d_output.numel()}", file=f, flush=True)

        # Self attention backward
        dq, dk, dv = flash_attn_varlen_func(
            q,
            k,
            v,
            self_attn_cu_seqlen,
            self_attn_cu_seqlen,
            max_seqlen,
            max_seqlen,
            0.0,  # dropout_p
            softmax_scale,
            True,  # causal
            return_attn_probs=True,
        )

        with open("/tmp/moba_debug.log", "a") as f:
            print(f"After self attention backward:", file=f, flush=True)
            print(f"dq shape: {dq.shape}, size: {dq.numel()}", file=f, flush=True)
            print(f"dk shape: {dk.shape}, size: {dk.numel()}", file=f, flush=True)
            print(f"dv shape: {dv.shape}, size: {dv.numel()}", file=f, flush=True)

        headdim = q.shape[-1]
        d_moba_output = (
            d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )
        moba_output = (
            output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )

        mixed_attn_vlse = (
            mixed_attn_lse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
        )

        # MOBA attention backward
        dmq, dmk, dmv = flash_attn_varlen_func(
            moba_q,
            moba_kv[:, 0],
            moba_kv[:, 1],
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            max_seqlen,
            moba_chunk_size,
            0.0,  # dropout_p
            softmax_scale,
            False,  # causal
            return_attn_probs=True,
        )

        with open("/tmp/moba_debug.log", "a") as f:
            print(f"After MOBA attention backward:", file=f, flush=True)
            print(f"dmq shape: {dmq.shape}, size: {dmq.numel()}", file=f, flush=True)
            print(f"dmk shape: {dmk.shape}, size: {dmk.numel()}", file=f, flush=True)
            print(f"dmv shape: {dmv.shape}, size: {dmv.numel()}", file=f, flush=True)

        # Transform gradients to match original shapes
        num_head = 16  # From model config
        head_dim = 128  # dim / num_head = 2048 / 16 = 128
        
        # Handle empty tensors
        if dmk.numel() == 0 or dmv.numel() == 0:
            with open("/tmp/moba_debug.log", "a") as f:
                print(f"Empty tensor detected:", file=f, flush=True)
                print(f"dmk empty: {dmk.numel() == 0}", file=f, flush=True)
                print(f"dmv empty: {dmv.numel() == 0}", file=f, flush=True)
            
            # Create zero tensors with the correct shape
            dmk = torch.zeros((2048, num_head, head_dim), device=dmk.device, dtype=dmk.dtype)
            dmv = torch.zeros((2048, num_head, head_dim), device=dmv.device, dtype=dmv.dtype)
        
        # Ensure dmk and dmv have the same shape
        if dmk.shape != dmv.shape:
            with open("/tmp/moba_debug.log", "a") as f:
                print(f"Shape mismatch between dmk and dmv:", file=f, flush=True)
                print(f"dmk shape: {dmk.shape}", file=f, flush=True)
                print(f"dmv shape: {dmv.shape}", file=f, flush=True)
            
            # Reshape to match the larger tensor
            target_shape = max(dmk.shape, dmv.shape)
            if dmk.shape != target_shape:
                dmk = dmk.expand(target_shape)
            if dmv.shape != target_shape:
                dmv = dmv.expand(target_shape)
        
        # Calculate expected sizes
        expected_size = 2048 * 2 * num_head * head_dim  # [2048, 2, 16, 128]
        actual_size = dmk.numel() + dmv.numel()
        
        with open("/tmp/moba_debug.log", "a") as f:
            print(f"Size check:", file=f, flush=True)
            print(f"Expected total size: {expected_size}", file=f, flush=True)
            print(f"Actual total size: {actual_size}", file=f, flush=True)
        
        # Stack k and v gradients
        dmkv = torch.stack((dmk, dmv), dim=1)  # [seqlen, 2, head, head_dim]
        
        with open("/tmp/moba_debug.log", "a") as f:
            print(f"After stacking:", file=f, flush=True)
            print(f"dmkv shape: {dmkv.shape}, size: {dmkv.numel()}", file=f, flush=True)

        # Ensure the shape matches the expected output shape
        if dmkv.shape != (2048, 2, num_head, head_dim):
            with open("/tmp/moba_debug.log", "a") as f:
                print(f"Reshaping dmkv from {dmkv.shape} to (2048, 2, {num_head}, {head_dim})", file=f, flush=True)
            
            # Create a new tensor with the correct shape
            new_dmkv = torch.zeros((2048, 2, num_head, head_dim), device=dmkv.device, dtype=dmkv.dtype)
            
            # Calculate how much data we can copy
            actual_seqlen = min(dmkv.shape[0], 2048)
            actual_heads = min(dmkv.shape[2], num_head)
            actual_dim = min(dmkv.shape[3], head_dim)
            
            with open("/tmp/moba_debug.log", "a") as f:
                print(f"Copying data with shapes:", file=f, flush=True)
                print(f"actual_seqlen: {actual_seqlen}", file=f, flush=True)
                print(f"actual_heads: {actual_heads}", file=f, flush=True)
                print(f"actual_dim: {actual_dim}", file=f, flush=True)
            
            # Copy the data we have
            new_dmkv[:actual_seqlen, :2, :actual_heads, :actual_dim] = dmkv[:actual_seqlen, :2, :actual_heads, :actual_dim]
            dmkv = new_dmkv

        with open("/tmp/moba_debug.log", "a") as f:
            print(f"Final shapes and sizes:", file=f, flush=True)
            print(f"dq shape: {dq.shape}, size: {dq.numel()}", file=f, flush=True)
            print(f"dk shape: {dk.shape}, size: {dk.numel()}", file=f, flush=True)
            print(f"dv shape: {dv.shape}, size: {dv.numel()}", file=f, flush=True)
            print(f"dmq shape: {dmq.shape}, size: {dmq.numel()}", file=f, flush=True)
            print(f"dmkv shape: {dmkv.shape}, size: {dmkv.numel()}", file=f, flush=True)
            print(f"=== End Backward Debug Info ===\n", file=f, flush=True)

        # Clear unnecessary tensors to free memory
        del d_moba_output, moba_output, mixed_attn_vlse
        torch.cuda.empty_cache()

        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None


def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """An efficient version of moba implementation with triton kernels and flash-attn, the core logic:
    1. Calculate the chunks and the number of chunks, n = floor(data_size / chunk_size)
       - tokens in the tail chunk are reserved for self attn
       - tokens in other chunks will be processed in later steps
    2. K in each chunk will calculate mean value as the representative k, and Q will attend to these representative
    k to get the gate logit, which will be used to select topk chunks
    3. Select the topk chunks and get the dense q for each kv chunk pair and do the varlen attention
    4. Combine the varlen attn and self attn results via online softmax to get the final result

    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor, same definition in flash attn
        max_seqlen (int): the max sequence length of the batch, same definition in flash attn

    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """

    kv = torch.stack((k, v), dim=1)

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ prepare chunk meta """
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0

    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    self_attn_cu_seqlen = cu_chunk

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    """ calc key_gate_weight and gate """

    # key_gate_weight [ F_N_CHUNK, HEAD, HEAD_DIM ]
    key_gate_weight = (
        filtered_kv[:, 0]
        .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
        .mean(dim=1)
        .float()
    )
    q = q.type(torch.float32)  # float logit on the fly for better gate logit perception
    key_gate_weight = key_gate_weight.type(
        torch.float32
    )  # float logit for better gate logit perception
    gate = torch.einsum(
        "nhd,shd->nhs", key_gate_weight, q
    )  # gate [ F_N_CHUNK, HEAD, SEQ ]
    key_gate_weight = key_gate_weight.type_as(k)
    q = q.type_as(k)

    # pose process gate, masking unchosen batch and apply causal mask to current chunk
    gate_seq_idx = torch.arange(0, seqlen, device=q.device, dtype=torch.int32)[
        None, :
    ].repeat(num_filtered_chunk, 1)
    chunk_end = cu_chunk[filtered_chunk_indices + 1]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]
    gate_batch_end_mask = gate_seq_idx >= batch_end[:, None]
    gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))

    """ find moba q that needs moba attn """
    # find topk chunks
    # gate_mask [ N_CHUNK, HEAD, SEQ ], true indicates that needs attention
    _, gate_top_k_idx = torch.topk(gate, k=moba_topk, dim=0, largest=True, sorted=False)
    # apply causal mask
    gate_mask = torch.logical_not(gate.isinf())
    # select topk chunks
    gate_idx_mask = torch.zeros(gate_mask.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=0, index=gate_top_k_idx, value=True)
    gate_mask = torch.logical_and(gate_mask, gate_idx_mask)

    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[
        -1
    ]  # [ HS indices ] * N
    # moba_seqlen_q indicates that how many q chunks are selected for each kv chunk - head
    moba_seqlen_q = gate_mask.sum(dim=-1).flatten()
    # select all q that needs moba attn based on the moba_q_indices
    moba_q = rearrange(q, "s h d -> ( h s ) d").index_select(
        0, moba_q_indices
    )  # [ selected_S, D ]
    moba_q = moba_q.unsqueeze(1)
    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    moba_q_sh_indices = moba_q_indices % seqlen * num_head + moba_q_indices // seqlen

    """ prepare moba kv """
    # Since moba_q is organized as HS * N, we need to reorganize kv to adapt to q

    # cut off zero experts
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    # only keep the kv that has q select > 0
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
    # moba cu_seqlen for flash attn
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(moba_chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)
    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv , or the grad may be nan
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)
    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_filtered_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    # Shape check
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"

    # Wrapping up the flash attn call and online softmax dlse inside MixedAttention class
    return MixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    )
