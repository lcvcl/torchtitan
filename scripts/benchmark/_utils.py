import argparse
import json
import os
import sys
import time
from pathlib import Path

from typing import Optional

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torchtitan.components.metrics import build_device_memory_monitor

from torchtitan.config_manager import ConfigManager
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_module, device_type

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def apply_tp_minus_sp(model: nn.Module, tp_mesh: DeviceMesh):
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    for _, transformer_block in model.layers.items():
        layer_plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


@record
def load_checkpoints(
    config_path: str,
    checkpoint_path: str,
    seed: Optional[int] = None,
    deterministic: bool = False,
):
    init_logger()
    # Load configuration from toml file
    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={config_path}"])

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    device_memory_monitor = build_device_memory_monitor()

    train_spec = get_train_spec(config.model.name)

    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    # Tokenizer setup
    tokenizer = train_spec.build_tokenizer_fn(config)

    model_cls = train_spec.cls
    model_args = train_spec.config[config.model.flavor]
    model_args.update_from_config(config, tokenizer)

    init_device = "meta" if world_size > 1 else device
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_cls.from_model_args(model_args)

    world_mesh = None
    # Init distributed env
    if world_size > 1:
        dist_utils.init_distributed(config)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=world_size,
            pp=1,
            world_size=world_size,
            enable_loss_parallel=False,
        )
        # Build world mesh for parallelism
        world_mesh = parallel_dims.build_mesh(device_type=device_type)

        # apply_tp (with Sequence Parallel) on unevenly sharded
        # sequences would require https://github.com/pytorch/torchtitan/pull/686
        apply_tp_minus_sp(model, world_mesh["tp"])

    dist_utils.set_determinism(world_mesh, device, seed, deterministic)

    # materalize model
    model.to_empty(device=device_type)
    model.eval()

    state_dict = {"model": model.state_dict()}

    # Checkpoint Loading
    begin = time.monotonic()
    logger.info(f"Loading chkpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )
    return model, tokenizer