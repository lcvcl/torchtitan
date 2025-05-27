import argparse
import json
import os
import sys
import time
from pathlib import Path

from typing import Optional

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.config_manager import ConfigManager
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools.utils import device_module, device_type
from safetensors.torch import save_file

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def save_to_safetensor(
    config_path: str,
    checkpoint_path: str,
    save_path: str,
    seed: Optional[int] = None,
):
    # Load configuration from toml file
    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={config_path}"])

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)

    train_spec = get_train_spec(config.model.name)

    print(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    # Tokenizer setup
    tokenizer = train_spec.build_tokenizer_fn(config)

    model_cls = train_spec.cls
    model_args = train_spec.config[config.model.flavor]
    model_args.update_from_config(config, tokenizer)

    init_device = "meta" if world_size > 1 else device
    with torch.device(init_device):
        model = model_cls.from_model_args(model_args)

    world_mesh = None

    dist_utils.set_determinism(world_mesh, device, seed, False)

    # materalize model
    model.to_empty(device=device_type)
    model.eval()

    state_dict = {"model": model.state_dict()}


    begin = time.monotonic()
    print(f"Loading chkpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)

    print(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")

    model_state_dict = model.state_dict()
    model_state_dict.pop("freqs_cis")
    save_file(model_state_dict, save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to load (required)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Checkpoint path to load (required)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    save_to_safetensor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        save_path=args.save_path,
        seed=args.seed
    )