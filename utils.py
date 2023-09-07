from typing import Dict
import argparse

def default_config():
    return {
        "model": "mlp",
        "dataset": "CIFAR10",
        "model_type": 0,
        "clamp_value": -1.0,
        "norm_type": "batch",
        "from_checkpoint": False
    }

def add_dict_to_argparser(
    parser: argparse.ArgumentParser,
    config_dict: Dict
    ):
    for k, v in config_dict.items():
        v_type = type(v)
        parser.add_argument(f"--{k}", type=v_type, default=v)
