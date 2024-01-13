import wandb
import argparse

def wandb_init(
    config: argparse.Namespace
    ):
    wandb.init(
        project="robgen-2024-icml",
        config = vars(config)
    )

def wandb_end():
    wandb.finish()
