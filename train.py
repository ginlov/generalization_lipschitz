from utils.train_base import train
from utils.utils import default_config, add_dict_to_argparser, create_model_from_config

import argparse


def start_train(
        args: argparse.Namespace,
    ):
    # create model
    model = create_model_from_config(args)
    dataset = args.dataset

    # training
    if args.model_type == 0:
        norm_type = "wo_norm"
    else:
        norm_type = args.norm_type
    log_file_name = "_".join([args.model, norm_type]) + ".txt"
    log_folder = "_".join([args.model, norm_type])
    training_config = {
        "model": model,
        "dataset": dataset,
        "log_file_name": log_file_name, 
        "clamp_value": args.clamp_value,
        "from_checkpoint": args.from_checkpoint,
        "log_folder": log_folder
    }

    train(**training_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())
    
    args = parser.parse_args()

    start_train(args)
