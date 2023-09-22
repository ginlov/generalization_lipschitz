from train_base import train
from torch import nn
from constant import MODEL_CONFIG, MODEL_MAP
from utils import default_config, add_dict_to_argparser

import argparse


def start_train(
        args: argparse.Namespace,
        debug: bool = False
    ):
    # create model
    config = MODEL_CONFIG[args.model]
    if args.model_type == 2:
        if args.norm_type == "BN":
            if args.model == "mlp":
                config["norm_layer"] = nn.BatchNorm1d
            if args.model == "mlp_1d":
                config["norm_layer"] = nn.BatchNorm1d
            else:
                config["norm_layer"] = nn.BatchNorm2d
        elif args.norm_type == "GN":
            config["norm_layer"] = nn.GroupNorm
        elif args.norm_type == "LN":
            config["norm_layer"] = nn.LayerNorm
        else:
            raise NotImplementedError("This norm type has not been implemented yet.")
    elif args.model_type == 1 and args.model in ["resnet", "resnet34", "resnet50"]:
        config["signal"] = 1
    elif args.model_type == 1:
        raise NotImplementedError("This setting is not support for vgg and mlp")
    
    dataset = args.dataset
    model = MODEL_MAP[args.model](**config)

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

    if debug:
        return config, training_config

    train(**training_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())
    
    args = parser.parse_args()

    start_train(args, False)
