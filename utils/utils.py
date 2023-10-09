from typing import Dict
from torch import nn
from utils.constant import MODEL_CONFIG, MODEL_MAP
from torchvision import transforms, datasets

import argparse
import torch

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

def create_model_from_config(
    args: argparse.Namespace
    ):
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
    
    return MODEL_MAP[args.model](**config)

def load_dataset(
    dataset: str = "CIFAR10"
    ):
    if dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="cifar_train", 
                                         train=True, 
                                        transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()                                        ]),
                                        download=True)
        val_dataset = datasets.CIFAR10(root="cifar_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "MNIST":
        train_dataset = datasets.MNIST(root="mnist_train", train=True, 
                                        transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        val_dataset = datasets.MNIST(root="mnist_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "SVHN":
        train_dataset = datasets.SVHN(root="svhn_train", split = 'train', 
                                        transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        val_dataset = datasets.SVHN(root="svhn_val",
                                    split = "test",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "Fashion_MNIST":
        train_dataset = datasets.FashionMNIST(root="fashion_mnist_train", train=True, 
                                        transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        val_dataset = datasets.FashionMNIST(root="fashion_mnist_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    return train_dataset, val_dataset

def loss_l1(y_pred, y_true):
    label_pred = torch.argmax(y_pred, dim=1)
    loss = (label_pred == y_true).int()
    return loss
