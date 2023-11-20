from os.path import isfile
from typing import Dict
from torch import nn
from utils.constant import MODEL_CONFIG, MODEL_MAP
from torchvision import transforms, datasets
from PIL import Image

import argparse
import torch
import numpy as np
import os

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
                                        transforms.ToTensor()                                        ]),
                                        download=True)
        val_dataset = datasets.CIFAR10(root="cifar_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "CIFAR10_AUG10":
        if os.path.isfile("cifar10_aug10.pth"):
            train_dataset = torch.load('cifar10_aug10.pth')
        else:
            train_dataset = create_augmented_dataset("CIFAR10", 10)
            torch.save(train_dataset, "cifar10_aug10.pth")
        val_dataset = datasets.CIFAR10(root="cifar_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "CIFAR10_AUG50":
        if os.path.isfile("cifar10_aug50.pth"):
            train_dataset = torch.load('cifar10_aug50.pth')
        else:
            train_dataset = create_augmented_dataset("CIFAR10", 50)
            torch.save(train_dataset, "cifar10_aug50.pth")
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
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        val_dataset = datasets.SVHN(root="svhn_val",
                                    split = "test",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "SVHN_AUG10":
        if os.path.isfile("svhn_aug10.pth"):
            train_dataset = torch.load("SVHN_AUG10.pth")
        else:
            train_dataset = create_augmented_dataset("SVHN", 10)
            torch.save(train_dataset, "svhn_aug10.pth")
        val_dataset = datasets.SVHN(root="svhn_val",
                                    split = "test",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "SVHN_AUG50":
        if os.path.isfile("svhn_aug50.pth"):
            train_dataset = torch.load("SVHN_AUG50.pth")
        else:
            train_dataset = create_augmented_dataset("SVHN", 50)
            torch.save(train_dataset, "svhn_aug50.pth")
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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x,
        y,
        transform,
        target_transform=None,
        **kwargs
    ):
        self.x = x
        self.y = y
        self.transform = transform
        self.target_transform = target_transform
        super().__init__(**kwargs)

    def __getitem__(self, index):
        img, label = self.x[index], self.y[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.x)

def create_augmented_dataset(
    original_dataset: str = "CIFAR10",
    num_augmented: int = 10
    ):
    if original_dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="cifar_train", 
                                         train=True, 
                                        transform=transforms.Compose([
                                        transforms.ToTensor()                                        ]),
                                        download=True)
    elif original_dataset == "SVHN":
        train_dataset = datasets.SVHN(root="svhn_train", split = 'train', 
                                        transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
    augmented_data = [train_dataset.data.reshape(-1, 3, 32, 32)]
    label = [train_dataset.targets]
    for _ in range(num_augmented):
        augmented_data.append(transforms.RandAugment()(torch.tensor(augmented_data[0], dtype=torch.uint8)).numpy())
        label.append(label[0])
    label = np.concatenate(np.array(label)).tolist()
    augmented_data = np.concatenate(augmented_data).reshape(-1, 32, 32, 3)
    return CustomDataset(
        x = augmented_data,
        y = label,
        transform = transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform = None
    )


def loss_l1(y_pred, y_true):
    label_pred = torch.argmax(y_pred, dim=1)
    loss = (label_pred != y_true).int().float() * 2
    return loss
