from os.path import isfile
from typing import Dict
from torch import nn
from torchvision.transforms.transforms import ToTensor
from utils.constant import MODEL_CONFIG, MODEL_MAP
from torchvision import transforms, datasets
from PIL import Image

import argparse
import torch
import numpy as np
import os
import pickle

def default_config():
    return {
        "model": "mlp",
        "dataset": "CIFAR10",
        "model_type": 0,
        "clamp_value": -1.0,
        "norm_type": "batch",
        "from_checkpoint": False,
        "num_epoch": 20
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
            # train_data = torch.load('cifar10_aug10.pth')
            with open("cifar10_aug10.pth", "rb") as f:
                train_data = pickle.load(f)
            train_dataset = CustomDataset(
                x=train_data["x"],
                y=train_data["y"],
                transform=transforms.ToTensor()
            )
        else:
            train_dataset = create_augmented_dataset("CIFAR10", 10)
            # torch.save({
            #     "x": train_dataset.x.clone(),
            #     "y": train_dataset.y.clone(),
            # }, "cifar10_aug10.pth")
            with open("cifar10_aug10.pth", "wb+") as f:
                pickle.dump({
                    "x": train_dataset.x,
                    "y": train_dataset.y
                }, f)
        val_dataset = datasets.CIFAR10(root="cifar_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "CIFAR10_AUG50":
        if os.path.isfile("cifar10_aug50.pth"):
            # train_data = torch.load('cifar10_aug50.pth')
            with open("cifar10_aug50.pth", "rb") as f:
                train_data = pickle.load(f)
            train_dataset = CustomDataset(
                x=train_data["x"],
                y=train_data["y"],
                transform=transforms.ToTensor()
            )
        else:
            train_dataset = create_augmented_dataset("CIFAR10", 50)
            print("train_dataset ok, start saving to file")
            # torch.save({
            #     "x": train_dataset.x,
            #     "y": train_dataset.y
            # }, "cifar10_aug50.pth")
            with open("cifar10_aug50.pth", "wb+") as f:
                pickle.dump({
                    "x": train_dataset.x,
                    "y": train_dataset.y
                }, f)
            print("end saving to file")
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
            # train_data = torch.load("SVHN_AUG10.pth")
            with open("svhn_aug10.pth", "rb") as f:
                train_data = pickle.load(f)
            train_dataset = CustomDataset(
                x=train_data["x"],
                y=train_data["y"],
                transform=transforms.ToTensor()
            )
        else:
            train_dataset = create_augmented_dataset("SVHN", 10)
            # torch.save({
            #     "x": train_dataset.x,
            #     "y": train_dataset.y
            # }, "svhn_aug10.pth")
            with open("svhn_aug10.pth", "wb+") as f:
                pickle.dump({
                    "x": train_dataset.x,
                    "y": train_dataset.y
                }, f)
        val_dataset = datasets.SVHN(root="svhn_val",
                                    split = "test",
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif dataset == "SVHN_AUG50":
        if os.path.isfile("svhn_aug50.pth"):
            # train_data = torch.load("SVHN_AUG50.pth")
            with open("svhn_aug50.pth", "rb") as f:
                train_data = pickle.load(f)
            train_dataset =  CustomDataset(
                x=train_data["x"],
                y=train_data["y"],
                transform=transforms.ToTensor()
            )
        else:
            train_dataset = create_augmented_dataset("SVHN", 50)
            # torch.save({
            #     "x": train_dataset.x,
            #     "y": train_dataset.y
            # }, "svhn_aug50.pth")
            with open("svhn_aug50.pth", "wb+") as f:
                train_data = pickle.dump({
                    "x": train_dataset.x,
                    "y": train_dataset.y
                }, f)
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
        label = [train_dataset.targets]
    elif original_dataset == "SVHN":
        train_dataset = datasets.SVHN(root="svhn_train", split = 'train', 
                                        transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        label = [train_dataset.labels]
    augmented_data = [train_dataset.data.reshape(-1, 3, 32, 32)]
    print("Start augmenting data")
    for _ in range(num_augmented):
        augmented_data.append(transforms.RandAugment()(torch.tensor(augmented_data[0], dtype=torch.uint8)).numpy())
        label.append(label[0])
    print("Ending augmenting data")
    label = np.concatenate(np.array(label)).tolist()
    print("Ending concatenating label")
    augmented_data = np.concatenate(augmented_data).reshape(-1, 32, 32, 3)
    print("Ending concatenating data")
    return CustomDataset(
        x = augmented_data,
        y = label,
        transform = transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform = None
    )

def loss_l1(y_pred, y_true):
    if isinstance(y_pred, np.ndarray):
        label_pred = torch.tensor(y_pred)
    else:
        label_pred = torch.argmax(y_pred, dim=1)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    loss = (label_pred != y_true).int().float() * 2
    return loss
