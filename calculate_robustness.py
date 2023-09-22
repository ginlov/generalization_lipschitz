from utils import default_config, add_dict_to_argparser
from constant import MODEL_CONFIG, MODEL_MAP
from robustness.split_partitions import *

import torch
import torchvision
from torchvision import datasets, transforms
import argparse

num_clusters =  [100, 1000, 5000, 10000]

def cal_robustness(args):
    if args.model in ["resnet", "resnet34", "resnet50"]:
        config = MODEL_CONFIG[args.model]
        config["norm_layer"] = torch.nn.BatchNorm2d
        model = MODEL_MAP[args.model](**config)
    else:
        model = MODEL_MAP[args.model](**MODEL_CONFIG[args.model])
    model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    # Print model's state_dict
    print("Model's state_dict:")
    print(model_checkpoint["state_dict"])
    model.load_state_dict(model_checkpoint["state_dict"])

    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root="cifar_train", train=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ]),
                                                   download=True)
        val_dataset = torchvision.datasets.CIFAR10(root="cifar_val", train=False, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]),
                                                   download=True)
    elif args.dataset == "SVHN":
        train_dataset = torchvision.datasets.SVHN(root='svhn_train', split='train',
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.ToTensor(),
                                            ]),
                                            download=True)
        val_dataset = torchvision.datasets.SVHN(root="svhn_val", split='test', 
                                                transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor()]),
                                                download=True)
    elif args.dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(root="mnist_train", train=True, 
                                        transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        val_dataset = torchvision.datasets.MNIST(root="mnist_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)
    elif args.ataset == "Fashion_MNIST":
        train_dataset = torchvision.datasets.FashionMNIST(root="fashion_mnist_train", train=True, 
                                        transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        ]),
                                        download=True)
        val_dataset = torchvision.datasets.FashionMNIST(root="fashion_mnist_val",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    download=True)

    else:
        raise NotImplementedError()


    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    loss = []
    train_loss = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_func.to(device)
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            output = model(batch[0].to(device))
            loss.append(torch.Tensor(loss_func(output, batch[1].to(device))).detach().cpu())
        for batch in train_dataloader:
            output = model(batch[0].to(device))
            train_loss.append(torch.Tensor(loss_func(output, batch[1].to(device))).detach().cpu())
    loss = torch.concatenate(loss)
    train_loss = torch.concatenate(train_loss)
    epsilon_list = []

    for num_cluster in num_clusters:
        local_epsilon = []
        for k in range(10):
            centroids = select_partition_centroid(num_cluster, train_dataset)
            train_indices = assign_partition(train_dataset, centroids)
            test_indices = assign_partition(val_dataset, centroids)
            max_index = torch.max(test_indices)
            epsilon = 0.0
            for i in range(max_index + 1):
                train_loss_values = train_loss[(train_indices==i).nonzero()]
                loss_values = loss[(test_indices==i).nonzero()]
            
                if loss_values.shape[0] < 1 or train_loss_values.shape[0] < 1:
                    continue
                loss_subtraction = torch.abs(torch.cdist(loss_values, train_loss_values, p=1))
                epsilon = max(epsilon, torch.max(loss_subtraction.reshape(-1)).item())
            local_epsilon.append(epsilon)
        epsilon_list.append(f"{torch.mean(torch.Tensor(local_epsilon)).item()}+-{torch.var(torch.Tensor(local_epsilon)).item()}")
    print(epsilon_list)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_robustness(args)
