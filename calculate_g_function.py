from utils.utils import default_config, add_dict_to_argparser, create_model_from_config, load_dataset, loss_l1, CustomDataset
from utils.split_partitions import *
from torch.utils import data
from torchvision import transforms

import torch
import argparse
import math

num_clusters = [100, 1000, 5000, 10000]
sigma = {
    "one": 0.01,
    "two": 0.05,
    "three": 0.1
}

def cal_g_function(args):
    model = create_model_from_config(args)
    if args.model != "resnet18_imagenet":
        model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
        model.load_state_dict(model_checkpoint["state_dict"])

    train_dataset, valid_dataset = load_dataset(args.dataset)
    num_items = len(train_dataset)
    if isinstance(train_dataset, CustomDataset):
        length_of_data = len(train_dataset.x)
        idx = np.random.choice(np.arange(length_of_data), 50000, replace=True)
        train_dataset = CustomDataset(
            x=train_dataset.x[idx],
            y=np.asarray(train_dataset.y)[idx],
            transform=transforms.ToTensor()
        )
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=128, shuffle=False)

    loss_func = loss_l1
    train_loss = []
    valid_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in train_dataloader:
            output = model(batch[0].to(device))
            train_loss.append(torch.Tensor(loss_func(output, batch[1].to(device))).detach().cpu())
        for batch in valid_dataloader:
            output = model(batch[0].to(device))
            valid_loss.append(torch.Tensor(loss_func(output, batch[1].to(device))).detach().cpu())

    train_loss = torch.concatenate(train_loss)
    valid_loss = torch.concatenate(valid_loss)
    C_temp = torch.max(valid_loss).item()
    C = torch.max(train_loss).item()
    C = max(C_temp, C)
    # num_items = train_loss.shape[0]
    print("Train loss by L1 loss: {}".format(torch.mean(train_loss).item()))

    for num_cluster in num_clusters:
        g_temp_values = {
            "one": [],
            "two": [],
            "three": []
        }
        for _ in range(10):
            centroids = select_partition_centroid(num_cluster, valid_dataset)
            train_indices = assign_partition(train_dataset, centroids)
            TD = torch.unique(train_indices).shape[0]
            for key, sigma_value in sigma.items():
                g_value = C * ((math.sqrt(2)+1)*math.sqrt(TD * math.log(2*num_cluster/sigma_value)/num_items) + 2*TD*math.log(2*num_cluster/sigma_value)/num_items)
                g_temp_values[key].append(g_value)
        for key in g_temp_values.keys():
            temp = torch.tensor(g_temp_values[key])
            print(f"Num cluster {num_cluster} sigma {sigma[key]}, values {torch.mean(torch.Tensor(temp)).item()}+-{torch.var(torch.Tensor(temp)).item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_g_function(args)
