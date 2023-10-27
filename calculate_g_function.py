from utils.utils import default_config, add_dict_to_argparser, create_model_from_config, load_dataset, loss_l1
from utils.split_partitions import *
from torch.utils import data

import torch
import argparse
import math

num_clusters = [100, 1000, 5000, 10000]
sigma = 1e-5

def cal_g_function(args):
    model = create_model_from_config(args)
    model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(model_checkpoint["state_dict"])

    train_dataset, valid_dataset = load_dataset(args.dataset)

    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=False)

    loss_func = loss_l1
    train_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in train_dataloader:
            output = model(batch[0].to(device))
            train_loss.append(torch.Tensor(loss_func(output, batch[1].to(device))).detach().cpu())

    train_loss = torch.concatenate(train_loss)
    C = torch.max(train_loss)
    num_items = train_loss.shape[0]

    for num_cluster in num_clusters:
        g_temp_values = []
        for _ in range(10):
            centroids = select_partition_centroid(num_cluster, valid_dataset)
            train_indices = assign_partition(train_dataset, centroids)
            TD = torch.unique(train_indices).shape[0]
            g_value = C * ((math.sqrt(2)+1)*math.sqrt(TD * math.log(2*num_cluster/sigma)/num_items) + 2*TD*math.log(2*num_cluster/sigma)/num_items)
            g_temp_values.append(g_value)
        g_temp_values = torch.concatenate(g_temp_values)
        print(f"Num cluster {num_cluster}, values {torch.mean(train_loss).item() + torch.mean(torch.Tensor(g_temp_values)).item()}+-{torch.var(torch.Tensor(g_temp_values)).item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_g_function(args)
