from utils.utils import default_config, add_dict_to_argparser, create_model_from_config, load_dataset, loss_l1
from utils.split_partitions import *
from torch.utils import data

import torch
import argparse
import numpy as np

num_clusters =  [100, 1000, 5000, 10000]

def cal_local_robustness(args):
    model = create_model_from_config(args)
    model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(model_checkpoint["state_dict"])

    train_dataset, val_dataset = load_dataset(args.dataset)
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_dataloader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    val_output = []
    train_output = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # loss_func.to(device)
    model.eval()

    ## Feed dataset through nn
    with torch.no_grad():
        for batch in val_dataloader:
            output = model(batch[0].to(device))
            val_output.append(output.detach().cpu())
        for batch in train_dataloader:
            output = model(batch[0].to(device))
            train_output.append(output.detach().cpu())
    val_output = torch.concatenate(val_output)
    train_output = torch.concatenate(train_output)
    print(val_output)
    print(train_output)

    epsilon_bound_2_list = []

    for num_cluster in num_clusters:
        temp_epsilon_bound_2_list = []

        for _ in range(10):
            centroids = select_partition_centroid(num_cluster, val_dataset)
            train_indices = assign_partition(train_dataset, centroids)
            test_indices = assign_partition(val_dataset, centroids)
            max_index = torch.max(test_indices)
            train_cluster_shape = []
            cluster_epsilon_list = []

            for i in range(max_index + 1):
                model_train_output = train_output[(train_indices==i).nonzero()]
                model_val_output = val_output[(test_indices==i).nonzero()]
            
                if model_train_output.shape[0] < 1 or model_val_output.shape[0] < 1:
                    continue
                train_cluster_shape.append(train_indices.shape[0])
                print(model_val_output.shape)
                print(model_train_output.shape)
                output_subtraction = torch.abs(torch.cdist(model_val_output, model_train_output, p=1))
                cluster_epsilon = torch.max(output_subtraction.reshape(-1)).item()
                cluster_epsilon_list.append(cluster_epsilon)

            epsilon_bound_2 = np.sum(np.array(train_cluster_shape) * np.array(cluster_epsilon_list)) / np.sum(train_cluster_shape)
            temp_epsilon_bound_2_list.append(epsilon_bound_2)

        epsilon_bound_2_list.append(f"{torch.mean(torch.Tensor(temp_epsilon_bound_2_list)).item()}+-{torch.var(torch.Tensor(temp_epsilon_bound_2_list)).item()}")
    print(f"epsilon bound 3 {epsilon_bound_2_list}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_local_robustness(args)
