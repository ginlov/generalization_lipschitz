from utils.utils import default_config, add_dict_to_argparser, create_model_from_config, load_dataset, loss_l1
from utils.split_partitions import *
from torch.utils import data

import torch
import argparse
import numpy as np

num_clusters =  [100, 1000, 5000, 10000]

def cal_robustness(args):
    model = create_model_from_config(args)
    model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(model_checkpoint["state_dict"])

    train_dataset, val_dataset = load_dataset(args.dataset)
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_dataloader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    loss_func = loss_l1

    loss = []
    train_loss = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # loss_func.to(device)
    model.eval()

    ## Feed dataset through nn
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
    epsilon_bound_2_list = []
    epsilon_bound_3_list = []

    for num_cluster in num_clusters:
        temp_epsilon_list = []
        temp_epsilon_bound_2_list = []
        temp_epsilon_bound_3_list = []

        for _ in range(10):
            centroids = select_partition_centroid(num_cluster, val_dataset)
            train_indices = assign_partition(train_dataset, centroids)
            test_indices = assign_partition(val_dataset, centroids)
            max_index = torch.max(test_indices)
            train_cluster_shape = []
            cluster_epsilon_list = []
            bound_3_list = []
            epsilon = 0.0

            for i in range(max_index + 1):
                train_loss_values = train_loss[(train_indices==i).nonzero()]
                loss_values = loss[(test_indices==i).nonzero()]
            
                if loss_values.shape[0] < 1 or train_loss_values.shape[0] < 1:
                    continue
                train_cluster_shape.append(train_indices.shape[0])
                loss_subtraction = torch.abs(torch.cdist(loss_values, train_loss_values, p=1))
                cluster_epsilon = torch.max(loss_subtraction.reshape(-1)).item()
                cluster_epsilon_list.append(cluster_epsilon)
                epsilon = max(epsilon, cluster_epsilon)
                bound_3 = max(torch.max(train_loss_values).item(), torch.max(loss_values).item()) - torch.mean(train_loss_values).item()
                bound_3_list.append(bound_3)

            epsilon_bound_2 = np.sum(np.array(train_cluster_shape) * np.array(cluster_epsilon_list)) / np.sum(train_cluster_shape)
            epsilon_bound_3 = np.sum(np.array(train_cluster_shape) * np.array(bound_3_list)) / np.sum(train_cluster_shape)
            temp_epsilon_bound_2_list.append(epsilon_bound_2)
            temp_epsilon_bound_3_list.append(epsilon_bound_3)
            temp_epsilon_list.append(epsilon)

        epsilon_list.append(f"{torch.mean(torch.Tensor(temp_epsilon_list)).item()}+-{torch.var(torch.Tensor(temp_epsilon_list)).item()}")
        epsilon_bound_2_list.append(f"{torch.mean(torch.Tensor(temp_epsilon_bound_2_list)).item()}+-{torch.var(torch.Tensor(temp_epsilon_bound_2_list)).item()}")
        epsilon_bound_3_list.append(f"{torch.mean(torch.Tensor(temp_epsilon_bound_3_list)).item()}+-{torch.var(torch.Tensor(temp_epsilon_bound_3_list)).item()}")
    print(f"epsilon {epsilon_list}")
    print(f"epsilon bound 2 {epsilon_bound_2_list}")
    print(f"epsilon bound 3 {epsilon_bound_3_list}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_robustness(args)
