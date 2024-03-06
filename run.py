from utils.train_base import train
from utils.utils import default_config, add_dict_to_argparser, create_model_from_config, load_dataset, loss_l1, CustomDataset
from utils.split_partitions import select_partition_centroid, assign_partition
from utils.wandb_utils import wandb_init, wandb_end
from torch.utils import data
from torchvision import transforms

import argparse
import numpy as np
import torch
import pandas as pd

num_cluster_list = [100, 1000, 5000, 10000]
sigma_list = [0.01, 0.05, 0.1]
# sigma = 0.01

def start_train(
        args: argparse.Namespace,
    ):
    # create model
    model = create_model_from_config(args)
    dataset = args.dataset
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
        "log_folder": log_folder,
        "epoch": args.num_epoch,
        "learning_rate": args.learning_rate,
        "config_weight_decay": args.weight_decay,
        "config_optimizer": args.optimizer
    }
    return train(**training_config)

def cal_g3(k, sigma, total_num_items, cluster_num_item, list_of_a, TS):
    a0 = max(list_of_a)
    k = torch.tensor(k)
    sigma = torch.tensor(sigma)
    total_num_items = torch.tensor(total_num_items)
    cluster_num_item = torch.tensor(cluster_num_item)
    list_of_a = torch.tensor(list_of_a)
    TS = torch.tensor(TS)
    g3_first = torch.sqrt(torch.log(2*k/sigma)) / total_num_items * torch.sum(torch.sqrt(cluster_num_item)*(a0 + torch.sqrt(torch.tensor(2))*list_of_a))
    g3_second = 2*torch.log(2*k / sigma) / total_num_items * (a0 * TS + torch.sum(list_of_a))
    return g3_first + g3_second


def the_rest_of_theorem_five(list_of_a, list_of_local_loss, list_of_num_item, num_items):
    return torch.sum(torch.tensor(list_of_num_item) * (torch.tensor(list_of_a) - torch.tensor(list_of_local_loss))) / num_items


def cal_related_terms(args):
    model = create_model_from_config(args)
    if args.model not in ["resnet18_imagenet", "regnet_imagenet"]:
        model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
        model.load_state_dict(model_checkpoint["state_dict"])

    # Load dataset
    train_dataset, valid_dataset = load_dataset(args.dataset)
    num_items = len(train_dataset)
    ## CustomDataset is with agumentation thus leads to many data points
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

    ## Set up loss function, device, ...
    loss_func = loss_l1
    train_loss = []
    valid_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    ## Feed forward to get loss values
    with torch.no_grad():
        for batch in train_dataloader:
            output = model(batch[0].to(device))
            train_loss.append(torch.Tensor(loss_func(output, batch[1].to(device))).detach().cpu())
        for batch in valid_dataloader:
            output = model(batch[0].to(device))
            valid_loss.append(torch.Tensor(loss_func(output, batch[1].to(device))).detach().cpu())
    train_loss = torch.concat(train_loss)
    valid_loss = torch.concat(valid_loss)

    C_temp = torch.max(valid_loss).item()
    C = torch.max(train_loss).item()
    C = max(C_temp, C)

    print("Averagte train loss by L1 loss: {}".format(torch.mean(train_loss).item()))

    list_of_rows = []
    for num_cluster in num_cluster_list:
        five_times_g_values = [[], [], []]
        five_times_robustness = []
        five_times_theorem_five = []
        five_times_local_robustness = []
        
        for _ in range(1):
            list_of_num_item = []
            list_of_a = []
            list_of_local_loss = []

            local_robustness_cluster_shape = []
            local_robustness = []

            centroids = select_partition_centroid(num_cluster, valid_dataset)
            train_indices = assign_partition(train_dataset, centroids)
            valid_indices = assign_partition(valid_dataset, centroids)

            unique_ids = torch.unique(train_indices)
            for each in unique_ids:
                cluster_loss = train_loss[train_indices == each]
                cluster_valid_loss = valid_loss[valid_indices == each]
                list_of_num_item.append(cluster_loss.shape[0])
                list_of_local_loss.append(cluster_loss.mean().item())
                if cluster_valid_loss.shape[0] != 0:
                    list_of_a.append(torch.concat([cluster_loss, cluster_valid_loss], dim=0).mean().item())

                    #local robustness
                    local_robustness_cluster_shape.append(cluster_loss.shape[0])
                    loss_subtraction = torch.abs(torch.cdist(cluster_valid_loss.reshape(-1, 1), cluster_loss.reshape(-1, 1)))
                    a_local_robustness = torch.max(loss_subtraction.reshape(-1)).item()
                    local_robustness.append(a_local_robustness)
                else:
                    list_of_a.append(cluster_loss.mean().item())

            TD = unique_ids.shape[0]
            for i, sigma in enumerate(sigma_list):
                g_value = cal_g3(k=num_cluster, sigma=sigma, total_num_items=num_items, cluster_num_item=list_of_num_item, list_of_a=list_of_a, TS=TD)
                five_times_g_values[i].append(g_value)

            total_loss_subtraction = torch.abs(torch.cdist(torch.tensor(valid_loss).reshape(-1, 1), torch.tensor(train_loss).reshape(-1, 1)))
            robustness = torch.max(total_loss_subtraction.reshape(-1)).item()
            local_robustness = np.sum(np.array(local_robustness_cluster_shape)*np.array(local_robustness)) / np.sum(local_robustness_cluster_shape)

            the_rest_theorem_five = the_rest_of_theorem_five(list_of_local_loss=list_of_local_loss, list_of_a = list_of_a, list_of_num_item=list_of_num_item, num_items=num_items)
            five_times_robustness.append(robustness)
            five_times_local_robustness.append(local_robustness)
            five_times_theorem_five.append(the_rest_theorem_five)
        list_of_rows.append({
            "model": args.model,
            "dataset": args.dataset,
            "lr": args.learning_rate,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "loss_l1": torch.mean(train_loss).item(),
            "num_cluster": num_cluster,
            f"g_value {sigma_list[0]} mean": torch.mean(torch.tensor(five_times_g_values[0])).item(),
            # f"g_value {sigma_list[0]} var": torch.var(torch.tensor(five_times_g_values[0])).item(),
            f"g_value {sigma_list[1]} mean": torch.mean(torch.tensor(five_times_g_values[1])).item(),
            # f"g_value {sigma_list[1]} var": torch.var(torch.tensor(five_times_g_values[1])).item(),
            f"g_value {sigma_list[2]} mean": torch.mean(torch.tensor(five_times_g_values[2])).item(),
            # f"g_value {sigma_list[2]} var": torch.var(torch.tensor(five_times_g_values[2])).item(),

            "theorem 3 mean": torch.mean(torch.tensor(five_times_robustness)).item(),
            # "theorem 3 variance": torch.var(torch.tensor(five_times_robustness)).item(),
            "theorem 4 mean": torch.mean(torch.tensor(five_times_local_robustness)).item(),
            # "theorem 4 variance": torch.var(torch.tensor(five_times_local_robustness)).item(),
            "theorem 5 mean": torch.mean(torch.tensor(five_times_theorem_five)).item(),
            # "theorem 5 variance": torch.var(torch.tensor(five_times_theorem_five)).item()
        })

    return pd.DataFrame(list_of_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())
    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    parser.add_argument("--num_runs", type=int, default=5)

    args = parser.parse_args()
    list_of_dfs = []
    for i in range(args.num_runs):
        train_best_loss, train_best_acc1, _, best_loss, best_acc1, __= start_train(args)

        output_df = cal_related_terms(args)
        output_df["train_loss"] = [train_best_loss]*output_df.shape[0]
        output_df["train_acc"] = [train_best_acc1.detach().cpu().item()]*output_df.shape[0]
        output_df["loss"] = [best_loss]*output_df.shape[0]
        output_df["acc"] = [best_acc1.detach().cpu().item()]*output_df.shape[0]

        list_of_dfs.append(output_df)
    final_df = pd.concat(list_of_dfs)
    final_df.to_excel(f"{args.model}_{args.dataset}_{args.learning_rate}_{args.optimizer}_{args.weight_decay}.xlsx", index=False)
