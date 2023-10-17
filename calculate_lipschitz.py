from utils.utils import default_config, add_dict_to_argparser, create_model_from_config, load_dataset
from utils.split_partitions import *
from torch.utils import data

import torch
import argparse
import numpy as np

num_clusters =  [100, 1000, 5000, 10000]

def cal_lipschitz(args):
    model = create_model_from_config(args)
    model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(model_checkpoint["state_dict"])

    train_dataset, _ = load_dataset(args.dataset)

    pseudo_dataset = generate_dataset()
    dataloader = data.DataLoader(pseudo_dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    model_output = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch[0].to(device))
            model_output.append(output.detach().cpu())

    model_output = torch.concatenate(model_output)

    for num_cluster in num_clusters:
        bound_5_list = []

        for _ in range(10):
            centroids = select_partition_centroid(num_cluster, train_dataset)
            indices = assign_partition(pseudo_dataset, centroids)
            max_index = torch.max(indices)
            cluster_shape = []
            cluster_lipschitz_list = []

            for i in range(max_index + 1):
                model_input_values = pseudo_dataset.X[(indices==i).nonzero()]
                model_output_values = model_output[(indices==i).nonzero()]
                number_items = model_input_values.shape[0]

                if number_items < 1:
                    continue

                model_input_values = model_input_values.reshape(number_items, -1)
                model_output_values = model_output_values.reshape(number_items, -1)
                cluster_shape.append(number_items)
                model_output_subtraction = torch.abs(torch.nn.functional.pdist(model_output_values, p=1))
                model_input_subtraction = torch.abs(torch.nn.functional.pdist(model_input_values, p=1))
                cluster_lipschitz = torch.max((model_output_subtraction/model_input_subtraction).reshape(-1)).item()
                cluster_lipschitz_list.append(cluster_lipschitz)

            print(f"cluster_shape {cluster_shape}, cluster lips {cluster_lipschitz_list}")
            bound_5 = np.sum(np.array(cluster_shape) * np.array(cluster_lipschitz_list)) / np.sum(np.array(cluster_shape))
            print(f"bound 5 {bound_5}")

            bound_5_list.append(bound_5)
            print(f"Num cluster {num_cluster}, values {torch.mean(torch.Tensor(bound_5_list)).item()}+-{torch.var(torch.Tensor(bound_5_list)).item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_lipschitz(args)
