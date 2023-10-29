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

    _, valid_dataset = load_dataset(args.dataset)

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
    _, num_class = model_output.shape
    model_output = torch.nn.functional.one_hot(torch.argmax(model_output, dim=1), num_class).float()

    for num_cluster in num_clusters:
        bound_5_list = []

        for _ in range(10):
            centroids = select_partition_centroid(num_cluster, valid_dataset)
            indices = assign_partition(pseudo_dataset, centroids)
            max_index = torch.max(indices)
            cluster_shape = []
            cluster_lipschitz_list = []
            cluster_diameter = []

            for i in range(max_index + 1):
                model_input_values = pseudo_dataset.X[(indices==i).nonzero()]
                model_output_values = model_output[(indices==i).nonzero()]
                number_items = model_input_values.shape[0]

                if number_items < 2:
                    continue

                model_input_values = model_input_values.reshape(number_items, -1)
                model_output_values = model_output_values.reshape(number_items, -1)
                cluster_shape.append(number_items)
                model_output_subtraction = torch.abs(torch.nn.functional.pdist(model_output_values, p=1))
                model_input_subtraction = torch.abs(torch.nn.functional.pdist(model_input_values, p=1))
                cluster_lipschitz = torch.max((model_output_subtraction/model_input_subtraction).reshape(-1)).item()
                cluster_lipschitz_list.append(cluster_lipschitz)
                cluster_diameter.append(torch.max(torch.abs(torch.nn.functional.pdist(model_input_values, p=1))).item())

            bound_5 = np.sum(np.array(cluster_shape) * np.array(cluster_diameter) * np.array(cluster_lipschitz_list)) / np.sum(np.array(cluster_shape))
            bound_5_list.append(bound_5)
        print(f"Num cluster {num_cluster}, values {torch.mean(torch.Tensor(bound_5_list)).item()}+-{torch.var(torch.Tensor(bound_5_list)).item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_lipschitz(args)
