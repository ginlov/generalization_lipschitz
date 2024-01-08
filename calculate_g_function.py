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

def cal_g3(k, sigma, total_num_items, cluster_num_item, list_of_a, TS):
    a0 = max(list_of_a)
    g3_first = torch.sqrt(torch.log(2*k/sigma)) / total_num_items * torch.sum(torch.sqrt(torch.tensor(cluster_num_item))*(a0 + torch.sqrt(torch.tensor(2))*torch.tensor(list_of_a)))
    g3_second = 2*torch.log(2*k / sigma) / total_num_items* (a0 * TS+ torch.sum(torch.tensor(list_of_a)))
    return g3_first + g3_second

def the_rest_of_theorem_five(list_of_a, list_of_local_loss, list_of_num_item, num_items):
    return torch.sum(torch.tensor(list_of_num_item) * (torch.tensor(list_of_a) - torch.tensor(list_of_local_loss))) / num_items
    

def cal_g_function(args):
    model = create_model_from_config(args)
    if args.model not in ["resnet18_imagenet", "regnet_imagenet"]:
        model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
        model.load_state_dict(model_checkpoint["state_dict"])

    ## Load dataset
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
    train_loss = torch.concatenate(train_loss)
    valid_loss = torch.concatenate(valid_loss)

    ## Calculate C - maximum value of the loss
    C_temp = torch.max(valid_loss).item()
    C = torch.max(train_loss).item()
    C = max(C_temp, C)

    # num_items = train_loss.shape[0]
    print("Train loss by L1 loss: {}".format(torch.mean(train_loss).item()))

    for num_cluster in num_clusters:
        print("ok")
        g_temp_values = {
            "one": [],
            "two": [],
            "three": []
        }
        list_of_num_item = []
        list_of_a = []
        list_of_local_loss = []
        for _ in range(1):
            print(f"Start {_} time")

            ## Asign centroids to validation dataset
            centroids = select_partition_centroid(num_cluster, valid_dataset)
            train_indices = assign_partition(train_dataset, centroids)
            valid_indices = assign_partition(valid_dataset, centroids)
            
            unique_ids = torch.unique(train_indices)
            for each in unique_ids:
                print(each.shape)
                print(train_indices.shape)
                print(train_indices == each)
                cluster_loss = train_loss[train_indices == each]
                cluster_valid_loss = valid_loss[valid_indices == each]
                list_of_num_item.append(cluster_loss.shape[0])
                list_of_local_loss.append(cluster_loss.mean().item())
                if cluster_valid_loss.shape[0] != 0:
                    list_of_a.append(torch.concatenate([cluster_loss, cluster_valid_loss], dim=0).mean().item())
                else:
                    list_of_a.append(cluster_loss.mean().item())

            ## Count number of clusters have items
            TD = unique_ids.shape[0]

            print(f"{_} time almost done")

            ## Calculate the rest of theorem the_rest_of_theorem_five
            the_rest = the_rest_of_theorem_five(list_of_local_loss=list_of_local_loss,
                                                list_of_a = list_of_a,
                                                list_of_num_item=list_of_num_item,
                                                num_items=num_items)

            print(f"the rest of theorem 5 {the_rest.item()}")
            ## Calculate g function
            for key, sigma_value in sigma.items():
                g_value = cal_g3(k=num_cluster, sigma=sigma_value, total_num_items=num_items, cluster_num_item = list_of_num_item, list_of_a = list_of_a, TS=TD)

                g_temp_values[key].append(g_value.item())

        ## Print out results
        for key in g_temp_values.keys():
            temp = torch.tensor(g_temp_values[key])
            print(f"Num cluster {num_cluster} sigma {sigma[key]}, values {torch.mean(torch.Tensor(temp)).item()}+-{torch.var(torch.Tensor(temp)).item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_g_function(args)
