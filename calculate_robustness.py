from utils.utils import default_config, add_dict_to_argparser, create_model_from_config, load_dataset, loss_l1, CustomDataset
from utils.split_partitions import *
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import torch
import argparse
import numpy as np
import joblib

num_clusters =  [100, 1000, 5000, 10000]

def cal_robustness(args):
    model = create_model_from_config(args)
    if args.model_checkpoint.endswith(".pth.tar"):
        model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
        model.load_state_dict(model_checkpoint["state_dict"])
    elif args.model_checkpoint.endswith(".pkl"):
        model = joblib.load(args.model_checkpoint)

    train_dataset, val_dataset = load_dataset(args.dataset)
    if isinstance(train_dataset, CustomDataset):
        length_of_data = len(train_dataset.x)
        idx = np.random.choice(np.arange(length_of_data), 50000, replace=True)
        train_dataset = CustomDataset(
            x=train_dataset.x[idx],
            y=np.asarray(train_dataset.y)[idx],
            transform=transforms.ToTensor()
        )
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_dataloader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    loss_func = loss_l1

    loss = []
    train_loss = []
    if args.model_checkpoint.endswith(".pth.tar"):
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
    elif args.model_checkpoint.endswith(".pkl"):
        for batch in val_dataloader:
            output = model.predict(batch[0].reshape(batch[0].shape[0], -1).numpy())
            loss.append(torch.Tensor(loss_func(output, batch[1])).detach())
        for batch in train_dataloader:
            output = model.predict(batch[0].reshape(batch[0].shape[0], -1).numpy())
            train_loss.append(torch.Tensor(loss_func(output, batch[1])).detach())
        loss = torch.cat(loss)
        train_loss = torch.cat(train_loss)

    epsilon_list = []
    epsilon_bound_2_list = []
    epsilon_bound_3_list = []
    localsen_list = []
    localavg_list = []

    for num_cluster in tqdm(num_clusters, desc="Processing clusters"):
        temp_epsilon_list = []
        temp_epsilon_bound_2_list = []
        temp_epsilon_bound_3_list = []
        temp_localsen = []
        temp_localavg = []

        for _ in tqdm(range(10), desc=f"Processing {num_cluster} clusters"):
            centroids = select_partition_centroid(num_cluster, val_dataset)
            train_indices = assign_partition(train_dataset, centroids)
            test_indices = assign_partition(val_dataset, centroids)
            max_index = torch.max(test_indices)
            train_cluster_shape = []
            cluster_epsilon_list = []
            bound_3_list = []
            localavg =[]
            localsen = []
            epsilon = 0.0

            for i in range(max_index + 1):
                train_loss_values = train_loss[(train_indices==i).nonzero()]
                loss_values = loss[(test_indices==i).nonzero()]
            
                if loss_values.shape[0] < 1 or train_loss_values.shape[0] < 1:
                    continue
                localavg.append(torch.mean(torch.concatenate([train_loss_values, loss_values])).item())
                train_cluster_shape.append(train_loss_values.shape[0])
                loss_subtraction = torch.abs(torch.cdist(loss_values, train_loss_values, p=1))
                localsen.append(torch.mean(loss_subtraction).item())
                cluster_epsilon = torch.max(loss_subtraction.reshape(-1)).item()
                cluster_epsilon_list.append(cluster_epsilon)
                epsilon = max(epsilon, cluster_epsilon)
                #bound_3 = max(torch.max(train_loss_values).item(), torch.max(loss_values).item()) - torch.mean(train_loss_values).item()
                #bound_3_list.append(bound_3)
            temp_localsen.append(np.sum(np.array(train_cluster_shape) * np.array(localsen)) / np.sum(train_cluster_shape))
            temp_localavg.append(np.sum(np.array(train_cluster_shape) * np.array(localavg)) / np.sum(train_cluster_shape))
            epsilon_bound_2 = np.sum(np.array(train_cluster_shape) * np.array(cluster_epsilon_list)) / np.sum(train_cluster_shape)
            #epsilon_bound_3 = np.sum(np.array(train_cluster_shape) * np.array(bound_3_list)) / np.sum(train_cluster_shape)
            temp_epsilon_bound_2_list.append(epsilon_bound_2)
            #temp_epsilon_bound_3_list.append(epsilon_bound_3)
            temp_epsilon_list.append(epsilon)
        localsen_list.append(f"{torch.mean(torch.Tensor(temp_localsen)).item()}+-{torch.var(torch.Tensor(temp_localsen)).item()}")
        localavg_list.append(f"{torch.mean(torch.Tensor(temp_localavg)).item()}+-{torch.var(torch.Tensor(temp_localavg)).item()}")
        epsilon_list.append(f"{torch.mean(torch.Tensor(temp_epsilon_list)).item()}+-{torch.var(torch.Tensor(temp_epsilon_list)).item()}")
        epsilon_bound_2_list.append(f"{torch.mean(torch.Tensor(temp_epsilon_bound_2_list)).item()}+-{torch.var(torch.Tensor(temp_epsilon_bound_2_list)).item()}")
        #epsilon_bound_3_list.append(f"{torch.mean(torch.Tensor(temp_epsilon_bound_3_list)).item()}+-{torch.var(torch.Tensor(temp_epsilon_bound_3_list)).item()}")
    print(f"Train loss {torch.mean(train_loss).item()}")
    print(f"Rob {epsilon_list}")
    print(f"LocalRob {epsilon_bound_2_list}")
    # print(f"epsilon bound 4 {epsilon_bound_3_list}")
    print(f"localsen {localsen_list}")
    print(f"Localavg {localavg_list}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    parser.add_argument("--model_checkpoint", type=str, default="model_best.pth.tar")
    args = parser.parse_args()
    cal_robustness(args)
