from torch.utils.data import Dataset
import random
from torch import Tensor
import torch

def select_partition_centroid(
    num_partitions: int,
    train_dataset: Dataset
    ):
    index_choices = random.choices(range(len(train_dataset)), 
                                   k=num_partitions)
    centroids = []
    for each in index_choices:
        centroids.append(train_dataset[each][0].reshape(-1))
    return torch.vstack(centroids)

def assign_partition(
    test_dataset: Dataset,
    centroids: Tensor
    ):
    val_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=len(test_dataset), 
        shuffle=False)
    for batch in val_loader:
        data = batch[0]

    batch, _, __, ___ = data.shape
    distance = torch.cdist(data.reshape(batch, -1), centroids, p=2)
    cluster_indices = torch.argmin(distance, dim=1)
    return cluster_indices

def calculate_robustness(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    loss_func,
    ):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    loss = []
    train_loss = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch[0])
            loss.append(torch.Tensor(loss_func(output, batch[1])))
        for batch in train_dataloader:
            output = model(batch[0])
            train_loss.append(torch.Tensor(loss_func(output, batch[1])))
    loss = torch.concatenate(loss)
    train_loss = torch.concatenate(train_loss)
    max_index = torch.max(test_indices)
    epsilon = 0.0
    for i in range(max_index + 1):
        train_loss_values = train_loss[(train_indices==i).nonzero()]
        loss_values = loss[(test_indices==i).nonzero()]
        
        if loss_values.shape[0] < 1 or train_loss_values.shape[0] < 1:
            continue
        loss_subtraction = torch.abs(torch.cdist(loss_values, train_loss_values, p=1))
        epsilon = max(epsilon, torch.max(loss_subtraction.reshape(-1)).item())

    return epsilon
