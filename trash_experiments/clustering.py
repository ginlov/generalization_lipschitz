from torchvision import datasets, transforms
import torch

from k_means_max_norm import KMeans

def main():
    def transform(img: torch.Tensor):
        return img * 2 - 1.0
    
    train_dataset = datasets.CIFAR10(root="cifar_train", train=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transform,
                                     ]),
                                     download=True
                                     )
    val_dataset = datasets.CIFAR10(root="cifar_val", train=False, 
                                   transform=transforms.Compose([
                                   # transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transform,
                                   ]),
                                   download=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True,
        num_workers=5, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False,
        num_workers=5, pin_memory=True)
    
    for data in val_loader:
        fit_data = data[0].detach().cpu().numpy()
        fit_label = data[1].detach().cpu().numpy()

    kmeans = KMeans(n_clusters=10, max_iter=200)
    kmeans.fit(fit_data, fit_label)