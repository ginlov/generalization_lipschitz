"""Examples of computing Jacobian bounds.

We use a small model with two convolutional layers and dense layers respectively.
The width of the model has been reduced for the demonstration here. And we use
data from CIFAR-10.

We show examples of:
- Computing Jacobian bounds
- Computing Linf local Lipschitz constants
- Computing JVP bounds
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
from model._mlp import MLP
import gc
def build_model(in_ch=3, in_dim=32, width=32, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(width * (in_dim-4)**2, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


torch.manual_seed(0)

# Create a small model and load pre-trained parameters.

# Prepare the dataset
test_data = datasets.CIFAR10('./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2009, 0.2009, 0.2009])]))

"""
# Example 1: Convert the model for Jacobian bound computation
model = BoundedModule(model_ori, x0, device=device)
model.augment_gradient_graph(x0)

# Sanity check to ensure that the new graph matches the original gradient computation
y = model_ori(x0.requires_grad_(True))
ret_ori = torch.autograd.grad(y.sum(), x0)[0].view(1, -1)
# After running augment_gradient_graph, the model takes an additional input
# (the second input) which is a linear mapping applied on the output of the
# model before computing the gradient. It is the same as "grad_outputs" in
# torch.autograd.grad, which is "the 'vector' in the vector-Jacobian product".
# Here, setting torch.ones(1, 10) is equivalent to computing the gradients for
# y.sum() above.
ret_new = model(x0, torch.ones(1, 10).to(x0))
assert torch.allclose(ret_ori, ret_new)

for eps in [0]:
    # The input region considered is an Linf ball with radius eps around x0.
    x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))
    # Compute the Linf locaal Lipscphitz constant
    lower, upper = model.compute_jacobian_bounds(x)
    print(f'Gap between upper and lower Jacobian bound for eps={eps:.5f}',
          (upper - lower).max())
    if eps == 0:
        assert torch.allclose(ret_new, lower.sum(dim=0, keepdim=True))
        assert torch.allclose(ret_new, upper.sum(dim=0, keepdim=True))

"""
# Example 2: Convert the model for Linf local Lipschitz constant computation

total_lipschitz = 0
list_of_number = []
for i in range(8):
        eps = 1./255
        model_ori = build_model(width=4, linear_size=32)
        model_ori = MLP(in_features = 3*32*32, cfg =  [1024, 512, 256, 64], norm_layer = None, num_classes = 10)
        model_ori.load_state_dict(torch.load('/kaggle/input/checkpointsMLP/model_best.pth.tar')["state_dict"])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_ori = model_ori.to(device)
        x_i = test_data[i][0].unsqueeze(0).to(device)
        torch.cuda.synchronize()
        # The input region considered is an Linf ball with radius eps around x0.
        model = BoundedModule(model_ori, x_i, device=device)
        # Set norm=np.inf for Linf local Lipschitz constant
        model.augment_gradient_graph(x_i, norm=np.inf)
        torch.cuda.synchronize()
        
        # Sanity check to ensure that the new graph matches the original gradient computation
        """
        y = model_ori(x_i.requires_grad_(True))
        ret_ori = torch.autograd.grad(y.sum(), x_i, retain_graph = False)[0].abs().sum().view(-1)
        ret_new = model(x_i, torch.ones(1, 10).to(x_i)).view(-1)
        assert torch.allclose(ret_ori, ret_new)
        """
        x = BoundedTensor(x_i, PerturbationLpNorm(norm=np.inf, eps=eps))
        # Compute the Linf locaal Lipschitz constant
        result = model.compute_jacobian_bounds(x)
        torch.cuda.synchronize()
        list_of_number.append(result)
        total_lipschitz += result * 2/255 * 1/1000
        #print(f'Linf local Lipschitz constant for eps={eps:.5f}', result)
        model.zero_grad()
        model_ori.zero_grad()
        model_ori.cpu()
        model.cpu()
        result.cpu()
        x.cpu()
        del model, model_ori, result, x
        gc.collect()
        torch.cuda.empty_cache()
print("Our term is:" + total_lipschitz)
print(list_of_number)
"""
# Example 3: Convert the model for Jacobian-Vector Product (JVP) computation
model = BoundedModule(model_ori, x0, device=device)
vector = torch.randn(x0.shape).to(x0)
# Set vector for JVP computation
model.augment_gradient_graph(x0, vector=vector)

# Sanity check to ensure that the new graph matches the original JVP
def func(x0):
    return model_ori(x0.requires_grad_(True))
ret_ori = torch.autograd.functional.jvp(func, x0, vector)[-1].view(-1)
ret_new = torch.zeros(10).to(x0)
for i in range(10):
    c = F.one_hot(torch.tensor([i], dtype=torch.long), 10).to(x0)
    ret_new[i] = model(x0, c)
assert torch.allclose(ret_ori, ret_new)

for eps in [0, 1./255, 4./255]:
    # The input region considered is an Linf ball with radius eps around x0.
    x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))
    # Compute the JVP
    lower, upper = model.compute_jacobian_bounds(x)
    print(f'JVP lower bound for eps={eps:.5f}', lower.view(-1))
    print(f'JVP upper bound for eps={eps:.5f}', upper.view(-1))
"""
