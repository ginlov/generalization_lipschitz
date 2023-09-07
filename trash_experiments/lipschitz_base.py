import torch
import argparse
from model._mlp import _mlp
from model._vgg import _vgg
from model._resnet import _resnet, BasicBlock

def lipschitz_cal(model, dataset):
    for i, cluster in enumerate(dataset):
        datapoint = []
        for j in range(150):
            datapoint.append(torch.Tensor(cluster['centroid']) + cluster['diameter'] / 2 * torch.rand(*cluster['centroid'].shape))

        datapoint = torch.stack(datapoint)
        datapoint.to('cuda')
        model.to('cuda')
        model.eval()
        with torch.no_grad():
            output = model(datapoint.view(150, 3, 32, 32).to('cuda')).detach().cpu()

        norm_max_output = torch.nn.functional.pdist(output, p=torch.inf)
        norm_max_input = torch.nn.functional.pdist(datapoint, p=torch.inf)
        norm_1_input = torch.nn.functional.pdist(datapoint, p=1)
        norm_2_input = torch.nn.functional.pdist(datapoint, p=2)

        lipschitz = norm_max_output / norm_max_input
        # dataset[i]['lipschitz_const'] = torch.max(lipschitz).item()  
        # dataset[i]['lipschitz_const_1'] = torch.max(norm_max_output / norm_1_input).item()
        dataset[i]['lipschitz_const_2'] = torch.max(norm_max_output/ norm_2_input).item()

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--model_type", type=int, default=0)
    parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()

    batch_norm = False
    if args.model_type == 0:
        batch_norm = False
    elif args.model_type == 2:
        batch_norm = True
    else:
        raise NotImplementedError()
    
    if args.model == 0:
        hidden_layer = [200, 200, 200, 200, 200, 200, 200, 200]
        model = _mlp(3 * 32 * 32, cfg=hidden_layer, batch_norm=batch_norm, num_classes=10)
    elif args.model == 1:
        model = _vgg("D", batch_norm=False, init_weights=batch_norm, num_classes=10)
    elif args.model == 2:
        if batch_norm:
            norm_layer = torch.nn.BatchNorm2d
        else:
            norm_layer = None
        model = _resnet(BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer, num_classes=10)
    else:
        raise NotImplementedError()
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    
    dataset = torch.load("cluster_info.pth")
    dataset = lipschitz_cal(model, dataset)
    torch.save(dataset, "lipschitz.pth")
