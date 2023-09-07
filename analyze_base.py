import os
import torch
import numpy as np
import argparse

from constant import SELECTED_LAYERS, VISUALIZE_LAYERS
from matplotlib import pyplot as plt
from utils import default_config, add_dict_to_argparser


def analyze(norm_type, model):
    selected_layers = SELECTED_LAYERS[model]
    visualize_layers = VISUALIZE_LAYERS[model]
    if not os.path.isdir("image"):
        os.mkdir("image")

    log_folder = "_".join([model, norm_type])
    assert os.path.isdir(log_folder), "Log folder does not exist"

    file_names = os.listdir(f"{log_folder}/variance")
    variance = {}

    # Merge saved mean and variance from all files into one dictionary. 
    for file in file_names:
        _, epoch, iters = file.split(".")[0].split("_")

        # HARD CODE NUMBER OF EPOCHS
        if int(epoch) > 40:
            continue

        # global iter counted from training
        global_iter = int(epoch) * 782 + int(iters)
        data = torch.load(f"{log_folder}/variance/{file}")
        mean_var = []
        var_var = []
        for each in data:
            mean_var.append(torch.mean(each.view(-1)).item())
            var_var.append(torch.var(each.view(-1)).item())
        mean_var = [mean_var[i] for i in selected_layers]
        var_var = [var_var[i] for i in selected_layers]
        variance[global_iter] = {
            "mean": mean_var,
            "var": var_var
        }

    # sort by iter order
    variance = dict(sorted(variance.items(), key=lambda x: x[0]))

    # extracts visualize layers
    visualize_mean = [{}, {}, {}, {}]
    visualize_var = [{}, {}, {}, {}]
    for i, key in enumerate(variance.keys()):
        if i % 5 == 0:
            for j in range(4):
                visualize_mean[j][key] = variance[key]['mean'][visualize_layers[j]]
                visualize_var[j][key] = np.sqrt(variance[key]['var'][visualize_layers[j]])

    # draw image
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('minibatch', fontdict={'fontsize': 18})
    ax.set_ylabel(r'$\sigma^{2}$', fontdict={'fontsize': 18})
    for i in range(4):
        ax.plot(list(visualize_mean[i].keys()), list(visualize_mean[i].values()),
                label=f"{norm_type}, Layer {visualize_layers[i] + 1}")
        ax.fill_between(list(visualize_mean[i].keys()),
                        np.clip(np.array(list(visualize_mean[i].values()))
                                - np.array(list(visualize_var[i].values())), 0, None),
                        np.array(list(visualize_mean[i].values())) + np.array(list(visualize_var[i].values())),
                        alpha=0.3)
    ax.legend(fontsize=12, loc="upper left")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.tight_layout()
    if norm_type == "BN":
        plt.savefig(f"image/{model}_{norm_type}.png", dpi=250)
    elif norm_type == "GN":
        plt.savefig(f"image/{model}_{norm_type}.png", dpi=250)
    elif norm_type == "LN":
        plt.savefig(f"image/{model}_{norm_type}.png", dpi=250)
    else:
        plt.savefig(f"image/{model}_wo_norm.png", dpi=250)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    args = parser.parse_args()

    norm_type = ""
    log_file = ""
    if args.model_type == 0:
        norm_type = "wo_norm"
    elif args.model_type == 2 or args.model_type == 1:
        norm_type = args.norm_type
    else:
        raise NotImplementedError()

    analyze(norm_type, args.model)
