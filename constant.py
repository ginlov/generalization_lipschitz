from model._resnet import BasicBlock, _resnet, Bottleneck
from model._vgg import _vgg
from model._mlp import _mlp

#############################################
# Resnet50: 56 modified layers ##############
# Resnet34: 39 modified layers ##############
# Resnet: 23 modified layers ################
# VGG: 17 modified layers ###################
# MLP: 10 modifiedlayers ####################
#############################################
SELECTED_LAYERS = {
    "resnet": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "resnet34": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    "resnet50": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                 54, 55],
    "vgg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "mlp": [0, 1, 2, 3, 4, 5, 6, 7]
}

VISUALIZE_LAYERS = {
    "resnet": [4, 9, 14, 19],
    "vgg": [0, 4, 9, 12],
    "mlp": [0, 2, 4, 7],
    "resnet34": [4, 9, 14, 19, 24, 29, 34],
    "resnet50": [4, 14, 24, 34, 44, 54]
}

COMPARE_LAYERS = {
    "resnet": [0, 2, 4, 6, 8, 10, 12, 14, 16, 19],
    "resnet34": [0, 4, 8, 12, 16, 20, 24, 28, 32, 33],
    "resnet50": [0, 6, 12, 18, 24, 30, 36, 42, 48]
}

MODEL_CONFIG = {
    "resnet": {
        "block": BasicBlock,
        "layers": [2, 2, 2, 2],
        "norm_layer": None,
        "num_classes": 10
    },
    "resnet34": {
        "block": BasicBlock,
        "layers": [3, 4, 6, 3],
        "norm_layer": None,
        "num_classes": 10
    },
    "resnet50": {
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "norm_layer": None,
        "num_classes": 10,
    },
    "vgg": {
        "cfg": "D",
        "norm_layer": None,
        "init_weights": True,
        "num_classes": 10
    },
    "mlp": {
        "in_features": 3 * 32 * 32,
        "cfg": [1024, 1024, 512, 512, 256, 256, 128, 64],
        "norm_layer": None,
        "num_classes": 10
    }
}

MODEL_MAP = {
    "resnet": _resnet,
    "resnet34": _resnet,
    "resnet50": _resnet,
    "mlp": _mlp,
    "vgg": _vgg
}
