import torch
from torch import nn, Tensor
from typing import Optional, Callable, Type, Union, List, Any
from model.modified_layer import ModifiedConv2d, ModifiedAdaptiveAvgPool2d, ModifiedLinear, ModifiedMaxPool2d


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return ModifiedConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return ModifiedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if norm_layer is not None:
            if norm_layer == nn.BatchNorm2d:
                self.bn1 = norm_layer(planes)
            elif norm_layer == nn.GroupNorm:
                self.bn1 = norm_layer(int(planes / 2), planes)
            elif norm_layer == nn.LayerNorm:
                self.bn1 = nn.GroupNorm(1, planes)
        else:
            self.bn1 = None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if norm_layer is not None:
            if norm_layer == nn.BatchNorm2d:
                self.bn2 = norm_layer(planes)
            elif norm_layer == nn.GroupNorm:
                self.bn2 = norm_layer(int(planes / 2), planes)
            elif norm_layer == nn.LayerNorm:
                self.bn2 = nn.GroupNorm(1, planes)
        else:
            self.bn2 = None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if norm_layer is not None:
            if norm_layer == nn.BatchNorm2d:
                self.bn1 = norm_layer(width)
            elif norm_layer == nn.GroupNorm:
                self.bn1 = norm_layer(int(width / 2), width)
            elif norm_layer == nn.LayerNorm:
                self.bn1 = nn.GroupNorm(1, width)
        else:
            self.bn1 = None
        # self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if norm_layer is not None:
            if norm_layer == nn.BatchNorm2d:
                self.bn2 = norm_layer(width)
            elif norm_layer == nn.GroupNorm:
                self.bn2 = norm_layer(int(width / 2), width)
            elif norm_layer == nn.LayerNorm:
                self.bn2 = nn.GroupNorm(1, width)
        else:
            self.bn2 = None
        # self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        if norm_layer is not None:
            if norm_layer == nn.BatchNorm2d:
                self.bn3 = norm_layer(planes * self.expansion)
            elif norm_layer == nn.GroupNorm:
                self.bn3 = norm_layer(int(planes * self.expansion / 2), planes * self.expansion)
            elif norm_layer == nn.LayerNorm:
                self.bn3 = nn.GroupNorm(1, planes * self.expansion)
        else:
            self.bn3 = None
        # self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.bn3 is not None:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_1d(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            signal=-1
    ) -> None:
        super().__init__()
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ModifiedConv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if norm_layer is None:
            self.bn1 = None
        else:
            if norm_layer == nn.BatchNorm2d:
                self.bn1 = norm_layer(self.inplanes)
            elif norm_layer == nn.GroupNorm:
                self.bn1 = norm_layer(int(self.inplanes / 2), self.inplanes)
            elif norm_layer == nn.LayerNorm:
                self.bn1 = nn.GroupNorm(1, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = ModifiedMaxPool2d(kernel_size=3, stride=2, padding=1)

        if self._norm_layer is None and signal == 1:
            must_norm = True
        else:
            must_norm = False
        self.layer1 = self._make_layer(block, 64, layers[0], must_norm=must_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       must_norm=must_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       must_norm=must_norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       must_norm=must_norm, final_block=True)
        self.avgpool = ModifiedAdaptiveAvgPool2d((1, 1))
        self.fc = ModifiedLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, ModifiedConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            must_norm: bool = False,
            final_block: bool = False,  # trick to remove norm layer at final block
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        if norm_layer is None and must_norm:
            norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_list = [conv1x1(self.inplanes, planes * block.expansion, stride)]
            if norm_layer is not None:
                if norm_layer == nn.BatchNorm2d:
                    downsample_list.append(norm_layer(planes * block.expansion))
                elif norm_layer == nn.GroupNorm:
                    downsample_list.append(norm_layer(int(planes * block.expansion / 2), planes * block.expansion))
                elif norm_layer == nn.LayerNorm:
                    downsample_list.append(nn.GroupNorm(1, planes * block.expansion))
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )
            downsample = nn.Sequential(*downsample_list)

        layers = [block(
            self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
        )]
        self.inplanes = planes * block.expansion

        # Trick to remove norm layer at the last block
        if final_block and must_norm:
            norm_layer = None
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet_1d(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        **kwargs: Any,
) -> ResNet_1d:
    model = ResNet_1d(block, layers, **kwargs)

    return model