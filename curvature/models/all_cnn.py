# All-CNN-C architecture for *CIFAR-100* dataset.
# Added by Xingchen Wan on 2 Dec. Modified from this github repository:
# https://github.com/fsschneider/DeepOBS/blob/develop/deepobs/pytorch/testproblems/testproblems_modules.py


import torch.nn as nn
from math import ceil

__all__ = ['AllCNN_CIFAR100']


def _determine_padding_from_tf_same(
    input_dimensions, kernel_dimensions, stride_dimensions
):
    """Implements tf's padding 'same' for kernel processes like convolution or pooling.
    Args:
        input_dimensions (int or tuple): dimension of the input image
        kernel_dimensions (int or tuple): dimensions of the convolution kernel
        stride_dimensions (int or tuple): the stride of the convolution
     Returns: A padding 4-tuple for padding layer creation that mimics tf's padding 'same'.
     """

    # get dimensions
    in_height, in_width = input_dimensions

    if isinstance(kernel_dimensions, int):
        kernel_height = kernel_dimensions
        kernel_width = kernel_dimensions
    else:
        kernel_height, kernel_width = kernel_dimensions

    if isinstance(stride_dimensions, int):
        stride_height = stride_dimensions
        stride_width = stride_dimensions
    else:
        stride_height, stride_width = stride_dimensions

    # determine the output size that is to achive by the padding
    out_height = ceil(in_height / stride_height)
    out_width = ceil(in_width / stride_width)

    # determine the pad size along each dimension
    pad_along_height = max(
        (out_height - 1) * stride_height + kernel_height - in_height, 0
    )
    pad_along_width = max(
        (out_width - 1) * stride_width + kernel_width - in_width, 0
    )

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


def hook_factory_tf_padding_same(kernel_size, stride):
    """Generates the torch pre forward hook that needs to be registered on
    the padding layer to mimic tf's padding 'same'"""

    def hook(module, input):
        """The hook overwrites the padding attribute of the padding layer."""
        image_dimensions = input[0].size()[-2:]
        module.padding = _determine_padding_from_tf_same(
            image_dimensions, kernel_size, stride
        )

    return hook


def tfconv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    tf_padding_type=None,
):
    modules = []
    if tf_padding_type == "same":
        padding = nn.ZeroPad2d(0)
        hook = hook_factory_tf_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    modules.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    )
    return nn.Sequential(*modules)


def mean_allcnnc():
    """The all convolution layer implementation of torch.mean().
    Use the backpack version of the flatten layer - edited by Xingchen Wan"""
    from backpack.core.layers import Flatten
    return nn.Sequential(nn.AvgPool2d(kernel_size=(6, 6)), Flatten())


class AllCNN_C(nn.Sequential):
    def __init__(self, num_classes=100):
        super(AllCNN_C, self).__init__()

        self.add_module("dropout1", nn.Dropout(p=0.2))

        self.add_module(
            "conv1", tfconv2d(in_channels=3,
                              out_channels=96,
                              kernel_size=3,
                              tf_padding_type="same",),
        )
        self.add_module("relu1", nn.ReLU())
        self.add_module(
            "conv2",
            tfconv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu2", nn.ReLU())
        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type="same",
            ),
        )
        self.add_module("relu3", nn.ReLU())

        self.add_module("dropout2", nn.Dropout(p=0.5))

        self.add_module(
            "conv4",
            tfconv2d(
                in_channels=96,
                out_channels=192,
                kernel_size=3,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu4", nn.ReLU())
        self.add_module(
            "conv5",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu5", nn.ReLU())
        self.add_module(
            "conv6",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type="same",
            ),
        )
        self.add_module("relu6", nn.ReLU())

        self.add_module("dropout3", nn.Dropout(p=0.5))

        self.add_module(
            "conv7", tfconv2d(in_channels=192, out_channels=192, kernel_size=3)
        )
        self.add_module("relu7", nn.ReLU())
        self.add_module(
            "conv8",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=1,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu8", nn.ReLU())
        self.add_module(
            "conv9",
            tfconv2d(
                in_channels=192,
                out_channels=num_classes,
                kernel_size=1,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu9", nn.ReLU())

        self.add_module("mean", mean_allcnnc())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.1)
                nn.init.xavier_normal_(module.weight)


import torchvision.transforms as transforms


class AllCNN_CIFAR100:
    base = AllCNN_C
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])