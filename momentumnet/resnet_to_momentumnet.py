# Authors: Michael Sander, Pierre Ablin
# License: MIT

"""
Illustration of the drop-in replacement aspect of Momentum ResNets.
"""

from copy import deepcopy
from momentumnet import MomentumNet

from torch import nn


def transform_to_momentumnet(
    model,
    sub_layers=["layer1", "layer2", "layer3", "layer4"],
    keep_first_layer=True,
    gamma=0.9,
    use_backprop=False,
    is_residual=False,
):
    """Return the MomentumNet counterpart of the model


    Parameters
    ----------
    model : a torch model
        The resnet one desires to turn into a Momentum ResNet.
    sub_layers : a list of strings
    (default ["layer1", "layer2", "layer3", "layer4"])
        The name of the submodules of the model one desires to make invertible.
    keep_first_layer : bool (default: True)
        Whether to leave to leave the first layer
        of each residual layer unchanged (useful if this first
        layer changes the dimension of the input).
    gamma : float (default: 0.9)
        The momentum term for the Momentum ResNet.
    use_backprop : bool (default: False)
        If True then the Momentum ResNet has a smaller memory footprint.
    is_residual : bool (default: False)
        If True then the forward rule is x + f(x)

    Returns
    -------
    mresnet : the MomentumNet ResNet counterpart of model

    Examples
    --------
    >>> import torch
    >>> from momentumnet import transform_to_momentumnet
    >>> from torchvision.models import resnet18
    >>> resnet = resnet18(pretrained=True)
    >>> layers = ["layer1", "layer2", "layer3", "layer4"]
    >>> mresnet = transform_to_momentumnet(resnet,
    ...                                    sub_layers=layers,
    ...                                    gamma=0.99, use_backprop=False)

    >>> import torch
    >>> from momentumnet import transform_to_momentumnet
    >>> transformer = torch.nn.Transformer(num_encoder_layers=6,
    ...                                    num_decoder_layers=6)
    >>> layers = ["encoder.layers", "decoder.layers"]
    >>> mtransformer = transform_to_momentumnet(transformer,
    ...                                         sub_layers=layers,
    ...                                         gamma=0.99,
    ...                                         use_backprop=False,
    ...                                         keep_first_layer=False)

    """
    mresnet = deepcopy(model)
    for residual_layer in sub_layers:
        splitted_key = residual_layer.split(".")
        parent_module = mresnet
        for i, key in enumerate(splitted_key):
            module = parent_module._modules[key]
            if i < len(splitted_key) - 1:
                parent_module = module
        if not keep_first_layer:
            momentumnet = nn.Sequential(
                MomentumNet(
                    module,
                    gamma=gamma,
                    use_backprop=use_backprop,
                    is_residual=is_residual,
                )
            )
        else:
            momentumnet = nn.Sequential(
                (
                    MomentumNet(
                        module[1:],
                        gamma=gamma,
                        use_backprop=use_backprop,
                        is_residual=is_residual,
                    )
                )
            )
            momentumnet = nn.Sequential(module[0], momentumnet)
        setattr(parent_module, key, momentumnet)
    return mresnet
