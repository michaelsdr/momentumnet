# Authors: Michael Sander, Pierre Ablin
# License: MIT

"""
Illustration of the drop-in replacement aspect of Momentum ResNets.
"""

from copy import deepcopy
from momentumnet import MomentumNetTransform

from torch import nn


def transform(
    model,
    residual_layers=["layer1", "layer2", "layer3", "layer4"],
    keep_first_layer=True,
    gamma=0.9,
    use_backprop=False,
):
    """Return the MomentumNet counterpart of the model


    Parameters
    ----------
    model : a torch model
        The resnet one desires to turn into a Momentum ResNet.
    residual_layers : a list of strings
        The name of the submodules of the model one desires to make invertible.
    gamma : float (default: 0.9)
        The momentum term for the Momentum ResNet.
    use_backprop : bool (default: False)
        If True then the Momentum ResNet has a smaller memory footprint.

    Return
    ------
    mresnet : the MomentumNet ResNet counterpart of model

    """
    momnet = deepcopy(model)
    for residual_layer in residual_layers:
        splitted_key = residual_layer.split(".")
        parent_module = momnet
        for i, key in enumerate(splitted_key):
            module = parent_module._modules[key]
            if i < len(splitted_key) - 1:
                parent_module = module
        if not keep_first_layer:
            momentumnet = nn.Sequential(
                MomentumNetTransform(
                    module, gamma=gamma, use_backprop=use_backprop
                )
            )
        else:
            momentumnet = nn.Sequential(
                (
                    MomentumNetTransform(
                        module[1:], gamma=gamma, use_backprop=use_backprop
                    )
                )
            )
            momentumnet = nn.Sequential(module[0], momentumnet)
        setattr(parent_module, key, momentumnet)
    return momnet


def transform_minus_id(layer):
    new_layer = deepcopy(layer)

    def forward(*input):
        return layer(*input) - input[0]

    new_layer.forward = forward
    return new_layer
