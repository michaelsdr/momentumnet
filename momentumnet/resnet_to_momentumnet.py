# Authors: Michael Sander, Pierre Ablin
# License: MIT
"""
Illustration of the drop-in replacement aspect of MomentumNets.

- Use example/from_resnet_to_momentumnet.py to transform ResNets
into MomentumNets, pretrained or not.

"""

from copy import deepcopy
from momentumnet.momentum_net import MomentumNet

from torch import nn
from momentumnet.models import (
    BasicBlock,
    MBasicBlock,
    Bottleneck,
    MBottleneck,
    MResNet,
)


# def transform(model, pretrained=False, gamma=0.9, use_backprop=False):
#     """Return the MomentumNet counterpart of the model


#     Parameters
#     ----------
#     model : a torchvision model
#         The resnet one desires to turn into a momentumnet
#     pretrained : bool (default: False)
#         Whether using a pretrained resnet and transfer its weights
#     gamma : float (default: 0.9)
#         The momentum term for the MomentumNet
#     use_backprop : bool (default: False)
#         If True then the MomentumNet has a smaller memory footprint

#     Return
#     ------
#     mresnet : the MomentumNet counterpart of model

#     """
#     resnet = model
#     layers = [
#         len(resnet.layer1),
#         len(resnet.layer2),
#         len(resnet.layer3),
#         len(resnet.layer4),
#     ]
#     num_classes = resnet.fc.out_features
#     try:
#         _ = resnet.layer1[0].conv3
#         mresnet = MResNet(
#             Bottleneck, MBottleneck, layers, num_classes, gamma=gamma, use_backprop=use_backprop
#         )
#     except AttributeError:
#         mresnet = MResNet(
#             BasicBlock, MBasicBlock, layers, num_classes, gamma=gamma, use_backprop=use_backprop
#         )
#     params1 = resnet.named_parameters()
#     params2 = mresnet.named_parameters()
#     if pretrained:
#         for (name1, param1), (name2, param2) in zip(params1, params2):
#             param2.data.copy_(param1)
#     return mresnet

def transform(model, residual_layers=['layer1', 'layer2', 'layer3', 'layer4'],
              keep_first_layer=False, gamma=0.9, use_backprop=False):
    """Return the MomentumNet counterpart of the model


    Parameters
    ----------
    model : a torchvision model
        The resnet one desires to turn into a momentumnet
    gamma : float (default: 0.9)
        The momentum term for the MomentumNet
    use_backprop : bool (default: False)
        If True then the MomentumNet has a smaller memory footprint

    Return
    ------
    mresnet : the MomentumNet counterpart of model

    """
    momnet = deepcopy(model)
    for residual_layer in residual_layers:
        splitted_key = residual_layer.split('.')
        parent_module = momnet
        for i, key in enumerate(splitted_key):
            module = parent_module._modules[key]
            if i < len(splitted_key) - 1:
                parent_module = module
        if keep_first_layer:
            module_minus_identity = [transform_minus_id(layer) for layer in module]
            momentumnet = MomentumNet(module_minus_identity, gamma=gamma, use_backprop=use_backprop)
        else:
            module_minus_identity = [transform_minus_id(layer) for layer in module[1:]]
            momentumnet = MomentumNet(module_minus_identity, gamma=gamma, use_backprop=use_backprop)
            momentumnet = nn.Sequential(module[0], momentumnet)   
        setattr(parent_module, key, momentumnet)         
    return momnet


def transform_minus_id(layer):
    new_layer = deepcopy(layer)
    def forward(*input):
        return layer(*input) - input[0]
    new_layer.forward = forward
    return new_layer

if __name__ == '__main__':
    import torch
    from torchvision.models import resnet18
    net = resnet18()
    x = torch.randn(2, 3, 10, 10)
    moment = transform(net, gamma=0)
    print(net(x) - moment(x))