"""
======================================================
From ResNets to Momentum ResNets
======================================================

This illustrates on two simple examples how to replace a ResNet with a MomentumNet 

"""  # noqa

# Authors: Michael Sander, Pierre Ablin
# License: MIT
import torch
from momentumnet import transform_to_momentumnet

#####################
# A torchvision model
#####################

from torchvision.models import resnet18

resnet = resnet18()
mresnet18 = transform_to_momentumnet(resnet, gamma=0.99, use_backprop=False)
x = torch.rand((64, 3, 7, 7), requires_grad=True)

##########################################
# It naturally extends the original ResNet
##########################################

x = torch.rand((64, 3, 7, 7))
resnet = resnet18()
lx = resnet(x)
mresnet = transform_to_momentumnet(resnet, gamma=0.0, use_backprop=False)
print(((resnet(x) - mresnet(x)) ** 2).sum())


######################################
# A Natural Language Transformer model
######################################

transformer = torch.nn.Transformer(num_encoder_layers=6, num_decoder_layers=6)
mtransformer = transform_to_momentumnet(
    transformer,
    residual_layers=["encoder.layers", "decoder.layers"],
    gamma=0.99,
    use_backprop=False,
    keep_first_layer=False,
)
