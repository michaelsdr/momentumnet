"""
======================================================
From ResNets to Momentum ResNets 2)
======================================================

This illustrates on two simple examples how to replace an existing
ResNet with a MomentumNet.


Michael E. Sander, Pierre Ablin, Mathieu Blondel,
Gabriel Peyre. Momentum Residual Neural Networks.
Proceedings of the 38th International Conference 
on Machine Learning, PMLR 139:9276-9287

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

##########################################
# It naturally extends the original ResNet

x = torch.rand((64, 3, 7, 7))
resnet = resnet18()
lx = resnet(x)
mresnet = transform_to_momentumnet(resnet, gamma=0.0, use_backprop=False)
# gamma = 0 should gives the exacts same model
print(((resnet(x) - mresnet(x)) ** 2).sum())


######################################
# A Natural Language Transformer model
######################################

transformer = torch.nn.Transformer(num_encoder_layers=6, num_decoder_layers=6)
mtransformer = transform_to_momentumnet(
    transformer,  # Specify the sublayers to transform
    residual_layers=["encoder.layers", "decoder.layers"],
    gamma=0.99,
    use_backprop=False,
    keep_first_layer=False,
)
