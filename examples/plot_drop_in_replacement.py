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
import matplotlib.pyplot as plt
import numpy as np

#####################
# A torchvision model
#####################

from torchvision.models import resnet18

resnet = resnet18(pretrained=True)
mresnet101 = transform_to_momentumnet(resnet, gamma=0.99, use_backprop=False)
x = torch.rand((64, 3, 7, 7), requires_grad=True)
loss = mresnet101(x).sum()
loss.backward()

##########################################
# It naturally extends the original ResNet
##########################################

x = torch.rand((64, 3, 7, 7))
resnet = resnet18(pretrained=True)
lx = resnet(x)
outputs = []
ys = np.linspace(0, 0.2, 20)
for gamma in ys:
    mresnet101 = transform_to_momentumnet(resnet, gamma=gamma)
    outputs.append(((lx - mresnet101(x)) ** 2).sum())

plt.figure(figsize=(10, 5))
plt.plot(ys, outputs, linewidth=4, color="red")
y_ = plt.ylabel("Squared norm difference with the original output")
x_ = plt.xlabel("Gamma")
plt.show()


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
