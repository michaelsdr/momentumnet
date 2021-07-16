"""
======================================================
From ResNets to Momentum ResNets 3)
======================================================

This illustrates on a more complex example how to replace an existing
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

########################################
# We will use a Vision Transformer model
########################################

########################################################################
# From https://arxiv.org/abs/2010.11929
# Code adapted from https://github.com/lucidrains/vit-pytorch

from vit_pytorch import ViT

v = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
)

################################################
# We first rename transformer layer from v to be
# consistent with our forward rule

v.transformer = v.transformer.layers

###################################################
# We simply modify the transformer module to have a
# Sequential form

v_modules = []
for i, _ in enumerate(v.transformer):
    for layer in v.transformer[i]:
        v_modules.append(layer)

v.transformer = torch.nn.Sequential(*v_modules)

#################################################
# Now we can transform it to its momentum version

mv = transform_to_momentumnet(
    v,
    ["transformer"],
    gamma=0.9,
    keep_first_layer=False,
    use_backprop=False,
    is_residual=True,
)
