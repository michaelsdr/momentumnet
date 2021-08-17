"""
======================================================
From ResNets to Momentum ResNets 1)
======================================================

This is a tutorial to use the transform_to_momentumnet
method:

Michael E. Sander, Pierre Ablin, Mathieu Blondel,
Gabriel Peyre. Momentum Residual Neural Networks.
Proceedings of the 38th International Conference 
on Machine Learning, PMLR 139:9276-9287


"""  # noqa

# Authors: Michael Sander, Pierre Ablin
# License: MIT
from torch import nn
from momentumnet import transform_to_momentumnet

####################################
# Let us define a toy Neural Network
####################################


class ResBlock(nn.Module):
    def __init__(self, functions):
        super(ResBlock, self).__init__()
        self.functions = functions

    def forward(self, x):

        for f in self.functions:
            x = x + f(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.res_layer1 = ResBlock(
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 2)
                    )
                    for _ in range(3)
                ]
            )
        )
        self.l1 = nn.Linear(2, 4)
        self.layer2 = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(4, 100), nn.ReLU(), nn.Linear(100, 4))
                for _ in range(4)
            ]
        )
        self.l2 = nn.Linear(4, 8)

        self.fc = nn.Linear(8, 10)

    def forward(self, x):

        out = self.res_layer1(x)  # Residual
        out = self.l1(out)
        out = self.layer2(out)  # Not Residual but same dimensions
        out = self.l2(out)
        out = self.fc(out)

        return out


net = Net()

###################################################
# We want to transform it into its Momentum version
###################################################

###############################################################################
# The first layer 'res_layer1' preserves dimension and is residual.
# It can be accessed through net.res_layer_1.functions so we will specify
# this attribute as the "sub_layers" parameter.
# One can transform this residual block into a momentum one as follow:

mnet1 = transform_to_momentumnet(
    net,
    ["res_layer1.functions"],  # attributes of the sublayers in net
    gamma=0.9,
    use_backprop=False,
    is_residual=True,
    keep_first_layer=False,
)

###############################################################################
# Note that layer2 is not residual but also preserves dimensions.
# It can be accessed through net.layer_2 so we will specify
# this attribute as the "sub_layers" parameter.
# One can transform it in the same way setting is_residual to False.

mnet = transform_to_momentumnet(
    mnet1,
    ["layer2"],
    gamma=0.9,
    use_backprop=False,
    is_residual=False,
    keep_first_layer=False,
)

###############################################################################
# net, mnet1, and mnet have the same parameters.
