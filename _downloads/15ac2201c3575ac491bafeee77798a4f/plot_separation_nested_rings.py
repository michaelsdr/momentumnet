"""
===================================================
Separation of nested rings using a Momentum ResNet.
===================================================

This example shows how a Momentum ResNet separates two nested rings


Michael E. Sander, Pierre Ablin, Mathieu Blondel,
Gabriel Peyre. Momentum Residual Neural Networks.
Proceedings of the 38th International Conference
on Machine Learning, PMLR 139:9276-9287


"""

# Authors: Michael Sander, Pierre Ablin
# License: MIT

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from momentumnet import MomentumNet
from momentumnet.toy_datasets import make_data

torch.manual_seed(1)

##############################
# Parameters of the simulation
##############################

hidden = 16
n_iters = 10
N = 1000

function = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 2))

# Network
mresnet = MomentumNet(
    [
        function,
    ]
    * n_iters,
    gamma=0.99,
)

criterion = nn.CrossEntropyLoss()

n_epochs = 30
lr_list = np.ones(n_epochs) * 0.5

optimizer = optim.Adam(mresnet.parameters(), lr=lr_list[0])


###################################
# Training
###################################


for i in range(n_epochs):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_list[i]
    optimizer.zero_grad()
    x, y = make_data(
        2000,
    )
    pred = mresnet(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

##############################################
# Plot the results
##############################################

x_, y_ = make_data(500)

fig, axis = plt.subplots(1, n_iters + 1, figsize=(n_iters + 1, 1))

for i in range(n_iters + 1):
    mom_net = MomentumNet(
        [
            function,
        ]
        * i,
        gamma=0.99,
        init_speed=0,
    )
    with torch.no_grad():
        pred_ = mom_net(x_)
        axis[i].scatter(pred_[:, 0], pred_[:, 1], c=y_ + 3, s=1)
        axis[i].axis("off")
plt.show()
