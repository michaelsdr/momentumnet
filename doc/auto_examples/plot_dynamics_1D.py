"""
==================================
Plotting the dynamics in 1D
==================================

This example compares the dynamics of a ResNet and a Momentum ResNet

"""  # noqa

# Authors: Michael Sander, Pierre Ablin
# License: MIT
import os
import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from momentumnet import MomentumNet
from momentumnet.toy_datasets import make_data_1D

if not os.path.isdir("figures"):
    os.mkdir("figures")
###########################################
# Fix random seed for reproducible figures
###########################################

torch.manual_seed(1)

##############################
# Parameters of the simulation
##############################

hidden = 16
n_iters = 15
gamma = 0.99
d = 1
function = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))
function_res = copy.deepcopy(function)

mom_net = MomentumNet(
    [
        function,
    ]
    * n_iters,
    gamma=gamma,
    init_speed=0,
)
res_net = MomentumNet(
    [
        function_res,
    ]
    * n_iters,
    gamma=0.0,
    init_speed=0,
)


def h(x):
    return -(x ** 3)


def Loss(pred, x):
    return ((pred - h(x)) ** 2).mean()


optimizer = optim.SGD(mom_net.parameters(), lr=0.01)

# Training

print("MomentumNet ->")
for i in range(301):
    optimizer.zero_grad()
    x = make_data_1D(200)
    pred = mom_net(x)
    loss = Loss(pred, x)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print("- " * 20)
        print("itr %s, loss = %.3f" % (i, loss.item()))

print("- " * 40)

optimizer = optim.SGD(res_net.parameters(), lr=0.01)


print("ResNet -->")
for i in range(2001):
    optimizer.zero_grad()
    x = make_data_1D(200)
    pred = res_net(x)
    loss = Loss(pred, x)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print("- " * 20)
        print("itr %s, loss = %.3f" % (i, loss.item()))


# Plot the output


n_plot = 8

num_plots = n_plot

plt.figure(figsize=(3, 4))
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(
    plt.cycler("color", plt.cm.jet(np.linspace(0.8, 0.95, num_plots)))
)

x_ = make_data_1D(n_plot)
x = np.linspace(-1, 1, n_plot)
x_ = torch.tensor(x).view(-1, d).float()
x_axis = np.arange(0, n_iters + 1)

preds = np.zeros((n_iters + 1, n_plot))

preds[0] = x_[:, 0]

for i in range(1, n_iters + 1):
    mom_net = MomentumNet(
        [
            function,
        ]
        * i,
        gamma=gamma,
        init_speed=0,
    )
    with torch.no_grad():

        pred_ = mom_net(x_)
        preds[i] = pred_[:, 0]

plt.plot(preds, x_axis, "-x", lw=2.5)
plt.xticks([], [])
plt.yticks([], [])
plt.ylabel("Depth")
plt.xlabel("Input")
plt.savefig("figures/mom_net_dynamics_1D.pdf")

num_plots = n_plot

plt.figure(figsize=(3, 4))
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(
    plt.cycler("color", plt.cm.jet(np.linspace(0.0, 0.1, num_plots)))
)

x_axis = np.arange(0, n_iters + 1)

preds_res = np.zeros((n_iters + 1, n_plot))

preds_res[0] = x_[:, 0]
for i in range(1, n_iters + 1):
    res_net = MomentumNet(
        [
            function_res,
        ]
        * i,
        gamma=0.0,
        init_speed=0,
    )

    with torch.no_grad():
        pred_ = res_net(x_)
        preds_res[i] = pred_[:, 0]

plt.plot(preds_res, x_axis, "-x", lw=2.5)
plt.xticks([], [])
plt.yticks([], [])
plt.ylabel("Depth")
plt.xlabel("Input")
plt.savefig("figures/res_net_dynamics_1D.pdf")
