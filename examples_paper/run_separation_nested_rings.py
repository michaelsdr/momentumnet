# Authors: Michael Sander, Pierre Ablin
# License: MIT

"""
Separtion of nested rings using a MomentumNet.

- Use example/run_separation_nested_rings.py to train the networks.
  The results are saved in results/.
- Use example/plot_separation_nested_rings.py to plot the results.
  The figures are saved in figures/.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from momentumnet import MomentumNet
from momentumnet.toy_datasets import make_data
import os
import numpy as np

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
model = "mom_net"
if model == "mom_net":
    gamma = 0.99
else:
    gamma = 0

N = 1000
ts = 1


function = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 2))

# Network
mom_net = MomentumNet(
    [
        function,
    ],
    gamma=gamma,
    n_iters=n_iters,
    learn_gamma=False,
    init_speed=0,
)

criterion = nn.CrossEntropyLoss()

if model == "mom_net":
    n_epochs = 1500
    lr_list = np.ones(n_epochs) * 0.2

else:
    n_epochs = 5000
    lr_list = np.ones(n_epochs) * 0.01
    lr_list[n_epochs // 2 :] /= 2
    lr_list[3 * n_epochs // 4 :] /= 10

optimizer = optim.Adam(mom_net.parameters(), lr=lr_list[0])


###################################
# Training
###################################

if __name__ == "__main__":
    for i in range(n_epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_list[i]
        optimizer.zero_grad()
        x, y = make_data(2000)
        pred = mom_net(x, n_iters=n_iters, ts=ts)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("itr %s, loss = %.3f" % (i, loss.item()))
            print("- " * 20)

    # Save the output

    x_, y_ = make_data(500)

    for i in range(n_iters + 1):
        with torch.no_grad():
            pred_ = mom_net(x_, n_iters=i, ts=ts)
        torch.save(pred_, "results/%.3d_mom.pt" % i)

    torch.save(y_, "results/labels.pt")
