# Authors: Michael Sander, Pierre Ablin
# License: MIT

import matplotlib.pyplot as plt
import torch
import imageio
import os

"""
==================================
Separtion of nested rings using a Momentum ResNet.
==================================

This example shows how a Momentum ResNet separates nested rings 

"""  # noqa

"""
Separtion of nested rings using a MomentumNet.
"""

if not os.path.isdir("figures"):
    os.mkdir("../examples/figures")


n_iters = 15
y_ = torch.load("results/labels.pt")


##############################################
# Plot the results
##############################################

for i in range(n_iters + 1):

    pred_ = torch.load("results/%.3d_mom.pt" % i)
    plt.figure(figsize=(10, 10))
    plt.scatter(pred_[:, 0], pred_[:, 1], c=y_ + 3, s=10)
    plt.axis("off")
    plt.savefig("figures/%.3d_mom.png" % i)
    plt.close("all")


images = []
for i in range(n_iters + 1):
    images.append(imageio.imread("figures/%.3d_mom.png" % i))

imageio.mimsave("figures/animation_separation.gif", images)
