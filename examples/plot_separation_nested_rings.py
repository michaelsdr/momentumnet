# Authors: Michael Sander, Pierre Ablin
# License: MIT

"""
Separtion of nested rings using a MomentumNet.

- Use example/run_separation_nested_rings.py to train the networks.
  The results are saved in results/.
- Use example/plot_separation_nested_rings.py to plot the results.
  The figures are saved in figures/.
"""
import matplotlib.pyplot as plt
import torch
import imageio
import os

try:
    os.mkdir('figures')
except:
    pass


n_iters = 15
y_ = torch.load('results/labels.pt')


##############################################
# Plot the results
##############################################

for i in range(n_iters + 1):

    pred_ = torch.load('results/%.3d_mom.pt' % i)
    plt.figure(figsize=(10, 10))
    plt.scatter(pred_[:, 0], pred_[:, 1], c=y_ + 3, s=10)
    plt.axis('off')
    plt.savefig('figures/%.3d_mom.png' % i)
    plt.close('all')


images = []
for i in range(n_iters + 1):
    images.append(imageio.imread('figures/%.3d_mom.png' % i))

imageio.mimsave('figures/animation_separation.gif', images)