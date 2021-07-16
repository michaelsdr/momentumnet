"""
=========================================
Momentum ResNets on a digit learning task
=========================================

In this example we train a Momentum ResNet and a ResNet
on the sklearn digit dataset
"""

# Authors: Michael Sander, Pierre Ablin
# License: MIT

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torch.optim as optim

from momentumnet import MomentumNet
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(1)
np.random.seed(1)

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True
)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

##############################
# Architecture
##############################

hidden = 32
n_iters = 10
N = 1000
d = X.shape[-1]

functions = [
    nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))
    for _ in range(n_iters)
]

# Network
mresnet = MomentumNet(functions, gamma=0.5)

net = nn.Sequential(mresnet, nn.Linear(64, 10))
criterion = nn.CrossEntropyLoss()

n_epochs = 75
lr_list = np.ones(n_epochs) * 0.01

optimizer = optim.Adam(mresnet.parameters(), lr=lr_list[0])

###################################
# Training
###################################

test_error_mresnet = []
for i in range(n_epochs):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_list[i]
    optimizer.zero_grad()
    output = mresnet(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if i % 30 == 0:
        print("itr %s, loss = %.3f" % (i, loss.item()))
        print("- " * 20)
    _, pred = mresnet(X_test).max(1)
    test_error_mresnet.append(
        (1 - pred.eq(y_test).sum().item() / y_test.shape[0]) * 100
    )


#########################
# Same for a ResNet
#########################

functions = [
    nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))
    for _ in range(n_iters)
]

resnet = MomentumNet(functions, gamma=0.0)

net = nn.Sequential(resnet, nn.Linear(64, 10))
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(resnet.parameters(), lr=lr_list[0])

test_error_resnet = []
for i in range(n_epochs):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_list[i]
    optimizer.zero_grad()
    output = resnet(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if i % 30 == 0:
        print("itr %s, loss = %.3f" % (i, loss.item()))
        print("- " * 20)
    _, pred = resnet(X_test).max(1)
    test_error_resnet.append(
        (1 - pred.eq(y_test).sum().item() / y_test.shape[0]) * 100
    )

#####################################
# Plotting the learning curves
#####################################

plt.figure(figsize=(8, 4))
plt.semilogy(test_error_mresnet, label="Momentum ResNet", color="red", lw=2.5)
plt.semilogy(test_error_resnet, label="ResNet", color="darkblue", lw=2.5)
plt.xlabel("Epochs")
plt.ylabel("Test error")
plt.legend()
plt.show()
