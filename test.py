import torch
from torch import nn
from momentumnet import MomentumNet
import argparse


parser = argparse.ArgumentParser(description="test")
parser.add_argument("-u", default=0, type=int)
args = parser.parse_args()

use_backprop = args.u
p = 3
one_layer = nn.Sequential(nn.Linear(p, 10),
                          nn.Tanh(),
                          nn.Linear(10, p))

print(use_backprop)
@profile
def func():
    n_layers = 10
    momnet = MomentumNet([one_layer, ] * n_layers, gamma=0.9, use_backprop=use_backprop)

    x = torch.randn(300, p)

    y = momnet(x)
    op = y.sum()
    op.backward()


func()