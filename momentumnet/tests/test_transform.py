# Authors: Michael Sander, Pierre Ablin
# License: MIT
import pytest

import torch
import torch.nn as nn

from momentumnet import transform
from numpy.testing import assert_raises

from torchvision.models import resnet18


@pytest.mark.parametrize("use_backprop", [True, False])
def test_resnet_vision(use_backprop):
    x = torch.randn(2, 3, 10, 10)
    net = resnet18()
    net(x)
    momnet = transform(net, use_backprop=use_backprop)
    momnet(x)
    momnet.train()
    loss = momnet(x).sum()
    loss.backward()


@pytest.mark.parametrize("use_backprop", [True, False])
def test_transformer(use_backprop):
    x = torch.randn(2, 16, 512)
    tgt = torch.randn(2, 16, 512)
    net = nn.Transformer()
    net(x, tgt)
    momnet = transform(net, ['encoder.layers', 'decoder.layers'],
                       use_backprop=use_backprop)
    momnet(x, tgt)
    momnet.train()
    loss = momnet(x, tgt).sum()
    loss.backward()
