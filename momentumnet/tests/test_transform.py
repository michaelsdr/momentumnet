# Authors: Michael Sander, Pierre Ablin
# License: MIT
import pytest

import torch
import torch.nn as nn

from momentumnet import transform
from numpy.testing import assert_raises

from torchvision.models import resnet18
from copy import copy

@pytest.mark.parametrize("use_backprop", [True, False])
def test_resnet_vision(use_backprop):
    x = torch.randn((2, 3, 10, 10), requires_grad=True)
    net = resnet18()
    net(x)
    momnet = transform(net, use_backprop=use_backprop)
    momnet(x)
    momnet.train()
    loss = momnet(x).sum()
    loss.backward()


@pytest.mark.parametrize("use_backprop", [True, False])
def test_transformer(use_backprop):
    x = torch.randn(2, 16, 512, requires_grad=True)
    tgt = torch.randn(2, 16, 512, requires_grad=True)
    net = nn.Transformer(num_encoder_layers=6, num_decoder_layers=6)
    momnet = transform(net, ['encoder.layers', 'decoder.layers'],
                       use_backprop=use_backprop, keep_first_layer=False)
    l = momnet(x, tgt).sum()
    l.backward()
    #print(x.grad, tgt.grad)


def test_outputs_transformer():
    x = torch.randn(2, 16, 512)
    tgt = torch.randn(2, 16, 512)
    net = nn.Transformer(num_encoder_layers=6, num_decoder_layers=6)
    mom_net = transform(net, ['encoder.layers'], gamma=0.99,
                       use_backprop=True, keep_first_layer=False)
    mom_no_backprop = transform(net, ['encoder.layers'], gamma=0.99,
                        use_backprop=False, keep_first_layer=False)
    mom_net.eval()
    mom_no_backprop.eval()
    assert torch.allclose(mom_net(x, tgt), mom_no_backprop(x, tgt), atol=1e-4, rtol=1e-3)
    x = torch.randn(2, 16, 512, requires_grad=True)
    mom_net_output = (mom_net(x, tgt) ** 2 + mom_net(x, tgt) ** 3).sum()
    mom_output = (mom_no_backprop(x, tgt) ** 2 + mom_no_backprop(x, tgt) ** 3).sum()
    #mom_net_output.backward()
    #mom_output.backward()
    params_mom_net = tuple(mom_net.parameters())
    params_mom = tuple(mom_no_backprop.parameters())
    grad_mom = torch.autograd.grad(mom_output, params_mom, allow_unused=True)
    grad_mom_net = torch.autograd.grad(mom_net_output, params_mom_net, allow_unused=True)

    for grad_1, grad_2 in zip(grad_mom_net, grad_mom):

        assert torch.allclose(grad_1, grad_2, atol=1e-3, rtol=1e-3)