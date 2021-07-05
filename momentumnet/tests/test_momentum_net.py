# Authors: Michael Sander, Pierre Ablin
# License: MIT
import pytest


import torch
import torch.nn as nn

from momentumnet import MomentumNet
from numpy.testing import assert_raises


torch.manual_seed(1)


@pytest.mark.parametrize("use_backprop", [True, False])
def test_dimension_layers(use_backprop):
    function = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 3))
    mom_net = MomentumNet(
        [
            function,
        ],
        gamma=0.9,
        use_backprop=use_backprop,
    )
    x = torch.rand(3)
    assert mom_net(x).shape == x.shape
    function = nn.Sequential(nn.Linear(3, 4), nn.Linear(3, 3), nn.Linear(3, 3))
    mom_net = MomentumNet(
        [
            function,
        ],
        gamma=0.9,
        use_backprop=use_backprop,
    )
    assert_raises(RuntimeError, mom_net, x)


@pytest.mark.parametrize("init_speed", [True, False])
def test_outputs_memory(init_speed):
    functions = [
        nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3))
        for _ in range(5)
    ]
    if init_speed:
        init_function = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3))
    else:
        init_function = None
    mom_no_backprop = MomentumNet(
        functions,
        gamma=0.99,
        init_speed=init_speed,
        init_function=init_function,
        use_backprop=False,
    )
    mom_net = MomentumNet(
        functions,
        gamma=0.99,
        init_speed=init_speed,
        init_function=init_function,
        use_backprop=True,
    )
    x = torch.rand(3, requires_grad=True)
    assert torch.allclose(mom_net(x), mom_no_backprop(x), atol=1e-5, rtol=1e-3)
    x = torch.rand(3, requires_grad=True)
    mom_net_output = (mom_net(x) ** 2 + mom_net(x) ** 3).sum()
    mom_output = (mom_no_backprop(x) ** 2 + mom_no_backprop(x) ** 3).sum()
    params_mom_net = tuple(mom_net.parameters())
    params_mom = tuple(mom_no_backprop.parameters())
    grad_mom_net = torch.autograd.grad(mom_net_output, (x,) + params_mom_net)
    grad_mom = torch.autograd.grad(mom_output, (x,) + params_mom)
    for grad_1, grad_2 in zip(grad_mom_net, grad_mom):
        assert torch.allclose(grad_1, grad_2, atol=1e-5, rtol=1e-3)
