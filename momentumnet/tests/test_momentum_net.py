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
        for _ in range(8)
    ]
    if init_speed:
        init_function = nn.Sequential(
            nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3)
        )
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


@pytest.mark.parametrize("init_speed", [True, False])
def test_outputs_memory_multiple_args(init_speed):
    class Custom(nn.Module):
        def __init__(self):
            super(Custom, self).__init__()
            self.layer1 = nn.Linear(3, 3)
            self.layer2 = nn.Linear(3, 3)

        def forward(self, x, mem):
            return self.layer1(x) + self.layer2(mem)

    functions = [Custom() for _ in range(5)]

    if init_speed:
        init_function = nn.Tanh()
    else:
        init_function = None
    mom_no_backprop = MomentumNet(
        functions,
        gamma=0.9,
        init_speed=init_speed,
        init_function=init_function,
        use_backprop=False,
    )
    mom_net = MomentumNet(
        functions,
        gamma=0.9,
        init_speed=init_speed,
        init_function=init_function,
        use_backprop=True,
    )
    x = torch.randn(10, 3, requires_grad=True)
    mem = torch.randn(10, 3, requires_grad=False)
    assert torch.allclose(
        mom_net(x, mem), mom_no_backprop(x, mem), atol=1e-4, rtol=1e-3
    )
    x = torch.randn(10, 3, requires_grad=True)
    mom_net_output = (mom_net(x, mem) ** 2 + mom_net(x, mem) ** 3).sum()
    mom_output = (
        mom_no_backprop(x, mem) ** 2 + mom_no_backprop(x, mem) ** 3
    ).sum()
    params_mom_net = tuple(mom_net.parameters())
    params_mom = tuple(mom_no_backprop.parameters())
    grad_mom_net = torch.autograd.grad(mom_net_output, (x,) + params_mom_net)
    grad_mom = torch.autograd.grad(mom_output, (x,) + params_mom)
    for grad_1, grad_2 in zip(grad_mom_net, grad_mom):
        assert torch.allclose(grad_1, grad_2, atol=1e-4, rtol=1e-3)


def test_two_inputs():
    class Custom(nn.Module):
        def __init__(self):
            super(Custom, self).__init__()
            self.layer1 = nn.Linear(3, 3)
            self.layer2 = nn.Linear(2, 3)

        def forward(self, x, mem):
            return self.layer1(x) + self.layer2(mem)

    functions = [Custom() for _ in range(3)]

    init_speed = False
    init_function = None
    mom_no_backprop = MomentumNet(
        functions,
        gamma=0.9,
        init_speed=init_speed,
        init_function=init_function,
        use_backprop=False,
    )
    mom_backprop = MomentumNet(
        functions,
        gamma=0.9,
        init_speed=init_speed,
        init_function=init_function,
        use_backprop=True,
    )
    x = torch.randn(1, 3, requires_grad=True)
    mem = torch.randn(1, 2, requires_grad=True)
    mom_output = (mom_no_backprop(x, mem) ** 2).sum()
    mom_output2 = (mom_backprop(x, mem) ** 2).sum()
    params_mom = tuple(mom_no_backprop.parameters())
    params_mom2 = tuple(mom_backprop.parameters())
    grad_mom = torch.autograd.grad(mom_output, (x, mem) + tuple(params_mom))
    grad_mom2 = torch.autograd.grad(mom_output2, (x, mem) + tuple(params_mom2))
    for g1, g2 in zip(grad_mom, grad_mom2):
        assert torch.allclose(g1, g2, atol=1e-4, rtol=1e-3)


def test_three_inputs():
    class Custom(nn.Module):
        def __init__(self):
            super(Custom, self).__init__()
            self.layer1 = nn.Linear(3, 3)
            self.layer2 = nn.Linear(2, 3)
            self.layer3 = nn.Linear(4, 3)

        def forward(self, x, mem, mem2):
            return self.layer1(x) + self.layer2(mem) + self.layer3(mem2)

    functions = [Custom() for _ in range(5)]

    init_speed = False
    init_function = None
    mom_no_backprop = MomentumNet(
        functions,
        gamma=0.9,
        init_speed=init_speed,
        init_function=init_function,
        use_backprop=False,
    )
    mom_backprop = MomentumNet(
        functions,
        gamma=0.9,
        init_speed=init_speed,
        init_function=init_function,
        use_backprop=True,
    )
    x_base = torch.randn(10, 3)
    mem_base = torch.randn(10, 2)
    mem2 = torch.randn(10, 4)
    gx_list = []
    gmem_list = []
    for mom in [mom_backprop, mom_no_backprop]:
        x = torch.clone(x_base)
        mem = torch.clone(mem_base)
        x.requires_grad = True
        mem.requires_grad = True
        output = mom(x, mem, mem2).sum()
        output.backward()
        gx_list.append(x.grad)
        gmem_list.append(mem.grad)
    assert torch.allclose(*gx_list)
    assert torch.allclose(*gmem_list)
