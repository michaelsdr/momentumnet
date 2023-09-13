# Authors: Michael Sander, Pierre Ablin
# License: MIT
import pytest
import torch
import torch.nn as nn

from momentumnet import transform_to_momentumnet, MomentumNet

from torchvision.models import resnet18, resnet101


torch.manual_seed(5)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(5)


@pytest.mark.parametrize(
    "use_backprop, depth", [(True, 18), (True, 101), (False, 18), (False, 101)]
)
def test_resnet_vision(use_backprop, depth):
    print(depth)
    x = torch.randn((2, 3, 10, 10), requires_grad=True)
    if depth == 18:
        net = resnet18()
    else:
        net = resnet101()
    momnet = transform_to_momentumnet(
        net, use_backprop=use_backprop, keep_first_layer=True
    )
    momnet.train()
    loss = momnet(x).sum()
    loss.backward()


@pytest.mark.parametrize("use_backprop", [True, False])
def test_transformer(use_backprop):
    x = torch.randn(2, 16, 512, requires_grad=True)
    tgt = torch.randn(2, 16, 512, requires_grad=True)
    net = nn.Transformer(num_encoder_layers=6, num_decoder_layers=6)
    momnet = transform_to_momentumnet(
        net,
        ["encoder.layers", "decoder.layers"],
        use_backprop=use_backprop,
        keep_first_layer=False,
    )
    loss = momnet(x, tgt).sum()
    loss.backward()


def test_outputs_transformer():
    x = torch.randn(2, 16, 512, requires_grad=True)
    tgt = torch.randn(2, 16, 512, requires_grad=True)
    net = nn.Transformer(num_encoder_layers=6, num_decoder_layers=6)
    mom_net = transform_to_momentumnet(
        net,
        ["encoder.layers", "decoder.layers"],
        gamma=0.99,
        use_backprop=True,
        keep_first_layer=False,
    )
    mom_no_backprop = transform_to_momentumnet(
        net,
        ["encoder.layers", "decoder.layers"],
        gamma=0.99,
        use_backprop=False,
        keep_first_layer=False,
    )
    mom_net.eval()
    mom_no_backprop.eval()
    assert torch.allclose(
        mom_net(x, tgt), mom_no_backprop(x, tgt), atol=1e-4, rtol=1e-4
    )
    x = torch.randn(2, 16, 512, requires_grad=True)
    tgt = torch.randn(2, 16, 512, requires_grad=True)
    mom_net_output = (mom_net(x, tgt) ** 2 + mom_net(x, tgt) ** 3).sum()
    mom_output = (
        mom_no_backprop(x, tgt) ** 2 + mom_no_backprop(x, tgt) ** 3
    ).sum()
    mom_net_output.backward()
    mom_output.backward()
    x = torch.randn(2, 16, 512, requires_grad=True)
    tgt = torch.randn(2, 16, 512, requires_grad=True)
    mom_net_output = (mom_net(x, tgt) ** 2 + mom_net(x, tgt) ** 3).sum()
    mom_output = (
        mom_no_backprop(x, tgt) ** 2 + mom_no_backprop(x, tgt) ** 3
    ).sum()
    params_mom_net = tuple(mom_net.parameters())
    params_mom = tuple(mom_no_backprop.parameters())
    grad_mom = torch.autograd.grad(mom_output, params_mom)
    grad_mom_net = torch.autograd.grad(mom_net_output, params_mom_net)

    for grad_1, grad_2 in zip(grad_mom_net, grad_mom):
        assert torch.allclose(grad_1, grad_2, atol=1e-4, rtol=1e-3)


def test_outputs_gradients_bn():
    x = torch.randn((20, 3, 10, 10), requires_grad=True)
    net = nn.Sequential(*[nn.BatchNorm2d(3) for _ in range(10)])
    mom_net = MomentumNet(net, gamma=0.9, use_backprop=True)
    mom_no_backprop = MomentumNet(net, gamma=0.9, use_backprop=False)
    mom_net.eval()
    mom_no_backprop.eval()
    assert torch.allclose(mom_net(x), mom_no_backprop(x), atol=1e-4, rtol=1e-4)
    x = torch.randn((20, 3, 10, 10), requires_grad=True)
    mom_net_output = (mom_net(x) ** 2 + mom_net(x) ** 3).mean()
    mom_output = (mom_no_backprop(x) ** 2 + mom_no_backprop(x) ** 3).mean()
    params_mom_net = tuple(mom_net.parameters())
    params_mom = tuple(mom_no_backprop.parameters())
    grad_mom = torch.autograd.grad(mom_output, params_mom)
    grad_mom_net = torch.autograd.grad(mom_net_output, params_mom_net)

    for grad_1, grad_2 in zip(grad_mom_net, grad_mom):
        assert torch.allclose(grad_1, grad_2, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize(
    "depth", [18, 101]
)
def test_outputs_gradients_resnets(depth):
    x = torch.randn((2, 3, 100, 100), requires_grad=True)
    if depth == 18:
        net = resnet18()
    else:
        net = resnet101()
    net.eval()
    x = torch.randn((2, 3, 100, 100), requires_grad=True)
    mom_net = transform_to_momentumnet(
            net, use_backprop=True, keep_first_layer=True
        )
    mom_no_backprop = transform_to_momentumnet(
            net, use_backprop=False, keep_first_layer=True
        )
    assert torch.allclose(mom_net(x), mom_no_backprop(x), atol=1e-4, rtol=1e-4)
    x = torch.randn((20, 3, 50, 50), requires_grad=True)
    mom_net_output = (mom_net(x) ** 2 + mom_net(x) ** 3).mean()
    mom_output = (mom_no_backprop(x) ** 2 + mom_no_backprop(x) ** 3).mean()
    params_mom_net = tuple(mom_net.parameters())
    params_mom = tuple(mom_no_backprop.parameters())
    grad_mom = torch.autograd.grad(mom_output, params_mom)
    grad_mom_net = torch.autograd.grad(mom_net_output, params_mom_net)

    for grad_1, grad_2 in zip(grad_mom_net, grad_mom):
        assert torch.allclose(grad_1, grad_2, atol=1e-3, rtol=1e-3)
