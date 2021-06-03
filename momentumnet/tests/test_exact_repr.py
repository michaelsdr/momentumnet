# Authors: Michael Sander, Pierre Ablin
# License: MIT

import torch

from momentumnet.exact_rep_pytorch import TorchExactRep


def test_val_torch():
    torch.manual_seed(0)
    x = torch.randn(10)
    x_rep = TorchExactRep(x)
    x_init_val = x_rep.val.clone()
    assert ((x - x_rep.val).abs() < 1e-7).byte().all()
    x_rep *= 3.0
    x *= 3.0
    assert ((x - x_rep.val).abs() < 1e-3).byte().all()
    x_rep += 2
    x += 2
    assert ((x - x_rep.val).abs() < 1e-3).byte().all()
    x_rep -= 2
    x -= 2
    assert ((x - x_rep.val).abs() < 1e-3).byte().all()
    x_rep /= 3
    x /= 3
    assert ((x - x_rep.val).abs() < 1e-3).byte().all()
    assert (x_rep.val == x_init_val).byte().all()
