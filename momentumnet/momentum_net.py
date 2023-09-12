# Authors: Michael Sander, Pierre Ablin
# License: MIT
import torch
import torch.nn as nn
from .exact_rep_pytorch import TorchExactRep
from memcnn import InvertibleModuleWrapper


class MomentumNetWithBackprop(nn.Module):
    """
    A class used to define a Momentum ResNet

    Parameters
    ----------
    functions : list of modules, list of Sequential or Sequential
        a list of Sequential to define the transformation at each layer
    gamma : float
        the momentum term
    init_speed : bool (default: False)
        if init_speed is True then specify an init_function for the velocity v
    init_function : Sequential (default: None)
        to initialize the velocity to init_function(x) before the forward pass
    is_residual : Bool (default: True)
        if True then the blocks are residual

    Methods
    -------
    forward(x)
        maps x to the output of the network
    """

    def __init__(
        self,
        functions,
        gamma,
        init_speed=False,
        init_function=None,
        is_residual=True,
    ):
        super(MomentumNetWithBackprop, self).__init__()
        if gamma < 0 or gamma > 1:
            raise Exception("gamma has to be between 0 and 1")
        self.n_functions = len(functions)
        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.gamma = gamma
        self.init_speed = init_speed
        self.is_residual = is_residual
        if init_function is not None:
            self.add_module("init", init_function)

    def forward(self, x, *function_args):
        n_iters = len(self.functions)
        if not self.init_speed:
            v = torch.zeros_like(x)
        else:
            v = self.init_function(x)
        gamma = self.gamma
        for i in range(n_iters):
            f = self.functions[i](x, *function_args)
            if not self.is_residual:
                f = f - x
            v = gamma * v + f * (1 - gamma)
            x = x + v
        return x, v

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    @property
    def init_function(self):
        return self._modules["init"]


class MomentumFwd(nn.Module):
    def __init__(self, function, gamma, is_residual):
        super().__init__()
        self.function = function
        self.gamma = gamma
        self.is_residual = is_residual

    def forward(self, x, v, fun_args):
        v *= self.gamma
        f = self.function(x, *fun_args)
        if not self.is_residual:
            f = f - x
        v += (1 - self.gamma) * f
        x = x + v.val
        return x, v 

    def inverse(self, x, v, fun_args):
        x = x - v.val
        if self.is_residual:
            f_eval = self.function(x, *fun_args)
        else:
            f_eval = self.function(x, *fun_args) - x
        v += -(1 - self.gamma) * f_eval
        v /= self.gamma
        return x, v


class MomentumNetNoBackprop(nn.Module):
    """
    A class used to define a Momentum ResNet with the memory tricks

    Parameters
    ----------
    functions : list of modules, list of Sequential or Sequential
        a list of Sequential to define the transformation at each layer
    gamma : float
        the momentum term
    init_speed : bool (default: False)
        if init_speed is True then specify an init_function for the velocity v
    init_function : Sequential (default: None)
        to initialize the velocity to init_function(x) before the forward pass
    is_residual : Bool (default: True)
        if True then the blocks are residual


    Methods
    -------
    forward(x)
        maps x to the output of the network
    """

    def __init__(
        self,
        functions,
        gamma,
        init_speed=False,
        init_function=None,
        is_residual=True,
    ):
        super(MomentumNetNoBackprop, self).__init__()
        if gamma < 0 or gamma > 1:
            raise Exception("gamma has to be between 0 and 1")
        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.gamma = gamma
        self.init_speed = init_speed
        self.n_functions = len(functions)
        self.v = None
        self.is_residual = is_residual
        if init_function is not None:
            self.add_module("init", init_function)
        self.nets = [
                InvertibleModuleWrapper(
                    MomentumFwd(functions[i], gamma, is_residual),
                    num_bwd_passes=1,
                    disable=False,
                )
                for i in range(len(functions))
            ]

    def forward(self, x, *function_args):
        params = []
        functions = self.functions
        if not self.init_speed:
            v = torch.zeros_like(x)
        else:
            v = self.init_function(x)
            params += find_parameters(self.init_function)
        for function in functions:
            params += find_parameters(function)
        for net in self.nets: 
            x, v = net(x.clone(), v, function_args)
        self.v = v
        return x, v

    def inverse(self, x, v, *function_args):
        for net in self.nets[::-1]:
            x, v = net.inverse(x, v, function_args)
        return x
    

    @property
    def init_function(self):
        return self._modules["init"]

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    def __getitem__(self, idx):
        return self.functions[idx]


class MomentumNet(nn.Module):
    """
    Create a Momentum Residual Network.

    It iterates

    .. code:: python

        v_{t + 1} = gamma * v_t + (1 - gamma) * f_t(x_t)
        x_{t + 1} = x_t + v_{t + 1}

    where the f_t are stored in `functions`.
    These forward equations can be reversed in closed-form,
    enabling learning without backpropagation. This process
    trades memory for computations.

    Parameters
    ----------
    functions : list of modules, list of Sequential or Sequential
        a list of Sequential to define the transformation at each layer.
        Each function in the list can take additional inputs. 'x' is assumed
        to be the first input.
    gamma : float
        the momentum term
    init_speed : bool (default: False)
        if init_speed is True then specify an init_function for the velocity v
    init_function : Sequential (default: None)
        to initialize the velocity to init_function(x) before the forward pass
    use_backprop : bool (default: True)
        if True then standard backpropagation is used,
        if False activations are not
        saved during the forward pass allowing memory savings
    is_residual : Bool (default: True)
        if True then the update rule is
        v_{t + 1} = (1 - gamma) * v_t + gamma * f_t(x_t)
        if False then the update rule is
        v_{t + 1} = (1 - gamma) * v_t + gamma * (f_t(x_t) - x_t)

    Methods
    -------
    forward(x, *args, **kwargs)
        maps x to the output of the network

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from momentumnet import MomentumNet
    >>> hidden = 8
    >>> d = 50
    >>> function = nn.Sequential(nn.Linear(d, hidden),
    ...                           nn.Tanh(), nn.Linear(hidden, d))
    >>> mresnet = MomentumNet([function,] * 10, gamma=0.99)
    >>> x = torch.randn(10, d)
    >>> mresnet(x).shape
    torch.Size([10, 50])

    Notes
    -----
    Implementation based on
    *Michael E. Sander, Pierre Ablin, Mathieu Blondel,
    Gabriel Peyre. Momentum Residual Neural Networks.
    Proceedings of the 38th International Conference
    on Machine Learning, PMLR 139:9276-9287*

    """

    def __init__(
        self,
        functions,
        gamma,
        init_speed=False,
        init_function=None,
        use_backprop=True,
        is_residual=True,
    ):

        super(MomentumNet, self).__init__()
        if use_backprop:
            self.network = MomentumNetWithBackprop(
                functions,
                gamma,
                init_speed,
                init_function,
                is_residual,
            )
        else:
            self.network = MomentumNetNoBackprop(
                functions,
                gamma,
                init_speed,
                init_function,
                is_residual,
            )
        self.use_backprop = use_backprop
        self.gamma = gamma
        self.init_speed = init_speed

    def forward(self, x, *args, **kwargs):
        x, v = self.network.forward(x, *args)
        return x

    @property
    def init_function(self):
        return self.network.init_function

    @property
    def functions(self):
        return self.network.functions


def find_parameters(module):
    """Find the parameters of the module
    code from: https://github.com/rtqichen/torchdiffeq

    Parameters
    ----------
    module : a torch module

    Returns
    -------
    list : the list of parameters of the module

    """
    assert isinstance(module, nn.Module)

    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if torch.is_tensor(v) and v.requires_grad
            ]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())
