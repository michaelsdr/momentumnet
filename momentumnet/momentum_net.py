# Authors: Michael Sander, Pierre Ablin
# License: MIT
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .exact_rep_pytorch import TorchExactRep


class MomentumNet(nn.Module):
    """
        A class used to define a Momentum ResNet

        ...

        Attributes
        ----------
        functions : list of nn, list of Sequential or Sequential
            a list of Sequential to define the transformation at each layer
        gamma : float
            the momentum term
        n_iters : int
            how many times to loop in functions in the forward pass (default 1)
        learn_gamma : bool
            whether to learn gamma
        init_speed : int
            if init_speed is not 0 then specify an init_function
        init_function : Sequential


        Methods
        -------
        forward(x, n_iters=None, ts=1)
            maps x to the output of the network
        """

    def __init__(
        self,
        functions,
        gamma,
        n_iters=1,
        learn_gamma=False,
        init_speed=0,
        init_function=None,
    ):
        super(MomentumNet, self).__init__()
        if gamma < 0 or gamma > 1:
            raise Exception("gamma has to be between 0 and 1")
        self.n_functions = len(functions)
        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.learn_gamma = learn_gamma
        if learn_gamma:
            self.gamma = Parameter(torch.tensor(gamma))
        else:
            self.gamma = gamma
        self.n_iters = n_iters
        self.init_speed = init_speed
        if init_function is not None:
            self.add_module("init", init_function)

    def forward(self, x, n_iters=None, ts=1):
        if n_iters is None:
            n_iters = self.n_iters
        if self.init_speed == 0:
            v = torch.zeros_like(x)
        else:
            v = self.init_function(x)
        gamma = self.gamma
        for i in range(n_iters):
            for function in self.functions:
                print(function(x).shape)
                v = gamma * v + function(x) * ts * (1 - gamma)
                x = x + v * ts
        return x

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    @property
    def init_function(self):
        return self._modules["init"]


class MomentumMemory(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, gamma, functions, init_function, *params):
        ctx.functions = functions
        ctx.gamma = gamma
        ctx.init_function = init_function
        v = TorchExactRep(v)
        with torch.no_grad():
            for function in functions:
                v *= gamma
                v += (1 - gamma) * function(x)
                x = x + v.val
        ctx.save_for_backward(x, v.intrep, v.aux.store)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        functions = ctx.functions
        gamma = ctx.gamma
        init_function = ctx.init_function
        x, v_intrep, v_store = ctx.saved_tensors
        v = TorchExactRep(0, from_representation=(v_intrep, v_store))
        grad_x = grad_output
        grad_v = torch.zeros_like(grad_x)
        grad_params = []
        with torch.set_grad_enabled(True):
            for function in functions[::-1]:
                x = x.detach().requires_grad_(False)
                x = x - v.val
                x = x.detach().requires_grad_(True)
                f_eval = function(x)
                grad_combi = grad_x + grad_v
                vjps = torch.autograd.grad(
                    f_eval, (x,) + tuple(function.parameters()), grad_combi
                )
                v += -(1 - gamma) * f_eval
                v /= gamma
                grad_params.append([(1 - gamma) * vjp for vjp in vjps[1:]])
                grad_x = grad_x + (1 - gamma) * vjps[0]
                grad_v = gamma * grad_combi
            if ctx.init_function is not None:
                x = x.detach().requires_grad_(True)
                f_eval = init_function(x)
                params = tuple(init_function.parameters())
                vjps = torch.autograd.grad(
                    f_eval, params, torch.zeros_like(grad_x)
                )
                grad_params.append([vjp for vjp in vjps])
            else:
                pass
        flat_params = []
        for param in grad_params[::-1]:
            flat_params += param
        return (grad_x, grad_v, None, None, None, *flat_params)


class Mom(nn.Module):
    """
            A class used to define a Momentum ResNet with the memory tricks

            ...

            Attributes
            ----------
            functions : list of nn, list of Sequential or Sequential
                a list of Sequential to define the transformation at each layer
            gamma : float
                the momentum term
            init_speed : int
                if init_speed is not 0 then specify an init_function
            init_function : Sequential


            Methods
            -------
            forward(x)
                maps x to the output of the network
            """

    def __init__(self, functions, gamma, init_speed=0, init_function=None):
        super(Mom, self).__init__()
        if gamma < 0 or gamma > 1:
            raise Exception("gamma has to be between 0 and 1")
        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.gamma = gamma
        self.init_speed = init_speed
        self.n_functions = len(functions)
        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.v = None
        if init_function is not None:
            self.add_module("init", init_function)

    def forward(self, x):
        if self.init_speed == 0:
            init_function = None
            v = torch.zeros_like(x)
            params = []
        else:
            v = self.init_function(x)
            params = list(self.init_function.parameters())
            init_function = self.init_function
        functions = self.functions
        for function in functions:
            params += list(function.parameters())
        output = MomentumMemory.apply(
            x, v, self.gamma, functions, init_function, *params
        )
        self.v = v
        return output

    def inverse(self, x):
        v = self.v
        gamma = self.gamma
        with torch.no_grad():
            for function in self.functions[::-1]:
                x = x - v.val
                v -= (1 - gamma) * function(x)
                v /= gamma
        return x

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    @property
    def init_function(self):
        return self._modules["init"]
