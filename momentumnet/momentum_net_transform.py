# Authors: Michael Sander, Pierre Ablin
# License: MIT
import torch
import torch.nn as nn
from .exact_rep_pytorch import TorchExactRep


class MomentumNetTransformWithBackprop(nn.Module):
    """
    A class used to define a Momentum ResNet

    ...

    Attributes
    ----------
    functions : list of nn, list of Sequential or Sequential
        a list of Sequential to define the transformation at each layer
    gamma : float
        the momentum term
    init_speed : int
        if init_speed is True then specify an init_function
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
        init_speed=False,
        init_function=None,
    ):
        super(MomentumNetTransformWithBackprop, self).__init__()
        if gamma < 0 or gamma > 1:
            raise Exception("gamma has to be between 0 and 1")
        self.n_functions = len(functions)
        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.gamma = gamma
        self.init_speed = init_speed
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
            v = gamma * v + (self.functions[i](x, *function_args) - x) * (1 - gamma)
            x = x + v
        return x

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    @property
    def init_function(self):
        return self._modules["init"]


class MomentumTransformMemory(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, gamma, functions, init_function, n_fun_args, *params):
        fun_args = params[:n_fun_args]
        ctx.functions = functions
        ctx.gamma = gamma
        ctx.init_function = init_function
        ctx.fun_args = fun_args
        ctx.params_require_grad = [param.requires_grad for param in params]
        n_iters = len(functions)
        v = TorchExactRep(v)
        with torch.no_grad():
            for i in range(n_iters):
                v *= gamma
                v += (1 - gamma) * (functions[i](x, *fun_args) - x)
                x = x + v.val
        ctx.save_for_backward(x, v.intrep, v.aux.store)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        functions = ctx.functions
        gamma = ctx.gamma
        fun_args = ctx.fun_args
        fun_args_requires_grad  = [param.requires_grad for param in fun_args]
        n_fun_grad = sum(fun_args_requires_grad)
        params_require_grad = ctx.params_require_grad
        n_iters = len(functions)
        x, v_intrep, v_store = ctx.saved_tensors
        v = TorchExactRep(0, from_representation=(v_intrep, v_store))
        grad_x = grad_output
        grad_v = torch.zeros_like(grad_x)
        grad_params = []
        with torch.set_grad_enabled(True):
            for i in range(n_iters):
                function = functions[n_iters - 1 - i]
                x = x.detach().requires_grad_(False)
                x = x - v.val
                x = x.detach().requires_grad_(True)
                f_eval = (function(x, *fun_args) - x)
                grad_combi = grad_x + grad_v
                backward_list = []
                for requires_grad, param in zip(params_require_grad, fun_args + tuple(function.parameters())):
                    if requires_grad:
                        backward_list.append(param)
                vjps = torch.autograd.grad(
                    f_eval, (x,) + tuple(backward_list), grad_combi
                )
                v += -(1 - gamma) * f_eval
                v /= gamma
                grad_params.append([(1 - gamma) * vjp for vjp in vjps[1:]])
                grad_x = grad_x + (1 - gamma) * vjps[0]
                grad_v = gamma * grad_combi
        flat_params_vjp = []

        for param in grad_params[::-1]:
            flat_params_vjp += param[n_fun_grad:]
        flat_param_fun = grad_params[::-1][0][:n_fun_grad]
        for param in grad_params[::-1][1:]:
            for j in range(n_fun_grad):
                flat_param_fun[j] = flat_param_fun[j] + param[j]
        flat_params_vjp = flat_param_fun + flat_params_vjp
        flat_params = []
        i = 0
        for requires_grad in params_require_grad:
            if requires_grad:
                flat_params.append(flat_params_vjp[i])
                i += 1
            else:
                flat_params.append(None)   # ENH: improve this to make it cleaner
        return (grad_x, grad_v, None, None, None, None, *flat_params)


class MomentumNetTransformNoBackprop(nn.Module):
    """
    A class used to define a Momentum ResNet with the memory tricks

    ...

    Attributes
    ----------
    functions : list of nn, list of Sequential or Sequential
        a list of Sequential to define the transformation at each layer
    gamma : float
        the momentum term
    init_speed : bool
        if init_speed is True then specify an init_function
    init_function : Sequential


    Methods
    -------
    forward(x)
        maps x to the output of the network
    """

    def __init__(self, functions, gamma, init_speed=False, init_function=None):
        super(MomentumNetTransformNoBackprop, self).__init__()
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

    def forward(self, x, *function_args):
        if not self.init_speed:
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
        #params += list(function_args)
        # function_args = list(function_args)
        n_fun_args = len(function_args)
        output = MomentumTransformMemory.apply(
            x, v, self.gamma, functions, init_function, n_fun_args, *function_args, *params,
        )
        self.v = v
        return output

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    @property
    def init_function(self):
        return self._modules["init"]


class MomentumNetTransform(nn.Module):
    """
    Create a Momentum Residual Network.

    It iterates

    v_{t + 1} = (1 - gamma) * v_t + gamma * f_t(x_t)
    x_{t + 1} = x_t + v_{t + 1}

    where the f_t are stored in `functions`.
    These forward equations can be reversed in closed-form,
    enabling learning without backpropagation. This process
    trades memory for computations.

    ...

    Attributes
    ----------
    functions : list of nn, list of Sequential or Sequential
        a list of Sequential to define the transformation at each layer
    gamma : float
        the momentum term
    init_speed : bool
        Whether to initialize v as 0 or as a function of the input.
    init_function : Sequential
        The initial function
    use_backprop : bool
        Whether to use backprop or not to compute the gradient of
        the parameters.


    Methods
    -------
    forward(x, n_iters=None)
        maps x to the output of the network
    """

    def __init__(
        self,
        functions,
        gamma,
        init_speed=False,
        init_function=None,
        use_backprop=True,
    ):
        super(MomentumNetTransform, self).__init__()
        if use_backprop:
            self.network = MomentumNetTransformWithBackprop(
                functions, gamma, init_speed, init_function
            )
        else:
            self.network = MomentumNetTransformNoBackprop(
                functions, gamma, init_speed, init_function
            )
        self.use_backprop = use_backprop
        self.gamma = gamma
        self.init_speed = init_speed

    def forward(self, x, *args, **kwargs):
        return self.network.forward(x, *args)

    @property
    def functions(self):
        return self.network.functions

    @property
    def init_function(self):
        return self.network.init_function
