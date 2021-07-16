# Authors: Michael Sander, Pierre Ablin
# License: MIT
import torch
import torch.nn as nn
from .exact_rep_pytorch import TorchExactRep


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
        return x

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    @property
    def init_function(self):
        return self._modules["init"]


class MomentumTransformMemory(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        v,
        gamma,
        functions,
        init_function,
        is_residual,
        n_fun_args,
        *params
    ):
        fun_args = params[:n_fun_args]
        ctx.functions = functions
        ctx.gamma = gamma
        ctx.init_function = init_function
        ctx.fun_args = fun_args
        ctx.is_residual = is_residual
        ctx.params_require_grad = [
            param.requires_grad for param in params if param is not None
        ]
        ctx.is_residual = is_residual
        n_iters = len(functions)
        v = TorchExactRep(v)
        with torch.no_grad():
            for i in range(n_iters):
                v *= gamma
                f = functions[i](x, *fun_args)
                if not is_residual:
                    f = f - x
                v += (1 - gamma) * f
                x = x + v.val
        ctx.save_for_backward(x, v.intrep, v.aux.store)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        functions = ctx.functions
        gamma = ctx.gamma
        fun_args = ctx.fun_args
        fun_args_requires_grad = [param.requires_grad for param in fun_args]
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
                if ctx.is_residual:
                    f_eval = function(x, *fun_args)
                else:
                    f_eval = function(x, *fun_args) - x
                grad_combi = grad_x + grad_v
                backward_list = []
                for requires_grad, param in zip(
                    params_require_grad,
                    fun_args + tuple(function.parameters()),
                ):
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
            if ctx.init_function is not None:
                x = x.detach().requires_grad_(True)
                f_eval = ctx.init_function(x)
                params = tuple(ctx.init_function.parameters())
                vjps = torch.autograd.grad(
                    f_eval, params, torch.zeros_like(grad_x)
                )
                grad_params.append([vjp for vjp in vjps])
            else:
                pass
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
                flat_params.append(
                    None
                )  # ENH: improve this to make it cleaner
        return (grad_x, grad_v, None, None, None, None, None, *flat_params)


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
        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.v = None
        self.is_residual = is_residual
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
        n_fun_args = len(function_args)
        output = MomentumTransformMemory.apply(
            x,
            v,
            self.gamma,
            functions,
            init_function,
            self.is_residual,
            n_fun_args,
            *function_args,
            *params,
        )
        self.v = v
        return output

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    @property
    def init_function(self):
        return self._modules["init"]


class MomentumNet(nn.Module):
    """
    Create a Momentum Residual Network.

    It iterates

    v_{t + 1} = (1 - gamma) * v_t + gamma * f_t(x_t)
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
        return self.network.forward(x, *args)

    @property
    def functions(self):
        return self.network.functions

    @property
    def init_function(self):
        return self.network.init_function
