from momentumnet import MomentumNet, Mom_without_mem, Mom
import torch
import torch.nn as nn
from momentumnet.exact_rep_pytorch import TorchExactRep

# functions = [nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3)) for _ in range(10)]
# init_function = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3))
# mom_net = Mom(functions, beta=0.01, init_speed=1, init_function=init_function)
# mom = Mom_without_mem(functions, beta=0.01, init_speed=1, init_function=init_function)
# x = torch.rand(3, requires_grad=True)
# print(mom_net(x) - mom(x))
# torch.allclose(mom_net(x), mom(x),rtol = 1e-04)
#
# functions = [nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3)) for _ in range(5)]
# init_function = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3))
# mom = Mom(functions, beta=0.01, init_speed=1, init_function=init_function)
# z = TorchExactRep(torch.tensor(1))
# n, d = z.float_to_rational(.01)
# mom_net = MomentumNet(functions, beta=0.01, init_speed=1, init_function=init_function)
# x = torch.rand(3, requires_grad=True)
res = []
for depth in range(150,200):
    functions = [nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 3)) for _ in range(depth)]
    init_function = nn.Sequential(nn.Linear(1, 1), nn.Tanh(), nn.Linear(1, 1))
    mom_net = MomentumNet(functions,  beta=0.1, init_speed=0, init_function=None)
    #mom = Mom(functions,  beta=0.01, init_speed=0, init_function=None)
    mom_mem = Mom(functions,  beta=0.1, init_speed=0, init_function=None)
    x = torch.rand(3, requires_grad=True)
    #assert torch.allclose(mom_net(x), mom(x),rtol=1e-4)
    mom_net_output = (mom_net(x)**3 + mom_net(x)**2).sum()
    mom_mem_output = (mom_mem(x)**3 + mom_mem(x)**2).sum()
    params = tuple(mom_net.parameters())
    grad_1 = torch.autograd.grad(mom_net_output,  params)
    # print('')
    # params = tuple(mom.parameters())
    # print(torch.autograd.grad(mom_output,  params))
    # print('')
    params = tuple(mom_mem.parameters())
    grad_2 = torch.autograd.grad(mom_mem_output,  params)
    res.append(torch.max(torch.abs(grad_1[2] - grad_2[2])))

#print(torch.autograd.grad(mom_net_output, x)[0] -  torch.autograd.grad(mom_output, x)[0])
#assert torch.allclose(torch.autograd.grad(mom_net_output, x)[0], torch.autograd.grad(mom_output, x)[0],rtol=1e-4)