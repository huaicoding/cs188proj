from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module

from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim

a = ones(1, 2)
b = ones(1, 2)
print(a.shape)
print(a, b)
c = tensordot(a, b)
print(c)
print(c > 0)