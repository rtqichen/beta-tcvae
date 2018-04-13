import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.dist import Normal

eps = 1e-8


class FactorialNormalizingFlow(nn.Module):

    def __init__(self, dim, nsteps):
        super(FactorialNormalizingFlow, self).__init__()
        self.dim = dim
        self.nsteps = nsteps
        self.x_dist = Normal()
        self.scale = nn.Parameter(torch.Tensor(self.nsteps, self.dim))
        self.weight = nn.Parameter(torch.Tensor(self.nsteps, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.nsteps, self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.scale.data.normal_(0, 0.02)
        self.weight.data.normal_(0, 0.02)
        self.bias.data.normal_(0, 0.02)

    def sample(self, batch_size):
        raise NotImplementedError

    def log_density(self, y, params=None):
        assert(y.size(1) == self.dim)
        x = y
        logdetgrad = Variable(torch.zeros(y.size()).type_as(y.data))
        for i in range(self.nsteps):
            u = self.scale[i][None]
            w = self.weight[i][None]
            b = self.bias[i][None]
            act = F.tanh(x * w + b)
            x = x + u * act
            logdetgrad = logdetgrad + torch.log(torch.abs(1 + u * (1 - act.pow(2)) * w) + eps)
        logpx = self.x_dist.log_density(x)
        logpy = logpx + logdetgrad
        return logpy
