from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

class MLP_policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc_1 = nn.Linear(hidden_dim, output_dim)
        self.output_fc_2 = nn.Linear(hidden_dim, output_dim)
        self.output_fc_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = [batch size, height, width]

        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))

        a1 = self.output_fc_1(h_2)
        a2 = self.output_fc_2(h_2)
        a3 = self.output_fc_3(h_2)

        return a1, a2, a3


class GMM_policy(nn.Module):
    def __init__(self, n_component, input_dim, hidden_dim):
        super().__init__()
        self.n_component = n_component
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_component, 1), requires_grad=True)
        torch.nn.init.uniform_(self.pi, 1. / self.n_component, 1. / self.n_component + 0.01)
        self.actor_mlp = MLP_policy(input_dim, hidden_dim, n_component)
        self.action_shape = 3

    def forward(self, x):

        a1, a2, a3 = self.actor_mlp(x)
        # mu.clamp_(float(0.0), float(float(self.action_shape - 1))) # # Clamp each dim of mu based on the (low,high) limits of that action dim
        # sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7  # Let sigma be (smoothly) +ve
        # sigma_stack = torch.stack([torch.stack([torch.stack([torch.eye(self.action_shape) * k for k in i]) for i in j]) for j in sigma])
        # dis_lst = []
        # sample_pi_all = torch.zeros_like(mu)
        logits_ = self.pi[0, 0, 0] * a1 + self.pi[0, 1, 0] * a2 + self.pi[0, 2, 0] * a3
        log_pi = torch.nn.LogSoftmax(dim=-1)(logits_)
        logits = log_pi.exp()
        action_distribution = Categorical(logits)
        sample_groups = action_distribution.sample()
        log_sample = action_distribution.log_prob(sample_groups)[:,:,None]
        return sample_groups, logits, log_sample



def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])



if __name__ == "__main__":
    n_component, input_dim, hidden_dim = 3, 2, 128
    gmm = GMM_policy(n_component, input_dim, hidden_dim)
    x = torch.rand(32, 50, 2)
    y = gmm(x)

    plot_grad_flow(gmm.named_parameters())