from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions.categorical import Categorical

from visualisation import plot_grad_flow


class MoeMlpPolicy(nn.Module):
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


class MoeGenPolicy(nn.Module):
    def __init__(self, n_component, input_dim, hidden_dim):
        super().__init__()
        self.n_component = n_component
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_component, 1), requires_grad=True)
        torch.nn.init.uniform_(self.pi, 1. / self.n_component, 1. / self.n_component + 0.01)
        self.actor_mlp = MoeMlpPolicy(input_dim, hidden_dim, n_component)
        self.action_shape = 3

    def forward(self, x):

        a1, a2, a3 = self.actor_mlp(x)

        # mu.clamp_(float(0.0), float(float(self.action_shape - 1)))
        # # Clamp each dim of mu based on the (low,high) limits of that action dim
        # sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7  # Let sigma be (smoothly) +ve
        # sigma_stack = torch.stack([torch.stack([torch.stack([torch.eye(self.action_shape)
        #                                                      * k for k in i]) for i in j]) for j in sigma])
        # dis_lst = []
        # sample_pi_all = torch.zeros_like(mu)

        logits_ = self.pi[0, 0, 0] * a1 + self.pi[0, 1, 0] * a2 + self.pi[0, 2, 0] * a3
        log_pi = torch.nn.LogSoftmax(dim=-1)(logits_)
        logits = log_pi.exp()
        action_distribution = Categorical(logits)
        sample_groups = action_distribution.sample()
        log_sample = action_distribution.log_prob(sample_groups)[:, :, None]
        return sample_groups, logits, log_sample


if __name__ == "__main__":
    n_comp, in_dim, hid_dim = 3, 2, 128
    gmm = MoeGenPolicy(n_comp, in_dim, hid_dim)
    data_x = torch.rand(32, 50, 2)
    y = gmm(data_x)

    plot_grad_flow(gmm.named_parameters(), logdir='.')
