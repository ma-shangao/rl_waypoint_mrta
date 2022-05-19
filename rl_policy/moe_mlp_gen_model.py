from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions.categorical import Categorical


class MoeMlpPolicy(nn.Module):
    def __init__(self, n_component, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.n_component = n_component
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.net = nn.ModuleList([])
        for i in range(n_component):
            self.net.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        # x = [batch size, height, width]
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))

        output_a = []
        for i in range(self.n_component):
            output_a.append(self.net[i](h_2))

        return output_a


class MoeGenPolicy(nn.Module):
    def __init__(self, n_component, input_dim, hidden_dim, cluster_num):
        super().__init__()
        self.n_component = n_component
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_component, 1), requires_grad=True)
        torch.nn.init.uniform_(self.pi, 1. / self.n_component, 1. / self.n_component + 0.01)
        self.actor_mlp = MoeMlpPolicy(n_component, input_dim, hidden_dim, cluster_num)

    def forward(self, x):

        output_a = self.actor_mlp(x)

        logits_ = torch.zeros_like(output_a[0])
        for i in range(self.n_component):
            logits_ += self.pi[0, i, 0] * output_a[i]

        log_pi = torch.nn.LogSoftmax(dim=-1)(logits_)
        logits = log_pi.exp()
        action_distribution = Categorical(logits)
        sample_groups = action_distribution.sample()
        log_sample = action_distribution.log_prob(sample_groups)[:, :, None]
        return sample_groups, logits, log_sample


if __name__ == "__main__":
    n_comp, in_dim, hid_dim, o_dim = 3, 2, 128, 3
    gmm = MoeGenPolicy(n_comp, in_dim, hid_dim, o_dim)

    data_x = torch.rand(32, 50, 2)
    y = gmm(data_x)
