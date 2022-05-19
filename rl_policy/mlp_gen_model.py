from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions.categorical import Categorical


class MlpPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc_1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = [batch size, height, width]

        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))

        a1 = self.output_fc_1(h_2)

        return a1


class MlpGenPolicy(nn.Module):
    def __init__(self, cluster_num, input_dim, hidden_dim):
        super().__init__()
        self.cluster_num = cluster_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.actor_mlp = MlpPolicy(input_dim, hidden_dim, cluster_num)

    def forward(self, x):
        a1 = self.actor_mlp(x)

        log_pi = torch.nn.LogSoftmax(dim=-1)(a1)
        logits = log_pi.exp()
        action_distribution = Categorical(logits)
        sample_groups = action_distribution.sample()
        log_sample = action_distribution.log_prob(sample_groups)[:, :, None]
        return sample_groups, logits, log_sample


if __name__ == "__main__":
    pass
