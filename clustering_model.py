from torch import nn
import torch.nn.functional as F


class ClusteringMLP(nn.Module):
    def __init__(self, k, input_dim, hidden_dim=8):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, k)

    def forward(self, x):
        # x = [batch size, height, width]

        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))

        s = F.softmax(self.output_fc(h_2), dim=-1)

        return s
