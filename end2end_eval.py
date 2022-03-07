import torch
import numpy as np


from clustering_model import ClusteringMLP

model = ClusteringMLP(3, 2)
model.load_state_dict(torch.load('example_model.pt'))
model.eval()

x = torch.rand(20, 2)
s = model(x)


