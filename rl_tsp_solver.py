import os
import numpy as np
import torch

from rl_tsp_utils import load_model


def rl_tsp_solver():
    model, _ = load_model('pretrained/tsp_100/')
    model.eval()  # Put in evaluation mode to not track gradients
