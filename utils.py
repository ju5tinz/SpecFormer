import torch

def threshold_tensor(input_tensor, threshold):
    return torch.where(input_tensor < threshold, 0.0, input_tensor)