import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import random_split

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def write_list_to_file(filename, in_list):
    with open(filename, 'w') as f:
        for item in in_list:
            f.write(item + '\n')

    return 

def get_list_from_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]
    return lines

def threshold_tensor(input_tensor, threshold):
    return torch.where(input_tensor < threshold, 0.0, input_tensor)

def generate_dataset_subsets(dataset, seed):
    total_size = len(dataset)
    train_size = int(0.96 * total_size)
    val_size = int(0.02 * total_size)
    test_size = total_size - train_size - val_size

    print("train size: ", train_size)
    print("val size: ", val_size)
    print("test size: ", test_size)

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    return train_dataset, val_dataset, test_dataset

def cosine_loss(output, target):
    return 1 - F.cosine_similarity(output, target, dim=1).mean()

class CosineLoss(nn.Module):
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self, dim: int = 0, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1_sqrt = x1.sqrt()
        x2_sqrt = x2.sqrt()
        return 1 - F.cosine_similarity(x1, x2, self.dim, self.eps)

class FilteredCosineLoss(nn.Module):
    """
    Cosine loss that filters out masked labels (value -1).

    Args:
        reduction: 'mean' (default), 'sum', or 'none' for per-sample losses.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.cosine_loss = CosineLoss()
        self.reduction = reduction

    def forward(self, outputs, labels):
        batches_n = outputs.shape[0]
        losses = []
        for i in range(batches_n):
            mask = labels[i] != -1
            loss = self.cosine_loss(outputs[i][mask], labels[i][mask])
            losses.append(loss)

        if self.reduction == 'none':
            return losses
        stacked = torch.stack(losses)
        if self.reduction == 'sum':
            return stacked.sum()
        return stacked.mean()