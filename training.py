import torch
torch.multiprocessing.set_start_method('spawn', force=True)
from spectrum_dataset import SpectrumDataset, collate_fn
from spectrum_model import SpectrumModel
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch import Tensor

max_seq_len = 40
batch_size = 128

epochs = 2000
patience = 5

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
    def __init__(self):
        super().__init__()
        self.cosine_loss = CosineLoss()

    def forward(self, outputs, labels):
        batches_n = outputs.shape[0]
        total_loss = 0
        for i in range(batches_n):
            mask = labels[i] != -1
            loss = self.cosine_loss(outputs[i][mask], labels[i][mask])
            total_loss += loss
        return total_loss / batches_n

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for sequences, labels, charges, NCEs in dataloader:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        charges = charges.to(device, non_blocking=True)
        NCEs = NCEs.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(sequences, charges, NCEs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sequences, labels, charges, NCEs in dataloader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            charges = charges.to(device, non_blocking=True)
            NCEs = NCEs.to(device, non_blocking=True)
            
            outputs = model(sequences, charges, NCEs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train(dataset_dir, exclude_files=[]):
    #__spec__ = None
    device = get_device()
    print(f"Using device: {device}")

    dataset = SpectrumDataset(root_dir=dataset_dir, exclude_file=exclude_files)
    print("dataset size:", len(dataset))

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    pin_memory = True
    if str(device) == "mps":
        pin_memory = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=pin_memory)

    model = SpectrumModel(token_size=dataset.token_dim, dict_size=dataset.ionDictN, seq_max_len=max_seq_len).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    crit = FilteredCosineLoss()

    training_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improvement = 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, opt, crit, device)
        val_loss = evaluate(model, val_loader, crit, device)

        training_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improvement = 0
            torch.save(model.state_dict(), "checkpoints/best_model_test.pth")
            print("Saved best model")
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} with no improvement for {epochs_no_improvement} epochs.")
            break

if __name__ == "__main__":
    train('processed/')
        