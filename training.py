import torch
import torch.multiprocessing
from spectrum_dataset import SpectrumDataset, collate_fn
from spectrum_model import SpectrumModel
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import generate_dataset_subsets, FilteredCosineLoss, get_device
import os

max_seq_len = 40
patience = 5

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains model for one epoch.
    """
    model.train()
    total_loss = 0.0
    for sequences, labels, charges, NCEs, _ in dataloader:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        charges = charges.to(device, non_blocking=True)
        NCEs = NCEs.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(sequences, charges, NCEs)
        losses = criterion(outputs, labels)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates model on validation/test set.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sequences, labels, charges, NCEs, _ in dataloader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            charges = charges.to(device, non_blocking=True)
            NCEs = NCEs.to(device, non_blocking=True)
            
            outputs = model(sequences, charges, NCEs)
            losses = criterion(outputs, labels)
            loss = torch.stack(losses).mean()
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(dataset_dir: str,
          model_args: dict = {},
          weights_file: str = None,
          filenames=None,
          batch_size: int = 128,
          learning_rate: float = 1e-4,
          weight_decay: float = 0.001,
          epochs: int = 100,
          split_dataset: bool = True,
          train_data_filenames=None,
          val_data_filenames=None,
          alphabet_path: str = 'config/amino_acid_alphabet.txt'):
    """
    Trains SpectrumModel on provided dataset.
    """
    if filenames is None:
        filenames = []
    if train_data_filenames is None:
        train_data_filenames = []
    if val_data_filenames is None:
        val_data_filenames = []
    device = get_device()
    print(f"Using device: {device}")

    dataset = None
    train_loader = None
    val_loader = None
    token_size = None
    dict_size = None

    if split_dataset:
        dataset = SpectrumDataset(root_dir=dataset_dir, alphabet_path=alphabet_path, filenames=filenames)
        print(f"Dataset size: {len(dataset)}")

        train_subset, val_subset, _ = generate_dataset_subsets(dataset=dataset, seed=42)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=False)

        token_size = dataset.token_dim
        dict_size = dataset.ionDictN

    else:
        train_dataset = SpectrumDataset(root_dir=dataset_dir, alphabet_path=alphabet_path, filenames=train_data_filenames)
        val_dataset = SpectrumDataset(root_dir=dataset_dir, alphabet_path=alphabet_path, filenames=val_data_filenames)

        print(f"Train size: {len(train_dataset)}")
        print(f"Val size: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=False)

        token_size = train_dataset.token_dim
        dict_size = train_dataset.ionDictN

    model = SpectrumModel(token_size=token_size, dict_size=dict_size, seq_max_len=max_seq_len, **model_args).to(device)

    if weights_file:
        print(f"Loading model weights from: {weights_file}")
        model.load_state_dict(torch.load(weights_file, map_location=device))

    opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = FilteredCosineLoss()

    training_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improvement = 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, opt, criterion, device)
        print(f"Epoch {epoch} trained")
        val_loss = evaluate(model, val_loader, criterion, device)

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
    train('processed/', 
          split_dataset=False, 
          train_data_filenames=['AItrain_LumosSynthetic_2022418v2.ann.txt', 'AItrain_QEHumanCho_2022418v2.ann.txt'],
          val_data_filenames=['ValidUniq2022418_202333.ann.txt'])
