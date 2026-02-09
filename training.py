import torch
import torch.multiprocessing
from spectrum_dataset import SpectrumDataset, get_collate_fn
from spectrum_model import SpectrumModel
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import generate_dataset_subsets, FilteredCosineLoss, get_device
from config import ModelConfig, TrainConfig, DataConfig
from datetime import datetime


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Trains model for one epoch."""
    model.train()
    total_loss = 0.0
    for sequences, labels, charges, NCEs, _ in dataloader:
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
    """Evaluates model on validation/test set."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sequences, labels, charges, NCEs, _ in dataloader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            charges = charges.to(device, non_blocking=True)
            NCEs = NCEs.to(device, non_blocking=True)

            outputs = model(sequences, charges, NCEs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(data_config: DataConfig,
          model_config: ModelConfig = None,
          train_config: TrainConfig = None,
          weights_file: str = None):
    """
    Trains SpectrumModel on provided dataset.

    Args:
        data_config: Data configuration (directories, files, alphabet).
        model_config: Model architecture configuration. Uses defaults if None.
        train_config: Training hyperparameters. Uses defaults if None.
        weights_file: Optional path to pretrained weights.
    """
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainConfig()

    device = get_device()

    start_time = datetime.now()
    print("=" * 60)
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {device}")
    print("=" * 60)

    collate_fn = get_collate_fn(model_config.max_seq_len)

    # Load datasets based on whether train/val files are specified
    if data_config.train_files and data_config.val_files:
        train_dataset = SpectrumDataset(
            root_dir=data_config.data_dir,
            alphabet_path=data_config.alphabet_path,
            filenames=data_config.train_files
        )
        val_dataset = SpectrumDataset(
            root_dir=data_config.data_dir,
            alphabet_path=data_config.alphabet_path,
            filenames=data_config.val_files
        )
        print(f"Train size: {len(train_dataset)}")
        print(f"Val size: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=train_config.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=train_config.num_workers, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=train_config.num_workers, pin_memory=False
        )
        token_size = train_dataset.token_dim
        dict_size = train_dataset.ionDictN
    else:
        # Auto-split mode
        dataset = SpectrumDataset(
            root_dir=data_config.data_dir,
            alphabet_path=data_config.alphabet_path
        )
        print(f"Dataset size: {len(dataset)}")

        train_subset, val_subset, _ = generate_dataset_subsets(dataset=dataset, seed=42)

        train_loader = DataLoader(
            train_subset, batch_size=train_config.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=train_config.num_workers, pin_memory=False
        )
        val_loader = DataLoader(
            val_subset, batch_size=train_config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=train_config.num_workers, pin_memory=False
        )
        token_size = dataset.token_dim
        dict_size = dataset.ionDictN

    model = SpectrumModel(
        token_size=token_size,
        dict_size=dict_size,
        embed_dim=model_config.embed_dim,
        num_heads=model_config.num_heads,
        seq_max_len=model_config.max_seq_len,
        penultimate_dim=model_config.penultimate_dim,
        dropout_rate=model_config.dropout_rate
    ).to(device)

    print("\nModel Details:")
    print(f"  Token size: {token_size}")
    print(f"  Dictionary size: {dict_size}")
    print(f"  Max sequence length: {model_config.max_seq_len}")
    print(f"  Embed dim: {model_config.embed_dim}")
    print(f"  Num heads: {model_config.num_heads}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    if weights_file:
        print(f"Loading model weights from: {weights_file}")
        model.load_state_dict(torch.load(weights_file, map_location=device))

    opt = optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    criterion = FilteredCosineLoss()

    best_val_loss = float('inf')
    epochs_no_improvement = 0

    for epoch in range(train_config.epochs):
        train_loss = train_epoch(model, train_loader, opt, criterion, device)
        print(f"Epoch {epoch} trained")
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{train_config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improvement = 0
            torch.save(model.state_dict(), train_config.checkpoint_path)
            print(f"Saved best model to {train_config.checkpoint_path}")
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= train_config.patience:
            print(f"Early stopping at epoch {epoch + 1} with no improvement for {epochs_no_improvement} epochs.")
            break

    end_time = datetime.now()
    duration = end_time - start_time
    print("\n" + "=" * 60)
    print(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training duration: {duration}")
    print("=" * 60)


if __name__ == "__main__":
    data_config = DataConfig(
        data_dir='processed/',
        train_files=['AItrain_LumosSynthetic_2022418v2.ann.txt', 'AItrain_QEHumanCho_2022418v2.ann.txt'],
        val_files=['ValidUniq2022418_202333.ann.txt']
    )
    train(data_config)
