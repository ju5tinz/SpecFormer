import os
import torch
from spectrum_dataset import SpectrumDataset, collate_fn
from torch.utils.data import DataLoader, Subset
from spectrum_model import SpectrumModel
from utils import threshold_tensor, generate_dataset_subsets, FilteredCosineLoss, get_device, get_list_from_file

file_path = ''
data_root_dir = os.path.join(file_path, "processed/")

max_seq_len = 40
batch_size = 16
threshold = 1e-4

def one_hot_to_seq(one_hot_seq, alphabet):
    """
    Converts a one-hot encoded sequence tensor to a string using the provided alphabet.
    """
    alphabet = sorted(alphabet)
    is_token_mask = torch.any(one_hot_seq, dim=1)
    actual_length = torch.sum(is_token_mask)
    indices = torch.argmax(one_hot_seq[:actual_length], dim=-1)
    seq = ''.join([alphabet[i] for i in indices])
    return seq

def load_ensemble_models(checkpoint_paths: list, token_size: int, dict_size: int, device: torch.device):
    """
    Loads multiple model checkpoints for ensemble inference.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        token_size: Input token dimension
        dict_size: Output dictionary size
        device: Device to load models on
    
    Returns:
        List of loaded models in eval mode
    """
    models = []
    for path in checkpoint_paths:
        model = SpectrumModel(
            token_size=token_size, dict_size=dict_size, seq_max_len=max_seq_len
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded model from: {path}")
    return models

def ensemble_predict(models: list, sequences, charges, NCEs, device):
    """
    Generates ensemble prediction by averaging outputs from multiple models.
    
    Args:
        models: List of models
        sequences: Input sequences tensor
        charges: Charges tensor
        NCEs: NCE values tensor
        device: Device to run inference on
    
    Returns:
        Averaged prediction tensor
    """
    sequences = sequences.to(device)
    charges = charges.to(device)
    NCEs = NCEs.to(device)
    
    outputs = []
    for model in models:
        output = model(sequences, charges, NCEs)
        outputs.append(output)
    
    # Stack and average predictions
    stacked = torch.stack(outputs, dim=0)
    return stacked.mean(dim=0)

def predict_dataset_ensemble(models, criterion, data_loader, amino_acid_alphabet, output_file_name, device):
    """
    Runs ensemble inference on a dataset and writes predictions to a file.
    """
    print(f"Writing ensemble predictions to: {output_file_name}", flush=True)
    losses = []
    total_batches = len(data_loader)
    total_samples = 0
    
    with open(output_file_name, "w") as f:
        for batch_idx, (sequences, labels, charges, NCEs, mass_range_strs) in enumerate(data_loader):
            labels = labels.to(device)
            output = ensemble_predict(models, sequences, charges, NCEs, device)
            output = threshold_tensor(output, threshold)
            batch_losses = criterion(output, labels)
            losses.extend([loss.item() for loss in batch_losses])
            
            batch_size = sequences.shape[0]
            total_samples += batch_size
            avg_batch_loss = sum(loss.item() for loss in batch_losses) / batch_size
            
            if batch_idx == 0 or (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
                print(f"Batch {batch_idx + 1}/{total_batches} | Samples: {total_samples} | Batch Loss: {avg_batch_loss:.6f} | Running Avg Loss: {sum(losses)/len(losses):.6f}", flush=True)
            
            # Move output to CPU for file writing
            output_cpu = output.cpu()
            
            for i in range(batch_size):
                seq = one_hot_to_seq(sequences[i], amino_acid_alphabet)
                f.write(f'>length={len(seq)}; sequence={seq}; charge={charges[i]}; NCE={int(NCEs[i])}; mass range={mass_range_strs[i]}\n')
                
                # Use nonzero for efficient sparse writing
                nonzero_indices = torch.nonzero(output_cpu[i], as_tuple=True)[0]
                for row_i in nonzero_indices:
                    f.write(f'{row_i.item()},{output_cpu[i][row_i].item()}\n')
    
    print(f"Completed: {total_samples} samples | Final Avg Loss: {sum(losses)/len(losses):.6f}", flush=True)
    return losses

def inference_ensemble(checkpoint_paths: list, filename: str, alphabet_path: str, batch_size: int = 16):
    """
    Runs ensemble inference on a single file using multiple model checkpoints.
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Ensemble size: {len(checkpoint_paths)} models")

    dataset = SpectrumDataset(root_dir='processed/', alphabet_path=alphabet_path, filenames=[filename])
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    alphabet = get_list_from_file(alphabet_path)

    models = load_ensemble_models(
        checkpoint_paths, 
        token_size=len(alphabet), 
        dict_size=dataset.ionDictN, 
        device=device
    )

    criterion = FilteredCosineLoss()

    with torch.no_grad():
        losses = predict_dataset_ensemble(
            models, criterion, dataloader, alphabet, f"predicted/ensemble_{filename}", device
        )
    
    return losses

def inference_ensemble_splits(checkpoint_paths: list, alphabet_path: str, batch_size: int = 16, subset_size: int = 60000):
    """
    Runs ensemble inference on train/val/test splits and returns losses.
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Ensemble size: {len(checkpoint_paths)} models")

    dataset = SpectrumDataset(root_dir='processed/', alphabet_path=alphabet_path, filenames=os.listdir('processed/'))
    print(f"Dataset size: {len(dataset)}")

    train_subset, val_subset, test_subset = generate_dataset_subsets(dataset, seed=42)

    random_train_indices = torch.randperm(len(train_subset))
    random_train_subset = Subset(train_subset, random_train_indices[:subset_size])

    train_loader = DataLoader(random_train_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    alphabet = get_list_from_file(alphabet_path)

    models = load_ensemble_models(
        checkpoint_paths,
        token_size=dataset.token_dim,
        dict_size=dataset.ionDictN,
        device=device
    )

    criterion = FilteredCosineLoss()

    with torch.no_grad():
        train_losses = predict_dataset_ensemble(models, criterion, train_loader, alphabet, "predicted/ensemble_train_predict.txt")
        val_losses = predict_dataset_ensemble(models, criterion, val_loader, alphabet, "predicted/ensemble_val_predict.txt")
        test_losses = predict_dataset_ensemble(models, criterion, test_loader, alphabet, "predicted/ensemble_test_predict.txt")

    return train_losses, val_losses, test_losses