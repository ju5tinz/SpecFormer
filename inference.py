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

def predict_dataset(model, criterion, data_loader, amino_acid_alphabet, output_file_name):
    """
    Runs inference on a dataset and writes predictions to a file.
    """
    print(f"Writing predictions to: {output_file_name}")
    losses = []
    with open(output_file_name, "w") as f:
        for sequences, labels, charges, NCEs, mass_range_strs in data_loader:
            output = model(sequences, charges, NCEs)
            output = threshold_tensor(output, threshold)
            batch_losses = criterion(output, labels)
            losses.extend(batch_losses)
            for i in range(sequences.shape[0]):
                seq = one_hot_to_seq(sequences[i], amino_acid_alphabet)
                f.write(f'>length={len(seq)}; sequence={seq}; charge={charges[i]}; NCE={int(NCEs[i])}; mass range={mass_range_strs[i]}\n')
                for row_i, row in enumerate(output[i]):
                    if row != 0:
                        f.write(f'{row_i},{row}\n')
    return losses

def inference_splits(checkpoint_path: str, alphabet_path: str, batch_size: int = 16, subset_size: int = 60000):
    """
    Runs inference on train/val/test splits and returns losses.
    """
    device = get_device()
    print(f"Using device: {device}")

    dataset = SpectrumDataset(root_dir='processed/', alphabet_path=alphabet_path, filenames=os.listdir('processed/'))
    print(f"Dataset size: {len(dataset)}")

    train_subset, val_subset, test_subset = generate_dataset_subsets(dataset, seed=42)

    random_train_indices = torch.randperm(len(train_subset))
    random_train_subset = Subset(train_subset, random_train_indices[:subset_size])

    train_loader = DataLoader(random_train_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    alphabet = get_list_from_file(alphabet_path)

    model = SpectrumModel(
        token_size=dataset.token_dim, dict_size=dataset.ionDictN, seq_max_len=max_seq_len
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = FilteredCosineLoss()

    with torch.no_grad():
        train_losses = predict_dataset(model, criterion, train_loader, alphabet, "predicted/train_predict.txt")
        val_losses = predict_dataset(model, criterion, val_loader, alphabet, "predicted/val_predict.txt")
        test_losses = predict_dataset(model, criterion, test_loader, alphabet, "predicted/test_predict.txt")

    return train_losses, val_losses, test_losses

def inference_file(filename: str, checkpoint_path: str, alphabet_path: str, batch_size: int = 16):
    """
    Runs inference on a single file and returns losses.
    """
    device = get_device()
    print(f"Using device: {device}")

    dataset = SpectrumDataset(root_dir='processed/', alphabet_path=alphabet_path, filenames=[filename])
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    alphabet = get_list_from_file(alphabet_path)

    model = SpectrumModel(
        token_size=len(alphabet), dict_size=dataset.ionDictN, seq_max_len=max_seq_len
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = FilteredCosineLoss()

    with torch.no_grad():
        losses = predict_dataset(model, criterion, dataloader, alphabet, f"predicted/{filename}")
    
    return losses