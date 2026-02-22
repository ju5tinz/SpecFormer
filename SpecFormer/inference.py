import os
import torch
from spectrum_dataset import SpectrumDataset, get_collate_fn
from torch.utils.data import DataLoader, Subset
from spectrum_model import SpectrumModel
from utils import threshold_tensor, generate_dataset_subsets, FilteredCosineLoss, get_device, get_list_from_file
from config import ModelConfig, InferenceConfig, DataConfig


def one_hot_to_seq(one_hot_seq, alphabet):
    """Converts a one-hot encoded sequence tensor to a string using the provided alphabet."""
    alphabet = sorted(alphabet)
    is_token_mask = torch.any(one_hot_seq, dim=1)
    actual_length = torch.sum(is_token_mask)
    indices = torch.argmax(one_hot_seq[:actual_length], dim=-1)
    seq = ''.join([alphabet[i] for i in indices])
    return seq


def load_ensemble_models(checkpoint_paths: list, token_size: int, dict_size: int,
                         model_config: ModelConfig, device: torch.device):
    """
    Loads multiple model checkpoints for ensemble inference.

    Args:
        checkpoint_paths: List of paths to model checkpoints.
        token_size: Input token dimension.
        dict_size: Output dictionary size.
        model_config: Model configuration.
        device: Device to load models on.

    Returns:
        List of loaded models in eval mode.
    """
    models = []
    for path in checkpoint_paths:
        model = SpectrumModel(
            token_size=token_size,
            dict_size=dict_size,
            embed_dim=model_config.embed_dim,
            num_heads=model_config.num_heads,
            seq_max_len=model_config.max_seq_len,
            penultimate_dim=model_config.penultimate_dim,
            dropout_rate=model_config.dropout_rate
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
        models: List of models.
        sequences: Input sequences tensor.
        charges: Charges tensor.
        NCEs: NCE values tensor.
        device: Device to run inference on.

    Returns:
        Averaged prediction tensor.
    """
    sequences = sequences.to(device)
    charges = charges.to(device)
    NCEs = NCEs.to(device)

    outputs = []
    for model in models:
        output = model(sequences, charges, NCEs)
        outputs.append(output)

    stacked = torch.stack(outputs, dim=0)
    return stacked.mean(dim=0)


def predict_dataset_ensemble(models, data_loader, alphabet, device,
                             inference_config: InferenceConfig,
                             output_file: str = None):
    """
    Runs ensemble inference on a dataset.

    Args:
        models: List of models for ensemble.
        data_loader: DataLoader for the dataset.
        alphabet: Amino acid alphabet list.
        device: Device to run inference on.
        inference_config: Inference configuration.
        output_file: Optional path to write predictions. If None, doesn't write.

    Returns:
        List of per-sample losses.
    """
    criterion = FilteredCosineLoss(reduction='none')
    losses = []
    total_batches = len(data_loader)
    total_samples = 0

    file_handle = open(output_file, "w") if output_file else None

    try:
        if output_file:
            print(f"Writing ensemble predictions to: {output_file}", flush=True)
    
        for batch_idx, (sequences, labels, charges, NCEs, mass_range_strs) in enumerate(data_loader):
            labels = labels.to(device)
            output = ensemble_predict(models, sequences, charges, NCEs, device)
            output = threshold_tensor(output, inference_config.threshold)

            batch_losses = criterion(output, labels)
            losses.extend([loss.item() for loss in batch_losses])

            batch_size = sequences.shape[0]
            total_samples += batch_size
            avg_batch_loss = sum(loss.item() for loss in batch_losses) / batch_size

            if batch_idx == 0 or (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
                print(f"Batch {batch_idx + 1}/{total_batches} | Samples: {total_samples} | "
                      f"Batch Loss: {avg_batch_loss:.6f} | Running Avg Loss: {sum(losses)/len(losses):.6f}",
                      flush=True)

            if file_handle:
                output_cpu = output.cpu()
                for i in range(batch_size):
                    seq = one_hot_to_seq(sequences[i], alphabet)
                    file_handle.write(f'>length={len(seq)}; sequence={seq}; charge={charges[i]}; '
                                      f'NCE={int(NCEs[i])}; mass range={mass_range_strs[i]}\n')
                    nonzero_indices = torch.nonzero(output_cpu[i], as_tuple=True)[0]
                    for row_i in nonzero_indices:
                        file_handle.write(f'{row_i.item()},{output_cpu[i][row_i].item()}\n')

        print(f"Completed: {total_samples} samples | Final Avg Loss: {sum(losses)/len(losses):.6f}", flush=True)
    finally:
        if file_handle:
            file_handle.close()

    return losses


def run_inference(checkpoint_paths: list,
                  data_config: DataConfig,
                  model_config: ModelConfig = None,
                  inference_config: InferenceConfig = None,
                  files: list = None,
                  write_predictions: bool = True):
    """
    Runs ensemble inference on specified files.

    Args:
        checkpoint_paths: List of paths to model checkpoints.
        data_config: Data configuration.
        model_config: Model configuration. Uses defaults if None.
        inference_config: Inference configuration. Uses defaults if None.
        files: List of specific files to run inference on. If None, uses all files in data_dir.
        write_predictions: Whether to write predictions to files.

    Returns:
        Dict mapping filenames to their per-sample losses.
    """
    if model_config is None:
        model_config = ModelConfig()
    if inference_config is None:
        inference_config = InferenceConfig()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Ensemble size: {len(checkpoint_paths)} models")

    alphabet = get_list_from_file(data_config.alphabet_path)
    collate_fn = get_collate_fn(model_config.max_seq_len)

    # Determine files to process
    if files is None:
        files = [f for f in os.listdir(data_config.data_dir) if not f.startswith('.')]

    results = {}

    for filename in files:
        print(f"\nProcessing: {filename}")
        dataset = SpectrumDataset(
            root_dir=data_config.data_dir,
            alphabet_path=data_config.alphabet_path,
            filenames=[filename]
        )
        print(f"Dataset size: {len(dataset)}")

        dataloader = DataLoader(
            dataset,
            batch_size=inference_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        models = load_ensemble_models(
            checkpoint_paths,
            token_size=len(alphabet),
            dict_size=dataset.ionDictN,
            model_config=model_config,
            device=device
        )

        output_file = None
        if write_predictions:
            os.makedirs(inference_config.output_dir, exist_ok=True)
            output_file = os.path.join(inference_config.output_dir, f"ensemble_{filename}")

        with torch.no_grad():
            losses = predict_dataset_ensemble(
                models, dataloader, alphabet, device, inference_config, output_file
            )

        results[filename] = losses

    return results


def run_inference_splits(checkpoint_paths: list,
                         data_config: DataConfig,
                         model_config: ModelConfig = None,
                         inference_config: InferenceConfig = None,
                         subset_size: int = 60000,
                         write_predictions: bool = True):
    """
    Runs ensemble inference on train/val/test splits.

    Args:
        checkpoint_paths: List of paths to model checkpoints.
        data_config: Data configuration.
        model_config: Model configuration. Uses defaults if None.
        inference_config: Inference configuration. Uses defaults if None.
        subset_size: Number of samples to use from train set.
        write_predictions: Whether to write predictions to files.

    Returns:
        Tuple of (train_losses, val_losses, test_losses).
    """
    if model_config is None:
        model_config = ModelConfig()
    if inference_config is None:
        inference_config = InferenceConfig()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Ensemble size: {len(checkpoint_paths)} models")

    alphabet = get_list_from_file(data_config.alphabet_path)
    collate_fn = get_collate_fn(model_config.max_seq_len)

    dataset = SpectrumDataset(
        root_dir=data_config.data_dir,
        alphabet_path=data_config.alphabet_path
    )
    print(f"Dataset size: {len(dataset)}")

    train_subset, val_subset, test_subset = generate_dataset_subsets(dataset, seed=42)

    random_train_indices = torch.randperm(len(train_subset))
    random_train_subset = Subset(train_subset, random_train_indices[:subset_size])

    train_loader = DataLoader(random_train_subset, batch_size=inference_config.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=inference_config.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=inference_config.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    models = load_ensemble_models(
        checkpoint_paths,
        token_size=dataset.token_dim,
        dict_size=dataset.ionDictN,
        model_config=model_config,
        device=device
    )

    os.makedirs(inference_config.output_dir, exist_ok=True)

    with torch.no_grad():
        train_file = os.path.join(inference_config.output_dir, "ensemble_train_predict.txt") if write_predictions else None
        val_file = os.path.join(inference_config.output_dir, "ensemble_val_predict.txt") if write_predictions else None
        test_file = os.path.join(inference_config.output_dir, "ensemble_test_predict.txt") if write_predictions else None

        print("\n--- Train Split ---")
        train_losses = predict_dataset_ensemble(models, train_loader, alphabet, device, inference_config, train_file)

        print("\n--- Validation Split ---")
        val_losses = predict_dataset_ensemble(models, val_loader, alphabet, device, inference_config, val_file)

        print("\n--- Test Split ---")
        test_losses = predict_dataset_ensemble(models, test_loader, alphabet, device, inference_config, test_file)

    return train_losses, val_losses, test_losses
