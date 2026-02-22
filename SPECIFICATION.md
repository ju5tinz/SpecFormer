# SpecFormer Technical Specification

## Overview

SpecFormer is a deep learning system for **peptide spectrum prediction** - predicting mass spectrometry (MS/MS) fragmentation patterns from amino acid sequences. Given a peptide sequence, charge state, and normalized collision energy (NCE), the model predicts the intensity distribution across a dictionary of possible fragment ions.

**Domain**: Proteomics / Mass Spectrometry
**Framework**: PyTorch

---

## Project Structure

```
SpecFormer/
├── config.py                 # Configuration dataclasses
├── spectrum_model.py         # Neural network architecture
├── talking_head_attention.py # Custom attention mechanism
├── spectrum_dataset.py       # Dataset and data loading
├── training.py               # Training loop
├── inference.py              # Inference and ensemble prediction
├── utils.py                  # Utilities and loss functions
├── config/
│   └── amino_acid_alphabet.txt   # Amino acid vocabulary
├── processed/                # Input data files (.ann.txt)
├── checkpoints/              # Saved model weights (.pth)
└── predicted/                # Inference output files
```

---

## Data Format

### Input Data Files (`.ann.txt`)

Annotated spectrum files with the following structure:

```
Dictionary size = 26916
>length=22; sequence=AAAAAAAAAAAAGAGAGAK; charge=2; NCE=29; mass range=[109-1602]
4,35087
40,-1
46,239262
48,120991
...
>length=15; sequence=ACDEFGHIKLMNPQR; charge=3; NCE=25; mass range=[100-1500]
...
```

**File Header:**
- Line 1: `Dictionary size = N` - Total number of possible fragment ions

**Sample Format:**
- Header line: `>length=L; sequence=SEQ; charge=C; NCE=N; mass range=[min-max]`
  - `length`: Sequence length
  - `sequence`: Amino acid sequence (uppercase letters)
  - `charge`: Precursor charge state (integer)
  - `NCE`: Normalized collision energy (float)
  - `mass range`: m/z range of the spectrum
- Data lines: `ion_index,intensity`
  - `ion_index`: Index into the ion dictionary (0 to dict_size-1)
  - `intensity`: Measured intensity value, or `-1` for missing/masked ions

### Alphabet File

One amino acid per line (sorted):
```
A
C
D
...
Y
```

Standard alphabet contains 22 characters (20 standard amino acids + O, U for selenocysteine/pyrrolysine).

---

## Configuration

All configuration is centralized in `config.py` using dataclasses.

### ModelConfig

Architecture hyperparameters for `SpectrumModel`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | int | 256 | Embedding dimension |
| `num_heads` | int | 64 | Number of attention heads |
| `max_seq_len` | int | 40 | Maximum sequence length (for positional embeddings) |
| `penultimate_dim` | int | 2048 | Penultimate layer dimension |
| `dropout_rate` | float | 0.2 | Dropout probability |

### TrainConfig

Training hyperparameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 512 | Training batch size |
| `learning_rate` | float | 1e-4 | AdamW learning rate |
| `weight_decay` | float | 1e-3 | AdamW weight decay |
| `epochs` | int | 100 | Maximum training epochs |
| `patience` | int | 5 | Early stopping patience |
| `checkpoint_path` | str | "checkpoints/best_model.pth" | Model save path |
| `num_workers` | int | 4 | DataLoader workers |

### InferenceConfig

Inference settings:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 16 | Inference batch size |
| `threshold` | float | 1e-4 | Intensity threshold (values below set to 0) |
| `output_dir` | str | "predicted/" | Output directory |

### DataConfig

Data paths and file selection:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | "processed/" | Directory containing .ann.txt files |
| `alphabet_path` | str | "config/amino_acid_alphabet.txt" | Path to alphabet file |
| `train_files` | list | ["AItrain_LumosSynthetic_2022418v2.ann.txt", "AItrain_QEHumanCho_2022418v2.ann.txt"] | Training files |
| `val_files` | list | ["ValidUniq2022418_202333.ann.txt"] | Validation files |

**Note:** When both `train_files` and `val_files` are empty lists, the system falls back to auto-split mode (96%/2%/2%).

---

## Model Architecture

### SpectrumModel

A transformer-based architecture using talking-head attention for spectrum prediction.

**Input:**
- `x`: `(batch_size, seq_len, token_size)` - One-hot encoded sequences
- `charges`: `(batch_size,)` - Charge states
- `NCEs`: `(batch_size,)` - Normalized collision energies

**Output:**
- `(batch_size, dict_size)` - Predicted ion intensities (sigmoid-activated, 0-1 range)

**Architecture Flow:**

```
Input Sequence (one-hot) ──► Linear Projection ──► + Global Embedding ──► + Positional Embedding
                                                        │
                                                        ▼
                                                    Dropout
                                                        │
                                                        ▼
                                                  LayerNorm
                                                        │
                                                        ▼
                                          ┌─── Talking-Head Attention ───┐
                                          │                              │
                                          ▼                              │
                                      Dropout ◄──────────────────────────┘
                                          │
                                          ▼
                                   Residual Add
                                          │
                                          ▼
                                     LayerNorm
                                          │
                                          ▼
                            ┌──── FFN Block (2 Linear + ReLU) ────┐
                            │                                     │
                            ▼                                     │
                        Dropout ◄─────────────────────────────────┘
                            │
                            ▼
                     Residual Add
                            │
                            ▼
                       LayerNorm
                            │
                            ▼
                  Penultimate (Linear + LayerNorm + ReLU + Dropout)
                            │
                            ▼
                    Output Projection
                            │
                            ▼
                        Sigmoid
                            │
                            ▼
                    Mean over seq_len
                            │
                            ▼
                   (batch_size, dict_size)
```

**Key Components:**

1. **Global Embedding**: Charge and NCE are projected and added to all positions
2. **Positional Embedding**: Learned embeddings for sequence positions
3. **Talking-Head Attention**: Attention with head-wise linear transformations pre/post softmax
4. **FFN Block**: Two-layer feedforward with ReLU activation
5. **Output**: Mean pooling across sequence positions, then sigmoid

### TalkingHeadAttention

Custom multi-head attention with "talking heads" - learned linear projections across attention heads before and after softmax.

**Parameters:**
- `embed_dim`: Must be divisible by `num_heads`
- `num_heads`: Number of attention heads
- `head_dim`: `embed_dim // num_heads`

**Key Difference from Standard Attention:**
- Pre-softmax: Linear projection across heads on attention scores
- Post-softmax: Linear projection across heads on attention weights

This allows heads to share information during attention computation.

---

## Dataset

### SpectrumDataset

Disk-backed PyTorch Dataset that efficiently handles large spectrum files.

**Initialization:**
```python
dataset = SpectrumDataset(
    root_dir="processed/",
    alphabet_path="config/amino_acid_alphabet.txt",
    filenames=["file1.ann.txt", "file2.ann.txt"],  # Optional filter
    generate_alphabet=False  # Set True to auto-generate alphabet
)
```

**Attributes:**
- `token_dim`: Size of one-hot encoding (alphabet size)
- `ionDictN`: Dictionary size (number of possible ions)
- `sample_pointers`: List of file offsets for efficient random access

**Returns per sample:**
- `sequence`: `(seq_len, token_dim)` tensor - One-hot encoded sequence
- `label`: `list[float]` - Ion intensities (length = ionDictN)
- `charge`: `int` - Charge state
- `NCE`: `float` - Normalized collision energy
- `mass_range_str`: `str` - Mass range string

### get_collate_fn

Factory function for DataLoader collation with configurable sequence padding:

```python
collate_fn = get_collate_fn(max_seq_len=40)
loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)
```

**Collation:**
- Pads sequences to `max_seq_len` with zeros
- Stacks labels into `(batch_size, dict_size)` tensor
- Converts charges/NCEs to tensors

---

## Loss Function

### FilteredCosineLoss

Cosine similarity loss that handles masked labels (value -1 indicates missing data).

```python
criterion = FilteredCosineLoss(reduction='mean')  # 'mean', 'sum', or 'none'
loss = criterion(outputs, labels)
```

**Computation:**
1. For each sample, create mask where `labels != -1`
2. Compute cosine loss on masked elements only: `1 - cosine_similarity(pred, target)`
3. Apply reduction across batch

**Reduction modes:**
- `'mean'`: Return scalar (mean of per-sample losses) - default
- `'sum'`: Return scalar (sum of per-sample losses)
- `'none'`: Return list of per-sample loss tensors

---

## Training API

### train()

Main training function with early stopping and checkpointing.

```python
from config import DataConfig, ModelConfig, TrainConfig
from training import train

# Use defaults (includes predefined train/val files)
train(DataConfig())

# Or customize
data_config = DataConfig(
    data_dir='processed/',
    train_files=['custom_train.ann.txt'],
    val_files=['custom_val.ann.txt']
)
model_config = ModelConfig(embed_dim=256, num_heads=64)
train_config = TrainConfig(epochs=100, patience=5, checkpoint_path='checkpoints/my_model.pth')

train(data_config, model_config, train_config)
```

**Data Loading Modes:**

1. **Explicit split** (when `train_files` and `val_files` are specified):
   - Loads separate datasets for training and validation

2. **Auto-split** (when `train_files` and `val_files` are empty):
   - Loads all files, splits 96%/2%/2% (train/val/test)
   - Uses seed=42 for reproducibility

**Training Loop:**
- Optimizer: AdamW
- Loss: FilteredCosineLoss
- Early stopping: Stops if validation loss doesn't improve for `patience` epochs
- Checkpointing: Saves best model (lowest validation loss)

---

## Inference API

### run_inference()

Run ensemble inference on specified files.

```python
from config import DataConfig, InferenceConfig
from inference import run_inference

data_config = DataConfig(data_dir='processed/')
inference_config = InferenceConfig(batch_size=16, threshold=1e-4, output_dir='predicted/')

results = run_inference(
    checkpoint_paths=['checkpoints/model1.pth', 'checkpoints/model2.pth'],
    data_config=data_config,
    inference_config=inference_config,
    files=['test1.ann.txt', 'test2.ann.txt'],  # Optional, default=all files
    write_predictions=True
)
# results: {'test1.ann.txt': [loss1, loss2, ...], 'test2.ann.txt': [...]}
```

### run_inference_splits()

Run inference on auto-generated train/val/test splits.

```python
train_losses, val_losses, test_losses = run_inference_splits(
    checkpoint_paths=['model1.pth', 'model2.pth'],
    data_config=data_config,
    subset_size=60000  # Limit train set samples for efficiency
)
```

**Ensemble Prediction:**
- Multiple models loaded from checkpoints
- Predictions averaged across all models
- Intensities below threshold set to 0

**Output Format:**
Predictions written in same format as input:
```
>length=22; sequence=AAAAAAAAAA...; charge=2; NCE=29; mass range=[109-1602]
4,0.035087
46,0.239262
...
```

---

## Utility Functions

### get_device()

Auto-detect best available device:
```python
device = get_device()  # Returns 'cuda', 'mps', or 'cpu'
```

### generate_dataset_subsets()

Split dataset into train/val/test (96%/2%/2%):
```python
train_subset, val_subset, test_subset = generate_dataset_subsets(dataset, seed=42)
```

### threshold_tensor()

Zero out values below threshold:
```python
output = threshold_tensor(predictions, threshold=1e-4)
```

---

## Usage Examples

### Training with Default Configuration

```python
from config import DataConfig
from training import train

# Use all defaults (includes default train/val files)
train(DataConfig())
```

### Training with Custom Configuration

```python
from config import DataConfig, ModelConfig, TrainConfig
from training import train

# Custom model architecture
model_config = ModelConfig(
    embed_dim=512,
    num_heads=32,
    penultimate_dim=2048,
    dropout_rate=0.1
)

# Custom training settings
train_config = TrainConfig(
    batch_size=256,
    learning_rate=5e-5,
    epochs=200,
    patience=10,
    checkpoint_path='checkpoints/large_model.pth'
)

# Custom data files
data_config = DataConfig(
    data_dir='processed/',
    train_files=['human_hcd_tryp_best.ann.txt'],
    val_files=['human_hcd_tryp_good.ann.txt']
)

train(data_config, model_config, train_config)
```

### Ensemble Inference

```python
from config import DataConfig, InferenceConfig
from inference import run_inference

checkpoints = [
    'checkpoints/model_fold1.pth',
    'checkpoints/model_fold2.pth',
    'checkpoints/model_fold3.pth'
]

data_config = DataConfig(data_dir='processed/')
inference_config = InferenceConfig(
    batch_size=32,
    threshold=1e-5,
    output_dir='predictions/'
)

results = run_inference(
    checkpoint_paths=checkpoints,
    data_config=data_config,
    inference_config=inference_config,
    files=['test_data.ann.txt']
)

# Analyze results
for filename, losses in results.items():
    print(f"{filename}: mean loss = {sum(losses)/len(losses):.6f}")
```

### Fine-tuning from Pretrained Weights

```python
from config import DataConfig
from training import train

data_config = DataConfig(
    data_dir='new_data/',
    train_files=['new_train.ann.txt'],
    val_files=['new_val.ann.txt']
)

train(
    data_config=data_config,
    weights_file='checkpoints/pretrained_model.pth'
)
```

---

## Model Dimensions Reference

For default configuration (`token_size=22`, `dict_size=26916`, `num_heads=64`, `penultimate_dim=2048`):

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| pre_attn_proj | (B, L, 22) | (B, L, 256) | 5,888 |
| global_data_projector | (B, 2) | (B, 256) | 768 |
| pos_embedding | L | (L, 256) | 10,240 |
| attention (TalkingHead) | (B, L, 256) | (B, L, 256) | 263,680 |
| post_attn_FFN1 | (B, L, 256) | (B, L, 256) | 65,792 |
| post_attn_FFN2 | (B, L, 256) | (B, L, 256) | 65,792 |
| penultimate_proj | (B, L, 256) | (B, L, 2048) | 526,336 |
| output_proj | (B, L, 2048) | (B, L, 26916) | 55,125,684 |
| **Total** | | | **~56M** |

---

## Hardware Requirements

- **Training**: GPU recommended (CUDA or Apple MPS)
- **Inference**: CPU viable for small batches; GPU for throughput
- **Memory**: ~2-4GB GPU memory for default batch sizes
