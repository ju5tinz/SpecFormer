# SpecFormer

Transformer-based deep learning system for **peptide MS/MS spectrum prediction**. Given a peptide sequence, charge state, and normalized collision energy (NCE), predicts fragment ion intensity distributions using talking-head attention.

## Quick Start

### Training

```python
from config import DataConfig
from training import train

# Train with defaults
train(DataConfig())

# Or customize
from config import ModelConfig, TrainConfig
train(
    DataConfig(train_files=["my_train.ann.txt"], val_files=["my_val.ann.txt"]),
    ModelConfig(embed_dim=512, num_heads=32),
    TrainConfig(epochs=200, patience=10),
)
```

### Ensemble Inference

```python
from config import DataConfig, InferenceConfig
from inference import run_inference

results = run_inference(
    checkpoint_paths=["checkpoints/ensemble_1.pth", "checkpoints/ensemble_2.pth"],
    data_config=DataConfig(data_dir="processed/"),
    files=["TestCom2022418_202336.ann.txt"],
)
```

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | Configuration dataclasses (`ModelConfig`, `TrainConfig`, `InferenceConfig`, `DataConfig`) |
| `spectrum_model.py` | Transformer model with talking-head attention |
| `talking_head_attention.py` | Custom multi-head attention with inter-head projections |
| `spectrum_dataset.py` | Dataset and collation for `.ann.txt` spectrum files |
| `training.py` | Training loop with early stopping and checkpointing |
| `inference.py` | Ensemble inference and prediction output |
| `utils.py` | Device detection, loss functions, dataset splitting |
