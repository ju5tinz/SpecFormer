import torch
from spectrum_model import SpectrumModel
from spectrum_dataset import SpectrumDataset, collate_fn
from torch.utils.data import DataLoader

def export_amino_acid_alphabet(amino_acid_alphabet, file_path):
    alphabet = sorted(amino_acid_alphabet)
    with open(file_path, "w") as f:
        for char in alphabet:
            f.write(char + "\n")

max_seq_len = 20

dataset = SpectrumDataset(root_dir="processed")
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

sequences, labels, charges, NCEs = next(iter(data_loader))

model = SpectrumModel(
    token_size=dataset.token_dim, dict_size=dataset.ionDictN, seq_max_len=max_seq_len
)
model.load_state_dict(torch.load("checkpoints/best_model_1.pth", map_location=torch.device('cpu')))
model.eval()

input = (sequences, charges, NCEs)

onnx_file_path = "onnx/best_model_1.onnx"
torch.onnx.export(
    model,
    input,
    onnx_file_path,
    input_names=["sequences", "charges", "NCEs"],
    output_names=["output"],
    dynamic_axes={
        "sequences": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=12,
    export_params=True,
)

print(f"Model successfully exported to {onnx_file_path}")

export_amino_acid_alphabet(dataset.aminoAcidAlphabet, "onnx/amino_acid_alphabet.txt")

print(f"Amino acid alphabet successfully exported to {onnx_file_path}")