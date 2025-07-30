import sys
import os
import torch
from spectrum_dataset import SpectrumDataset, collate_fn
from torch.utils.data import DataLoader
from spectrum_model import SpectrumModel
from utils import threshold_tensor

file_path = ''
data_root_dir = os.path.join(file_path, "processed/")

max_seq_len = 20
batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SpectrumDataset(root_dir=data_root_dir, exclude_file=["human_hcd_tryp_best_processed"])
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

sequences, labels, charges, NCEs = next(iter(loader))

model = SpectrumModel(
    token_size=dataset.token_dim, dict_size=dataset.ionDictN, seq_max_len=max_seq_len
)
model.load_state_dict(torch.load("checkpoints/best_model_1.pth", map_location=torch.device('cpu')))
model.eval()

output = model(sequences, charges, NCEs)
output = threshold_tensor(output, 1e-5)

# Write the model output to a file, with each line containing the sequence and its corresponding output
output_file_path_with_seq = "inference_output_with_sequence.txt"
output_np = output.detach().cpu().numpy()
with open(output_file_path_with_seq, "w") as f:
    for seq, row in zip(dataset.sequenceList[:len(output_np)], output_np):
        f.write(seq + "\t" + "\n".join(map(str, row)) + "\n")
print(f"Model output with sequence written to {output_file_path_with_seq}")
