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
threshold = 1e-4

def one_hot_to_seq(one_hot_seq, alphabet):
    alphabet = sorted(alphabet)
    is_token_mask = torch.any(one_hot_seq, dim=1)
    actual_length = torch.sum(is_token_mask)
    indices = torch.argmax(one_hot_seq[:actual_length], dim=-1)
    seq = ''.join([alphabet[i] for i in indices])
    return seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SpectrumDataset(root_dir=data_root_dir, exclude_file=["human_hcd_tryp_best_processed"])
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = SpectrumModel(
    token_size=dataset.token_dim, dict_size=dataset.ionDictN, seq_max_len=max_seq_len
)
model.load_state_dict(torch.load("checkpoints/best_model_1.pth", map_location=torch.device('cpu')))
model.eval()

output_file_path_with_seq = "inference_output_with_sequence.txt"
with open(output_file_path_with_seq, "w") as f:
    for (sequences, _, charges, NCEs) in loader:
        output = model(sequences, charges, NCEs)
        output = threshold_tensor(output, threshold)
        for i in range(sequences.shape[0]):
            seq = one_hot_to_seq(sequences[i], dataset.aminoAcidAlphabet)
            f.write(f'>length={len(seq)}; sequence={seq}; charge={charges[i]}; NCE={int(NCEs[i])}\n')
            for row_i, row in enumerate(output[i]):
                if row != 0:
                    f.write(f'{row_i},{row}\n')