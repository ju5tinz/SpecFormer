import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import write_list_to_file, get_list_from_file

max_seq_len = 40

def collate_fn(batch):
    """
    Pads sequences and stacks batch data for DataLoader.
    """
    sequences, labels, charges, NCEs, mass_range_strs = zip(*batch)
    sequences = torch.stack([F.pad(s, (0, 0, 0, max_seq_len - len(s)), value=0.0) for s in sequences])
    labels = torch.tensor(labels, dtype=torch.float32)
    charges = torch.tensor(charges, dtype=torch.int32)
    NCEs = torch.tensor(NCEs, dtype=torch.float32)
    return sequences, labels, charges, NCEs, mass_range_strs

class SpectrumDataset(Dataset):
    """
    Efficient disk-backed dataset for spectrum data.
    Scans files to build index of byte-offsets. Reads sample data directly from disk.
    """
    def __init__(self, root_dir: str, alphabet_path: str, filenames=None, generate_alphabet: bool = False):
        if filenames is None:
            filenames = []
        self.sample_pointers = []
        self.aminoAcidAlphabetSet = set()
        self.ionDictN = 0

        for data_file in sorted(os.listdir(root_dir)):
            if data_file.startswith(".") or (filenames and data_file not in filenames):
                continue
            file_path = os.path.join(root_dir, data_file)
            with open(file_path, 'r') as f:
                print(f"Processing: {file_path}")
                # Get ionDictN from the first line, assuming it's consistent
                if self.ionDictN == 0 and (line := f.readline()):
                    self.ionDictN = int(line.split("=")[1].strip())
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.startswith(">"):
                        header = line.strip()
                        sequence = header.split("; ")[1].split("=")[1]
                        self.aminoAcidAlphabetSet.update(sequence)
                        self.sample_pointers.append({
                            'file_path': file_path,
                            'header': header,
                            'data_offset': f.tell()
                        })
        self.alphabet_list = []
        if generate_alphabet:
            self.alphabet_list = sorted(list(self.aminoAcidAlphabetSet))
            write_list_to_file(alphabet_path, self.alphabet_list)
        else:
            self.alphabet_list = sorted(get_list_from_file(alphabet_path))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet_list)}
        self.token_dim = len(self.alphabet_list)

    def __len__(self):
        return len(self.sample_pointers)

    def __getitem__(self, idx: int):
        pointer = self.sample_pointers[idx]
        header_parts = pointer['header'][1:].split('; ')
        sequence_str = header_parts[1].split('=')[1]
        charge = int(header_parts[2].split('=')[1])
        NCE = float(header_parts[3].split('=')[1])
        mass_range_str = header_parts[4].split('=')[1]
        spectrum_dict = {}
        with open(pointer['file_path'], 'r') as f:
            f.seek(pointer['data_offset'])
            for line in f:
                if line.startswith(">") or not line.strip():
                    break
                lineList = line.strip().split(",")
                spectrum_dict[int(lineList[0])] = float(lineList[1])
        label = self._spectrum_dict_to_list(spectrum_dict)
        sequence = self._one_hot_encode_sequence(sequence_str)
        return sequence, label, charge, NCE, mass_range_str

    def _spectrum_dict_to_list(self, d):
        result = [0.0] * self.ionDictN
        if not d:
            return result
        for k, v in d.items():
            if k < self.ionDictN:
                result[k] = v
        return result
    
    def _one_hot_encode_sequence(self, sequence: str):
        one_hot = torch.zeros((len(sequence), self.token_dim), dtype=torch.float32)
        for i, char in enumerate(sequence):
            if char in self.char_to_idx:
                one_hot[i, self.char_to_idx[char]] = 1
        return one_hot