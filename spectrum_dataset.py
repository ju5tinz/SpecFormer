import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

max_seq_len = 40

def collate_fn(batch):
    sequences, labels, charges, NCEs = zip(*batch)
    sequences = torch.stack([F.pad(s, (0, 0, 0, max_seq_len - len(s)), value=0.0) for s in sequences])
    labels = torch.tensor(labels, dtype=torch.float32)
    charges = torch.tensor(charges, dtype=torch.int32)
    NCEs = torch.tensor(NCEs, dtype=torch.float32)
    return sequences, labels, charges, NCEs

class SpectrumDataset(Dataset):
    """
    This Dataset implementation is designed for extreme memory efficiency.
    __init__: Scans files to build a tiny index of byte-offsets.
    __getitem__: Reads data for a single item directly from disk using the offset.
    """
    def __init__(self, root_dir, exclude_file=[]):
        self.sample_pointers = []
        self.aminoAcidAlphabet = set()
        self.ionDictN = 0

        for data_file in os.listdir(root_dir):
            if data_file.startswith(".") or data_file in exclude_file:
                continue
            
            file_path = os.path.join(root_dir, data_file)
            with open(file_path, 'r') as f:
                print("processing: ", f)
                # Get ionDictN from the first line, assuming it's consistent
                if self.ionDictN == 0 and (line := f.readline()):
                    self.ionDictN = int(line.split("=")[1].strip())

                while True:
                    line = f.readline()
                    if not line:
                        break # End of file

                    if line.startswith(">"):
                        header = line.strip()
                        # Get sequence from header to build the alphabet
                        sequence = header.split("; ")[1].split("=")[1]
                        self.aminoAcidAlphabet.update(sequence)
                        
                        # The pointer stores:
                        # 1. The path to the file.
                        # 2. The header string (small metadata).
                        # 3. The byte offset of the *next* line, where the spectrum data begins.
                        self.sample_pointers.append({
                            'file_path': file_path,
                            'header': header,
                            'data_offset': f.tell() 
                        })

        # Create mapping from alphabet for one-hot encoding
        self.alphabet_list = sorted(list(self.aminoAcidAlphabet))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet_list)}
        self.token_dim = len(self.alphabet_list)

    def __len__(self):
        return len(self.sample_pointers)

    def __getitem__(self, idx):
        # 1. Get the lightweight pointer for the requested sample
        pointer = self.sample_pointers[idx]
        
        # 2. Parse metadata directly from the stored header string. No disk I/O needed for this.
        header_parts = pointer['header'][1:].split('; ')
        sequence_str = header_parts[1].split('=')[1]
        charge = int(header_parts[2].split('=')[1])
        NCE = float(header_parts[3].split('=')[1])
        
        # 3. Read ONLY this sample's spectrum data directly from the file on disk.
        spectrum_dict = {}
        with open(pointer['file_path'], 'r') as f:
            f.seek(pointer['data_offset']) # JUMP to the correct position
            for line in f:
                # Stop reading if we hit the next sample's header or an empty line
                if line.startswith(">") or not line.strip():
                    break
                
                lineList = line.strip().split(",")
                spectrum_dict[int(lineList[0])] = float(lineList[1])
            
        # 4. Process the data into tensors
        label = self._spectrum_dict_to_list(spectrum_dict)
        sequence = self._one_hot_encode_sequence(sequence_str)

        return sequence, label, charge, NCE

    def _spectrum_dict_to_list(self, d):
        result = [0.0] * self.ionDictN
        if not d:
            return result
        for k, v in d.items():
            if k < self.ionDictN:
                result[k] = v
        return result
    
    def _one_hot_encode_sequence(self, sequence):
        one_hot = torch.zeros((len(sequence), self.token_dim), dtype=torch.float32)
        for i, char in enumerate(sequence):
            if char in self.char_to_idx:
                one_hot[i, self.char_to_idx[char]] = 1
        return one_hot