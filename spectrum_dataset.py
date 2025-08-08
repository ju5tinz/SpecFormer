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
    def __init__(self, root_dir, exclude_file=[]):
        """
        Initializes the dataset by scanning files to create an index (pointers),
        but does NOT load the spectrum data into memory.
        """
        self.root_dir = root_dir
        self.sample_pointers = []
        self.aminoAcidAlphabet = set()
        self.ionDictN = None

        print("Building dataset index...")
        for data_file in os.listdir(root_dir):
            if data_file.startswith(".") or data_file in exclude_file:
                continue
            
            file_path = os.path.join(root_dir, data_file)
            with open(file_path, 'r') as f:
                print("processing: ", f)
                # Read header to get ionDictN, assuming it's consistent
                if self.ionDictN == None:
                    self.ionDictN = int(f.readline().split("=")[1].strip())

                # Build the index
                current_spectrum_lines = []
                header_info = None
                for line in f:
                    if line.startswith(">"):
                        # If we have a pending sample, save it
                        if header_info:
                            self.sample_pointers.append({
                                'header': header_info,
                                'spectrum_lines': current_spectrum_lines.copy()
                            })
                        
                        # Start a new sample
                        header_info = line.strip()
                        current_spectrum_lines.clear()
                        
                        # Also update the alphabet from the sequence in the header
                        sequence = header_info.split("; ")[1].split("=")[1]
                        self.aminoAcidAlphabet.update(sequence)
                    elif header_info: # Only add lines if they belong to a sample
                        current_spectrum_lines.append(line.strip())

                # Add the last sample in the file
                if header_info:
                    self.sample_pointers.append({
                        'header': header_info,
                        'spectrum_lines': current_spectrum_lines.copy()
                    })

        self.alphabet_list = sorted(list(self.aminoAcidAlphabet))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet_list)}
        self.token_dim = len(self.alphabet_list)
        print(f"Indexing complete. Found {len(self.sample_pointers)} samples.")
    
    def __len__(self):
        return len(self.sample_pointers)
    
    def __getitem__(self, idx):
        """
        This method loads and processes ONE sample on-demand.
        """
        # 1. Get the pointer to the data for the requested index
        pointer = self.sample_pointers[idx]
        
        # 2. Parse the metadata from the stored header string
        header_parts = pointer['header'][1:].split('; ')
        sequence_str = header_parts[1].split('=')[1]
        charge = int(header_parts[2].split('=')[1])
        NCE = float(header_parts[3].split('=')[1])
        
        # 3. Process the spectrum data (which is just a list of strings)
        spectrum_dict = {}
        for line in pointer['spectrum_lines']:
            lineList = line.split(",")
            spectrum_dict[int(lineList[0])] = float(lineList[1])
        
        label = self.spectrum_dict_to_list(spectrum_dict)
        
        # 4. One-hot encode the sequence
        sequence = self.one_hot_encode_sequence(sequence_str)

        # Note: We return python lists/ints/floats and let the collate_fn handle tensor conversion
        return sequence, label, charge, NCE

    def spectrum_dict_to_list(self, d):
        result = [0.0] * self.ionDictN
        if not d:
            return result
        for k, v in d.items():
            if k < self.ionDictN:
                result[k] = v
        return result
    
    def one_hot_encode_sequence(self, sequence):
        one_hot = torch.zeros((len(sequence), self.token_dim), dtype=torch.float32)
        for i, char in enumerate(sequence):
            if char in self.char_to_idx:
                one_hot[i, self.char_to_idx[char]] = 1
        return one_hot