import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

max_seq_len = 20

def collate_fn(batch):
    # len(batch) is batch size
    # batch is a list of tuples (sequence, label)
    # sequence has shape (sequence_length, feature_size)
    # label is 1D tensor
    sequences, labels, charges, NCEs = zip(*batch)
    sequences = torch.stack([F.pad(s, (0, 0, 0, max_seq_len - len(s)), value=0.0) for s in sequences])
    labels = torch.tensor(labels, dtype=torch.float32)
    charges = torch.tensor(charges, dtype=torch.int32)
    NCEs = torch.tensor(NCEs, dtype=torch.float32)
    return sequences, labels, charges, NCEs

class SpectrumDataset(Dataset):
    def __init__(self, root_dir, exclude_file=[]):
        self.sequenceList = []
        self.chargeList = []
        self.NCEList = []
        self.SpectrumDictList = []
        self.aminoAcidAlphabet = set()
        currIdx = -1
        self.ionDictN = 0
        
        for data_file in os.listdir(root_dir):
            if data_file.startswith("."):
                continue
            if data_file in exclude_file:
                continue
            with open(os.path.join(root_dir, data_file), 'r') as f:
                print("processing: ", f)
                self.ionDictN = int(f.readline().split("=")[1].strip())
                for line in f:
                    if line.startswith(">"):
                        currIdx += 1
                        sequence = line[1:].split("; ")[1].split("=")[1]
                        self.aminoAcidAlphabet.update(sequence)
                        self.sequenceList.append(sequence)
                        self.chargeList.append(int(line[1:].split("; ")[2].split("=")[1]))
                        self.NCEList.append(float(line[1:].split("; ")[3].split("=")[1]))
                        self.SpectrumDictList.append({})
                    else:
                        lineList = line.split(",")
                        self.SpectrumDictList[currIdx][int(lineList[0])] = float(lineList[1])

        self.token_dim = len(self.aminoAcidAlphabet)
    
    def __len__(self):
        return len(self.sequenceList)
    
    def __getitem__(self, idx):
        label = self.spectrum_dict_to_list(self.SpectrumDictList[idx])
        sequence = self.one_hot_encode_sequence(self.sequenceList[idx])
        charge = torch.tensor(self.chargeList[idx], dtype=torch.int32)
        NCE = torch.tensor(self.NCEList[idx], dtype=torch.float32)

        return sequence, label, charge, NCE

    def spectrum_dict_to_list(self, d):
        if not d:
            print("no data")
            return []
        result = [0] * self.ionDictN
        for k, v in d.items():
            result[k] = v
        return result
    
    def one_hot_encode_sequence(self, sequence):
        alphabet = sorted(self.aminoAcidAlphabet)
        char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
        one_hot = torch.zeros((len(sequence), len(alphabet)), dtype=torch.float32)
        for i, char in enumerate(sequence):
            one_hot[i, char_to_idx[char]] = 1
        return one_hot
