# dataset_gait.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class GaitDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.sequences = data["sequences"]  # массив объектов (list of arrays)
        self.labels_raw = data["labels"]    # массив строк
        # маппим person_id -> int label
        self.label2idx = {lab: i for i, lab in enumerate(sorted(set(self.labels_raw)))}
        self.idx2label = {i: lab for lab, i in self.label2idx.items()}
        self.labels = np.array([self.label2idx[lab] for lab in self.labels_raw], dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].astype(np.float32)  # (T_i, 41)
        label = self.labels[idx]
        return seq, label

def collate_fn(batch):
    # batch: list of (seq, label), где seq: (T_i, 41)
    sequences, labels = zip(*batch)
    lengths = [s.shape[0] for s in sequences]
    max_len = max(lengths)

    padded = []
    for s in sequences:
        T, D = s.shape
        if T < max_len:
            pad = np.zeros((max_len - T, D), dtype=np.float32)
            s_pad = np.concatenate([s, pad], axis=0)
        else:
            s_pad = s
        padded.append(s_pad)

    x = torch.from_numpy(np.stack(padded, axis=0))  # (B, max_len, 41)
    y = torch.from_numpy(np.array(labels, dtype=np.int64))
    return x, y, torch.tensor(lengths, dtype=torch.int64)