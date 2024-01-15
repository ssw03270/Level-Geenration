import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from tqdm import tqdm

class ModelNetDataset(Dataset):
    def __init__(self, data_type='train'):
        super(ModelNetDataset, self).__init__()

        self.data_type = data_type

        if data_type == 'train':
            self.dir_path = '../preprocessing/processed_voxels/train'
        else:
            self.dir_path = '../preprocessing/processed_voxels/test'

        self.parent_sequences = []
        self.child_sequences = []
        self.dir_sequences = []

        for dirpath, _, filenames in os.walk(self.dir_path):
            for filename in filenames:
                try:
                    with open(f'{self.dir_path}/{filename}', 'rb') as f:
                        data = pickle.load(f)
                        self.parent_sequences.append(data['parent_sequences'])
                        self.child_sequences.append(data['child_sequences'])
                        self.dir_sequences.append(data['dir_sequences'])
                except:
                    print(f'{self.dir_path}/{filename}')

        self.data_length = len(self.parent_sequences)
        print(f'{data_type}: {self.data_length}')

    def __getitem__(self, idx):
        parent_sequences = self.parent_sequences[idx]
        child_sequences = self.child_sequences[idx]
        dir_sequences = self.dir_sequences[idx]

        max_voxel = 256
        pad_mask = np.ones(max_voxel)
        pad_mask[max_voxel - 1] = 0

        parent_sequences = [max_voxel - 1] + parent_sequences
        child_sequences = [[0, 0, 0]] + child_sequences
        dir_sequences = [[0, 0, 0, 0, 0, 0]] + dir_sequences

        for i in range(max_voxel - len(parent_sequences)):
            parent_sequences.append(max_voxel - 2)
            child_sequences.append([0, 0, 0])
            dir_sequences.append([0, 0, 0, 0, 0, 0])
            pad_mask[max_voxel - i - 1] = 0

        parent_sequences = torch.tensor(parent_sequences, dtype=torch.long)
        child_sequences = torch.tensor(child_sequences, dtype=torch.float32)
        dir_sequences = torch.tensor(dir_sequences, dtype=torch.long)
        pad_mask = torch.tensor(pad_mask, dtype=torch.bool)

        return parent_sequences, child_sequences, dir_sequences, pad_mask

    def __len__(self):
        return self.data_length