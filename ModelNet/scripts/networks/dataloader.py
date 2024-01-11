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

        self.x_sequences = []
        self.y_sequences = []
        self.edges = []

        for dirpath, _, filenames in os.walk(self.dir_path):
            for filename in filenames:
                try:
                    with open(f'{self.dir_path}/{filename}', 'rb') as f:
                        data = pickle.load(f)
                        self.x_sequences.extend(data['x_sequences'])
                        self.y_sequences.extend(data['y_sequences'])
                        self.edges.extend(data['edges'])
                except:
                    print(f'{self.dir_path}/{filename}')

        self.data_length = len(self.x_sequences)
        print(f'{data_type}: {self.data_length}')

    def __getitem__(self, idx):
        x_sequences = self.x_sequences[idx]
        y_sequences = self.y_sequences[idx]
        edges = self.edges[idx]

        x_sequences = torch.tensor(x_sequences, dtype=torch.float32)
        y_sequences = torch.tensor(y_sequences, dtype=torch.float32)
        edges = torch.tensor(edges, dtype=torch.long)

        return x_sequences, y_sequences, edges

    def __len__(self):
        return self.data_length