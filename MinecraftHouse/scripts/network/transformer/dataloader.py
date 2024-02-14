import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pickle
import os

class HouseDataset(Dataset):
    def __init__(self, data_type='train'):
        super(HouseDataset, self)

        self.data_type = data_type

        self.root_path = f'/local_datasets/house_datasets/{data_type}'
        # self.root_path = f'../../../datasets/preprocessed/house_datasets/{data_type}'

        self.filenames = []
        for filename in os.listdir(self.root_path):
            self.filenames.append(f'{self.root_path}/{filename}')

        self.data_length = len(self.filenames)
        print(f'data_length: {self.data_length}')

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        local_grid = np.array(data['local_grid'])
        node_list = data['node_list']
        gt_grid = data['gt_grid']
        gt_id = data['gt_id']

        position_feature = np.array(node_list)[:, :3]
        id_feature = np.array(node_list)[:, 3:].reshape(-1)

        return {
            'local_grid': local_grid,
            'position_feature': position_feature,
            'id_feature': id_feature,
            'gt_grid': gt_grid,
            'gt_id': gt_id
        }

    def __len__(self):
        return self.data_length
