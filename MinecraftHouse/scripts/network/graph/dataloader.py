import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import numpy as np
import pickle
import os


class GraphDataset(Dataset):
    def __init__(self, data_type='train', transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(transform, pre_transform)

        self.data_type = data_type

        self.file_path = f'../../../datasets/preprocessed/{data_type}_dataset.pkl'
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)

        self.local_grids = data['local_grids']
        self.node_lists = data['node_lists']
        self.edge_lists = data['edge_lists']
        self.gt_grids = data['gt_grids']
        self.gt_ids = data['gt_ids']

        self.data_length = len(self.gt_ids)
        print(f'data_length: {self.data_length}')
        print(f'node_lists: {len(self.node_lists)}')
        print(f'edge_lists: {len(self.edge_lists)}')

    def get(self, idx):
        local_grid = self.local_grids[idx]
        node_list = self.node_lists[idx]
        edge_index = self.edge_lists[idx]
        gt_grid = self.gt_grids[idx]
        gt_id = self.gt_ids[idx]

        local_grid = torch.tensor(local_grid, dtype=torch.long)
        position_feature = torch.tensor(np.array(node_list)[:, :3], dtype=torch.float32)
        id_feature = torch.tensor(np.array(node_list)[:, 3:], dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        gt_grid = torch.tensor(gt_grid, dtype=torch.long)
        gt_id = torch.tensor(gt_id, dtype=torch.long)

        data = Data(local_grid=local_grid, position_feature=position_feature, id_feature=id_feature,
                    edge_index=edge_index, gt_grid=gt_grid, gt_id=gt_id, num_nodes=len(node_list))
        return data

    def len(self):
        return self.data_length
