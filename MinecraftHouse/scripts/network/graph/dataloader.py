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

        self.root_path = f'/local_datasets/house_datasets/{data_type}'
        # self.root_path = f'../../../datasets/preprocessed/house_datasets/{data_type}'

        self.filenames = []
        for filename in os.listdir(self.root_path):
            self.filenames.append(f'{self.root_path}/{filename}')

        self.data_length = len(self.filenames)
        print(f'data_length: {self.data_length}')

    def get(self, idx):
        file_name = self.filenames[idx]
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        local_grid = data['local_grid']
        node_list = data['node_list']
        edge_index = data['edge_list']
        gt_grid = data['gt_grid']
        gt_id = data['gt_id']

        local_grid = torch.tensor(local_grid, dtype=torch.long)
        position_feature = torch.tensor(np.array(node_list)[:, :3], dtype=torch.float32)
        id_feature = torch.tensor(np.array(node_list)[:, 3:], dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        gt_grid = torch.tensor(gt_grid, dtype=torch.long)
        gt_id = torch.tensor(gt_id, dtype=torch.long)
        each_num_nodes = torch.tensor([len(node_list)], dtype=torch.long)
        temporal_edge_index = torch.tensor(
            np.concatenate(
                (
                    [np.arange(len(node_list) - 1), np.arange(1, len(node_list))],  # 이전 -> 다음 노드 엣지
                    [np.arange(len(node_list)), np.arange(len(node_list))]  # 자기 자신으로의 엣지
                ),
                axis=1
            ),
            dtype=torch.long
        )
        differences = torch.diff(position_feature, dim=0)
        direction_feature = torch.cat((torch.zeros(1, 3, dtype=torch.float32), differences), dim=0)

        data = Data(local_grid=local_grid, position_feature=position_feature, id_feature=id_feature,
                    edge_index=edge_index, gt_grid=gt_grid, gt_id=gt_id, num_nodes=len(node_list),
                    each_num_nodes=each_num_nodes, temporal_edge_index=temporal_edge_index,
                    direction_feature=direction_feature)
        return data

    def len(self):
        return self.data_length
