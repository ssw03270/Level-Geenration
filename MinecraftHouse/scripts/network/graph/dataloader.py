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

        self.folder_path = '../../../datasets/preprocessed_graph/graph_gpickle'
        # self.folder_path = '/local_datasets/graph_gpickle'

        all_file_paths = []
        for filename in tqdm(os.listdir(self.folder_path)):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(file_path):
                all_file_paths.append(file_path)
        all_file_paths.sort()

        total_length = len(all_file_paths)
        train_length = int(total_length * 0.8)
        validation_length = int(total_length * 0.1)

        if self.data_type == 'train':
            self.file_paths = all_file_paths[:train_length]
        elif self.data_type == 'validation':
            self.file_paths = all_file_paths[train_length:train_length + validation_length]
        elif self.data_type == 'test':
            self.file_paths = all_file_paths[train_length + validation_length:]

        self.data_length = len(self.file_paths)
        print(f'data_length: {self.data_length}')

    def get(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)

        position_feature = torch.tensor(np.array([graph.nodes[node]['position'] for node in graph.nodes()]), dtype=torch.float32)
        id_feature = torch.tensor(np.array([graph.nodes[node]['id'] for node in graph.nodes()]), dtype=torch.long)
        category_feature = torch.tensor(np.array([graph.nodes[node]['category'] for node in graph.nodes()]), dtype=torch.long)
        idx_feature = torch.tensor(np.array([graph.nodes[node]['idx'] for node in graph.nodes()]), dtype=torch.long)

        next_position = torch.tensor(np.array(graph.graph['gt']['next_position']), dtype=torch.float32)
        next_id = torch.tensor(np.array(graph.graph['gt']['next_id']), dtype=torch.long)
        next_category = torch.tensor(np.array(graph.graph['gt']['next_category']), dtype=torch.long)

        edge_index = []
        edge_attr = []

        for (source, target, data) in graph.edges(data=True):
            edge_index.append([source, target])
            edge_type = data['edge_type']
            if edge_type == 'temporal':
                edge_attr.append(1)  # 'temporal' 엣지
            elif edge_type == 'spatial':
                edge_attr.append(0)  # 'spatial' 엣지

        # PyTorch 텐서로 변환
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(position_feature=position_feature, id_feature=id_feature,
                    category_feature=category_feature, idx_feature=idx_feature,
                    next_position=next_position, next_id=next_id, next_category=next_category,
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=graph.number_of_nodes())

        return data

    def len(self):
        return self.data_length