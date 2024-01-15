import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from tqdm import tqdm

class CraftAssistDataset(Dataset):
    def __init__(self, data_type='train'):
        super(CraftAssistDataset, self).__init__()

        self.data_type = data_type

        if data_type == 'train':
            self.file_path = '../../datasets/training_data.pkl'
        else:
            self.file_path = '../../datasets/training_data.pkl'

        self.position_sequences = []
        self.block_id_sequences = []
        self.block_semantic_sequences = []
        self.dir_sequences = []
        self.parent_sequences = []
        self.pad_mask_sequences = []

        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            input_sequences = data['input_sequences']
            output_sequences = data['output_sequences']

        block_semantic_values = set()
        for input_sequence in input_sequences:
            for input_data in input_sequence:
                block_semantic_values.add(input_data[2])
        block_semantic_to_index = {value: idx+3 for idx, value in enumerate(block_semantic_values)}  # 시작 인덱스를 3으로 변경합니다.

        for input_sequence, output_sequence in zip(input_sequences, output_sequences):
            position_sequence = []
            block_id_sequence = []
            block_semantic_sequence = []
            dir_sequence = []
            parent_sequence = []

            data_length = len(input_sequence)
            if data_length > 2040:
                continue

            for input_data, output_data in zip(input_sequence, output_sequence):
                position_sequence.append(input_data[0])
                block_id_sequence.append(input_data[1])
                block_semantic_sequence.append(input_data[2])
                dir_sequence.append(output_data[0])
                parent_sequence.append(output_data[1])

            pad_length = 2048 - 2 - data_length
            position_sequence = [[0, 0, 0]] + position_sequence + [[0, 0, 0]] + [[0, 0, 0]] * pad_length
            block_id_sequence = [0] + block_id_sequence + [0] + [0] * pad_length
            block_semantic_sequence = [0] + [block_semantic_to_index[value] for value in block_semantic_sequence] + [1] + [2] * pad_length
            dir_sequence = [0] + dir_sequence + [0] + [0] * pad_length
            parent_sequence = [0] + parent_sequence + [0] + [0] * pad_length
            pad_mask_sequence = [1] * (2048 - pad_length) + [0] * pad_length

            self.position_sequences.append(position_sequence)
            self.block_id_sequences.append(block_id_sequence)
            self.block_semantic_sequences.append(block_semantic_sequence)
            self.dir_sequences.append(dir_sequence)
            self.parent_sequences.append(parent_sequence)
            self.pad_mask_sequences.append(pad_mask_sequence)

        self.position_sequences = self.min_max_scaling(np.array(self.position_sequences))

        self.data_length = len(self.position_sequences)
        print(f'{data_type}: {self.data_length}')

    def __getitem__(self, idx):
        position_sequence = self.position_sequences[idx]
        block_id_sequence = self.block_id_sequences[idx]
        block_semantic_sequence = self.block_semantic_sequences[idx]
        dir_sequence = self.dir_sequences[idx]
        parent_sequence = self.parent_sequences[idx]
        pad_mask_sequence = self.pad_mask_sequences[idx]

        position_sequence = torch.tensor(position_sequence, dtype=torch.float32)
        block_id_sequence = torch.tensor(block_id_sequence, dtype=torch.long)
        block_semantic_sequence = torch.tensor(block_semantic_sequence, dtype=torch.long)
        dir_sequence = torch.tensor(dir_sequence, dtype=torch.long)
        parent_sequence = torch.tensor(parent_sequence, dtype=torch.long)
        pad_mask_sequence = torch.tensor(pad_mask_sequence, dtype=torch.bool)

        return position_sequence, block_id_sequence, block_semantic_sequence, parent_sequence, dir_sequence, pad_mask_sequence

    def __len__(self):
        return self.data_length

    def min_max_scaling(self, position_sequence):
        min_val = np.min(position_sequence, axis=(0, 1))
        max_val = np.max(position_sequence, axis=(0, 1))

        scaled_position_sequence = (position_sequence - min_val) / (max_val - min_val)
        scaled_position_sequence = scaled_position_sequence * 2 - 1
        return scaled_position_sequence