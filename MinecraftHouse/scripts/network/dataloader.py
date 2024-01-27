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
            self.input_path = '../../datasets/preprocessed/sequence_datasets.pkl'
            self.text_path = '../../datasets/preprocessed/text_sequence_datasets.pkl'
        else:
            self.input_path = '../../datasets/preprocessed/sequence_datasets.pkl'
            self.text_path = '../../datasets/preprocessed/text_sequence_datasets.pkl'

        self.text_sequences = []

        self.id_sequences = []
        self.category_sequences = []
        self.direction_sequences = []

        self.next_category_sequences = []
        self.next_id_sequences = []
        self.next_direction_sequences = []

        self.pad_mask_sequences = []

        with open(self.input_path, 'rb') as f:
            data = pickle.load(f)
            direction_sequences = data['direction_sequences']
            id_sequences = data['id_sequences']
            category_sequences = data['category_sequences']

        with open(self.text_path, 'rb') as f:
            data = pickle.load(f)
            text_sequences = data['texts']

        category_values = set()
        for category_sequence in category_sequences:
            for category in category_sequence:
                category_values.add(category)

        # 집합을 리스트로 변환하고 정렬
        self.sorted_category_values = sorted(list(category_values))

        # 정렬된 리스트를 사용하여 인덱스 매핑 생성
        category_to_index = {value: idx + 3 for idx, value in enumerate(self.sorted_category_values)}
        for idx, value in enumerate(self.sorted_category_values):
            print(value, idx + 3)

        for direction_sequence_data, id_sequence_data, category_sequence_data, text_sequence \
                in zip(direction_sequences, id_sequences, category_sequences, text_sequences):
            direction_sequence = []
            id_sequence = []
            category_sequence = []

            data_length = len(direction_sequence_data)
            if data_length > 2040:
                continue

            for direction_data, id_data, category_data \
                    in zip(direction_sequence_data, id_sequence_data, category_sequence_data):
                direction_sequence.append(direction_data)
                id_sequence.append(id_data)
                category_sequence.append(category_data)

            pad_length = 2048 - data_length
            category_sequence = [category_to_index[value] for value in category_sequence]

            next_direction_sequence = direction_sequence + [0, 0, 0] * pad_length
            next_id_sequence = id_sequence + [0] * pad_length
            next_category_sequence = category_sequence + [1] + [0] * (pad_length - 1)

            direction_sequence = [0, 0, 0] + direction_sequence + [0, 0, 0] * (pad_length - 1)
            id_sequence = [0] + id_sequence + [0] * (pad_length - 1)
            category_sequence = [2] + category_sequence + [1] + [0] * (pad_length - 2)

            pad_mask_sequence = [1] * (2048 - (pad_length - 2)) + [0] * (pad_length - 2)

            self.text_sequences.append(text_sequence)

            self.direction_sequences.append(direction_sequence)
            self.id_sequences.append(id_sequence)
            self.category_sequences.append(category_sequence)

            self.next_direction_sequences.append(next_direction_sequence)
            self.next_category_sequences.append(next_category_sequence)
            self.next_id_sequences.append(next_id_sequence)

            self.pad_mask_sequences.append(pad_mask_sequence)

        self.data_length = len(self.direction_sequences)
        print(f'{data_type}: {self.data_length}')

        self.min_val = None
        self.max_val = None
        self.direction_sequences = self.min_max_scaling(np.array(self.direction_sequences))

    def __getitem__(self, idx):
        text_sequence = self.text_sequences[idx]

        direction_sequence = self.direction_sequences[idx]
        id_sequence = self.id_sequences[idx]
        category_sequence = self.category_sequences[idx]

        next_category_sequence = self.next_category_sequences[idx]
        next_id_sequence = self.next_id_sequences[idx]
        next_direction_sequence = self.next_direction_sequences[idx]

        pad_mask_sequence = self.pad_mask_sequences[idx]

        direction_sequence = torch.tensor(direction_sequence, dtype=torch.float32)
        id_sequence = torch.tensor(id_sequence, dtype=torch.long)
        category_sequence = torch.tensor(category_sequence, dtype=torch.long)

        next_direction_sequence = torch.tensor(next_direction_sequence, dtype=torch.float32)
        next_category_sequence = torch.tensor(next_category_sequence, dtype=torch.long)
        next_id_sequence = torch.tensor(next_id_sequence, dtype=torch.long)

        pad_mask_sequence = torch.tensor(pad_mask_sequence, dtype=torch.bool)

        return direction_sequence, id_sequence, category_sequence, next_direction_sequence, \
            next_category_sequence, next_id_sequence, pad_mask_sequence, text_sequence

    def __len__(self):
        return self.data_length

    def min_max_scaling(self, direction_sequence):
        if self.min_val is None:
            self.min_val = np.min(direction_sequence, axis=(0, 1))
            self.max_val = np.max(direction_sequence, axis=(0, 1))

        scaled_direction_sequence = (direction_sequence - self.min_val) / (self.max_val - self.min_val)
        scaled_direction_sequence = scaled_direction_sequence * 2 - 1
        return scaled_direction_sequence