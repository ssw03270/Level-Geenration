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
            self.input_path = '../../datasets/preprocessed/reorder_sequence_datasets.pkl'
            self.output_path = '../../datasets/preprocessed/output_sequence_datasets.pkl'
            self.text_path = '../../datasets/preprocessed/text_sequence_datasets.pkl'
        else:
            self.input_path = '../../datasets/preprocessed/reorder_sequence_datasets.pkl'
            self.output_path = '../../datasets/preprocessed/output_sequence_datasets.pkl'
            self.text_path = '../../datasets/preprocessed/text_sequence_datasets.pkl'

        self.text_sequences = []

        self.position_sequences = []
        self.id_sequences = []
        self.category_sequences = []

        self.next_category_sequences = []
        self.next_id_sequences = []
        self.next_parent_sequences = []
        self.next_direction_sequences = []

        self.real_position_sequences = []
        self.pad_mask_sequences = []

        with open(self.input_path, 'rb') as f:
            data = pickle.load(f)
            coords_sequences = data['reorder_coords_sequences']
            id_sequences = data['reorder_id_sequences']
            category_sequences = data['reorder_category_sequences']

        with open(self.output_path, 'rb') as f:
            data = pickle.load(f)
            parent_sequences = data['parent_sequences']
            direction_sequences = data['direction_sequences']

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
            print(value, idx+3)

        for coords_sequence_data, id_sequence_data, category_sequence_data, parent_sequence, direction_sequence, text_sequence \
                in zip(coords_sequences, id_sequences, category_sequences, parent_sequences, direction_sequences, text_sequences):
            position_sequence = []
            id_sequence = []
            category_sequence = []

            next_parent_sequence = []
            next_dir_sequence = []

            data_length = len(coords_sequence_data)
            if data_length > 2040:
                continue

            for coords_data, id_data, category_data, parent_data, direction_data, text_data \
                    in zip(coords_sequence_data, id_sequence_data, category_sequence_data, parent_sequence, direction_sequence, text_sequence):
                position_sequence.append(coords_data)
                id_sequence.append(id_data)
                category_sequence.append(category_data)

                next_parent_sequence.append(parent_data)
                next_dir_sequence.append(direction_data)

            pad_length = 2048 - 2 - data_length
            category_sequence = [category_to_index[value] for value in category_sequence]

            next_category_sequence = category_sequence[1:] + [1] + [2] * (pad_length + 2)
            next_id_sequence = id_sequence[1:] + [0] + [0] * (pad_length + 2)
            next_parent_sequence = [0] + next_parent_sequence + [0] + [0] * pad_length
            next_dir_sequence = [0] + next_dir_sequence + [0] + [0] * pad_length

            position_sequence = [[0, 0, 0]] + position_sequence + [[0, 0, 0]] + [[0, 0, 0]] * pad_length
            id_sequence = [0] + id_sequence + [0] + [0] * pad_length
            category_sequence = [0] + category_sequence + [1] + [2] * pad_length

            pad_mask_sequence = [1] * (2048 - pad_length) + [0] * pad_length

            self.text_sequences.append(text_sequence)
            print(len(position_sequence))
            self.position_sequences.append(position_sequence)
            self.id_sequences.append(id_sequence)
            self.category_sequences.append(category_sequence)

            self.next_category_sequences.append(next_category_sequence)
            self.next_id_sequences.append(next_id_sequence)
            self.next_parent_sequences.append(next_parent_sequence)
            self.next_direction_sequences.append(next_dir_sequence)

            self.pad_mask_sequences.append(pad_mask_sequence)

        self.min_val = None
        self.max_val = None
        self.real_position_sequences = self.position_sequences
        self.position_sequences = self.min_max_scaling(np.array(self.position_sequences))

        self.data_length = len(self.position_sequences)
        print(f'{data_type}: {self.data_length}')

    def __getitem__(self, idx):
        text_sequence = self.text_sequences[idx]

        position_sequence = self.position_sequences[idx]
        id_sequence = self.id_sequences[idx]
        category_sequence = self.category_sequences[idx]

        next_category_sequence = self.next_category_sequences[idx]
        next_id_sequence = self.next_id_sequences[idx]
        next_parent_sequence = self.next_parent_sequences[idx]
        next_direction_sequence = self.next_direction_sequences[idx]

        real_position_sequence = self.real_position_sequences[idx]
        pad_mask_sequence = self.pad_mask_sequences[idx]

        position_sequence = torch.tensor(position_sequence, dtype=torch.float32)
        id_sequence = torch.tensor(id_sequence, dtype=torch.long)
        category_sequence = torch.tensor(category_sequence, dtype=torch.long)

        next_category_sequence = torch.tensor(next_category_sequence, dtype=torch.long)
        next_id_sequence = torch.tensor(next_id_sequence, dtype=torch.long)
        next_parent_sequence = torch.tensor(next_parent_sequence, dtype=torch.long)
        next_direction_sequence = torch.tensor(next_direction_sequence, dtype=torch.long)

        real_position_sequence = torch.tensor(real_position_sequence, dtype=torch.long)
        pad_mask_sequence = torch.tensor(pad_mask_sequence, dtype=torch.bool)

        return position_sequence, id_sequence, category_sequence, next_category_sequence, next_id_sequence, \
            next_parent_sequence, next_direction_sequence, real_position_sequence, pad_mask_sequence, \
            text_sequence

    def __len__(self):
        return self.data_length

    def min_max_scaling(self, position_sequence):
        if self.min_val is None:
            self.min_val = np.min(position_sequence, axis=(0, 1))
            self.max_val = np.max(position_sequence, axis=(0, 1))

        scaled_position_sequence = (position_sequence - self.min_val) / (self.max_val - self.min_val)
        scaled_position_sequence = scaled_position_sequence * 2 - 1
        return scaled_position_sequence

    def restore_min_max_scaling(self, scaled_position_sequence):
        # [0, 1] 범위로 되돌리기
        original_scale_sequence = (scaled_position_sequence + 1) / 2
        # 원래의 범위로 복원
        restored_position_sequence = original_scale_sequence * (self.max_val - self.min_val) + self.min_val
        return restored_position_sequence