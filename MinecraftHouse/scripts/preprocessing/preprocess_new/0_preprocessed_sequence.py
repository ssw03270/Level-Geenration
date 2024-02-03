from collections import deque
import numpy as np
import pickle
from tqdm import tqdm
import json
import os
import warnings

# DeprecationWarning을 무시하도록 설정
warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == '__main__':
    split_path = '../../../datasets/splits.json'

    with open(split_path, 'r') as file:
        split_dict = json.load(file)

    train_list = split_dict['train']
    val_list = split_dict['val']
    test_list = split_dict['test']

    root_path = '../../../datasets/house_data/houses/'

    for data_type in split_dict:
        block_pos_list = []
        block_id_list = []

        for dir_path in tqdm(split_dict[data_type]):
            folder_path = os.path.join(root_path, dir_path).replace(':', '_')
            placed_path = os.path.join(folder_path, 'placed.json')
            schematic_path = os.path.join(folder_path, 'schematic.npy')

            with open(placed_path, 'r') as file:
                placed_data = json.load(file)
            schematic_data = np.load(schematic_path)

            placed_sequence = []
            for place in placed_data:
                if place[4] == 'P':
                    if place[2] in placed_sequence:
                        placed_sequence.remove(place[2])
                    placed_sequence.append(place[2])

                elif place[4] == 'B':
                    if place[2] in placed_sequence:
                        placed_sequence.remove(place[2])

            placed_sequence = np.array(placed_sequence)
            if len(placed_sequence) < 100:
                continue

            sorted_placed_sequence = placed_sequence[np.lexsort((placed_sequence[:, 2], placed_sequence[:, 1], placed_sequence[:, 0]))]

            schematic_sequence = [[x, y, z] for y in range(schematic_data.shape[0])
                                  for z in range(schematic_data.shape[1])
                                  for x in range(schematic_data.shape[2])
                                  if np.any(schematic_data[y, z, x] >= 1)]
            schematic_sequence = np.array(schematic_sequence)
            sorted_schematic_sequence = schematic_sequence[np.lexsort((schematic_sequence[:, 2], schematic_sequence[:, 1], schematic_sequence[:, 0]))]

            transform = sorted_schematic_sequence[0] - sorted_placed_sequence[0]
            placed_sequence += transform

            max_x = max(placed_sequence[:, 0]) + 1
            max_y = max(placed_sequence[:, 1]) + 1
            max_z = max(placed_sequence[:, 2]) + 1
            placed_sequence = placed_sequence.tolist()
            occupy_grid = np.zeros((max_x, max_y, max_z))
            id_sequence = []

            is_wrong_file = False
            for place in reversed(placed_data):
                if place[4] == 'P':
                    coord = (place[2] + transform).tolist()
                    if coord in placed_sequence:
                        if 0 <= coord[0] < occupy_grid.shape[0] and 0 <= coord[1] < occupy_grid.shape[1] and 0 <= coord[2] < occupy_grid.shape[2]:
                            if occupy_grid[coord[0], coord[1], coord[2]] == 0:
                                occupy_grid[coord[0], coord[1], coord[2]] = 1
                                id_sequence.append(np.asarray(place[3], dtype=np.uint8).astype(np.int64)[0])
                        else:
                            print(dir_path)
                            is_wrong_file = True
                            break
            id_sequence.reverse()

            if is_wrong_file:
                continue

            block_pos_list.append(placed_sequence)
            block_id_list.append(id_sequence)

        with open(f'../../../datasets/preprocessed/preprocessed_{data_type}.pkl', 'wb') as f:
            pickle.dump({'block_pos_list': block_pos_list,
                         'block_id_list': block_id_list}, f)

