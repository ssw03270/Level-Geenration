import pickle
import json
import numpy as np
from tqdm import tqdm
import os

def calc_transformation(placed_list):
    json_sequence = []
    for place in placed_list:
        if place[4] == 'P':
            json_sequence.append(place[2])

        elif place[4] == 'B':
            if place[2] in json_sequence:
                json_sequence.remove(place[2])

    schematic_sequence = [[x, y, z] for x in range(schematic.shape[0])
                          for y in range(schematic.shape[1])
                          for z in range(schematic.shape[2])
                          if schematic[x, y, z] >= 1]

    schematic_sequence = np.array(schematic_sequence)
    json_sequence = np.array(json_sequence)

    sorted_schematic_sequence = schematic_sequence[
        np.lexsort((schematic_sequence[:, 2], schematic_sequence[:, 1], schematic_sequence[:, 0]))]
    sorted_json_sequence = json_sequence[np.lexsort((json_sequence[:, 2], json_sequence[:, 1], json_sequence[:, 0]))]
    transform = sorted_schematic_sequence[0] - sorted_json_sequence[0]

    return transform

if __name__ == '__main__':
    file_path = '../../../datasets/instance_segmentation_data/preprocessed_training_data_with_terrain.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    schematics = data['schematics']
    house_names = data['house_names']

    data_length = len(house_names)
    for idx in tqdm(range(data_length)):
        house_name = house_names[idx]
        schematic = schematics[idx]

        name = house_name.replace(':', '_')
        file_path = f'../../datasets/house_data/houses/{name}/placed.json'

        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r') as file:
            placed_list = json.load(file)

        transform = calc_transformation(placed_list)
        generate_sequence = []

        for jdx in range(len(placed_list)):
            placed_tick = placed_list[jdx][0]
            placed_coord = (placed_list[jdx][2] + transform).tolist()
            placed_block = placed_list[jdx][3]
            placed_action = placed_list[jdx][4]

            if placed_action == 'P':
                if schematic.shape[0] > placed_coord[0] and schematic.shape[1] > placed_coord[1] and schematic.shape[2] > placed_coord[2]:
                    if schematic[placed_coord[0], placed_coord[1], placed_coord[2]] > 0:
                        if placed_coord not in generate_sequence:
                            generate_sequence.append(placed_coord)

        schematic_length = 0
        schematic_sequence = []
        for i in range(schematic.shape[0]):
            for j in range(schematic.shape[1]):
                for k in range(schematic.shape[2]):
                    if schematic[i, j, k] > 0:
                        schematic_length += 1
                        schematic_sequence.append([i, j, k])

        # print(sorted(generate_sequence))
        # print(schematic_sequence)
        if len(generate_sequence) != schematic_length:
            print(house_name)
            continue

