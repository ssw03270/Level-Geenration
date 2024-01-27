import pickle
import json
import numpy as np
from tqdm import tqdm

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
    file_path = '../../../datasets/preprocessed_graph/annotated_datasets.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    schematics = data['schematics']
    house_names = data['house_names']
    annotated_schematics = data['annotated_schematics']
    annotation_lists = data['annotation_lists']

    coords_sequences = []
    category_sequences = []
    id_sequences = []

    data_length = len(house_names)
    for idx in tqdm(range(data_length)):
        house_name = house_names[idx]
        schematic = schematics[idx]
        annotated_schematic = annotated_schematics[idx]
        annotation_list = annotation_lists[idx]

        name = house_name.replace(':', '_')
        file_path = f'../../../datasets/house_data/houses/{name}/placed.json'

        try:
            with open(file_path, 'r') as file:
                placed_list = json.load(file)
        except:
            continue

        transform = calc_transformation(placed_list)
        coords_sequence = []
        voxel_map = np.zeros((100, 100, 100))

        for jdx in range(len(placed_list)):
            placed_tick = placed_list[jdx][0]
            placed_coord = (placed_list[jdx][2] + transform + 10).tolist()
            placed_block = placed_list[jdx][3]
            placed_action = placed_list[jdx][4]

            if placed_action == 'P':
                offsets = [(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (-1, 0, 0), (1, 0, 0)]
                is_place_able = False
                for dx, dy, dz in offsets:
                    x = placed_coord[0] + dx
                    y = placed_coord[1] + dy
                    z = placed_coord[2] + dz

                    if voxel_map[x, y, z] >= 1:
                        is_place_able = True
                        break

                voxel_map[placed_coord[0], placed_coord[1], placed_coord[2]] = 1
                if not is_place_able:
                    if voxel_map[placed_coord[0], placed_coord[1] - 1, placed_coord[2]] == 1:
                        print(house_name)
                    voxel_map[placed_coord[0], placed_coord[1] - 1, placed_coord[2]] = 2

            elif placed_action == 'B':
                is_break_able = False

                try:
                    if voxel_map[placed_coord[0], placed_coord[1], placed_coord[2]] >= 1:
                        is_break_able = True
                except:
                    continue

                voxel_map[placed_coord[0], placed_coord[1], placed_coord[2]] = 0
                # if not is_break_able:
                #     print(house_name, jdx, 'is not break able')