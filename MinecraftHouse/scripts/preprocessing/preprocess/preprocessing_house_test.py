import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = '../../../datasets/instance_segmentation_data/preprocessed_training_data_with_terrain.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    schematics = data['schematics']
    annotated_schematics = data['annotated_schematics']
    annotation_lists = data['annotation_list']
    house_names = data['house_names']

    for idx in range(len(schematics)):
        schematic = schematics[idx]
        house_name = house_names[idx]
        name = house_name.replace(':', '_')
        file_path = f'../../datasets/house_data/houses/{name}/placed.json'
        with open(file_path, 'r') as file:
            placed_list = json.load(file)
        print(placed_list)
        json_sequence = []
        height_sequence = []
        for place in placed_list:
            if place[4] == 'P':
                is_able = False
                for dir in [[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]:
                    if [place[2][0] + dir[0], place[2][1] + dir[1], place[2][2] + dir[2]] in json_sequence:
                        is_able = True
                if not is_able:
                    height_sequence.append([place[2][0], place[2][1] - 1, place[2][2]])

                json_sequence.append(place[2])

            elif place[4] == 'B':
                if place[2] in json_sequence:
                    json_sequence.remove(place[2])
                else:
                    height_sequence.append([place[2][0], place[2][1] - 1, place[2][2]])

        height_sequence = np.array(height_sequence)
        schematic_sequence = [[x, y, z] for x in range(schematic.shape[0])
                                        for y in range(schematic.shape[1])
                                        for z in range(schematic.shape[2])
                                        if schematic[x, y, z] >= 1]

        schematic_sequence = np.array(schematic_sequence)
        json_sequence = np.array(json_sequence)

        # print(np.lexsort((json_sequence[:, 2], json_sequence[:, 1], json_sequence[:, 0])))
        sorted_schematic_sequence = schematic_sequence[np.lexsort((schematic_sequence[:, 2], schematic_sequence[:, 1], schematic_sequence[:, 0]))]
        sorted_json_sequence = json_sequence[np.lexsort((json_sequence[:, 2], json_sequence[:, 1], json_sequence[:, 0]))]
        transform = sorted_schematic_sequence[0] - sorted_json_sequence[0]

        print(sorted_schematic_sequence.tolist())
        print((sorted_json_sequence + transform).tolist())
        print((json_sequence + transform).tolist())
        print((height_sequence + transform).tolist())

        json_sequence = (json_sequence + transform).tolist()
        height_sequence = (height_sequence + transform).tolist()

        new_sequence = height_sequence
        for json_seq in json_sequence:
            is_able = False
            for new_seq in new_sequence:
                for dir in [[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]:
                    if [json_seq[0] + dir[0], json_seq[1] + dir[1], json_seq[2] + dir[2]] == new_seq:
                        is_able = True
            if not is_able:
                print(json_seq)
            new_sequence.append(json_seq)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        height_sequence = np.array(height_sequence)
        ax.scatter(height_sequence[:, 0], height_sequence[:, 2], height_sequence[:, 1], c='b')

        max_range = max(schematic.shape)
        ax.set_xlim([0, max_range])
        ax.set_ylim([0, max_range])
        ax.set_zlim([0, max_range])

        # 레이블 및 제목 설정
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('3D Data Visualization')

        plt.show()

        print('--')