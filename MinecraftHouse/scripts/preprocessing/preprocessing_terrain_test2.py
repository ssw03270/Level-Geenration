import pickle
import json
import numpy as np
import plotly.graph_objs as go
from collections import deque

def create_cube(center, size=1):
    # 정육면체의 중심에서 꼭지점으로의 방향 벡터
    dirs = np.array([[1, 1, -1],
                     [1, -1, -1],
                     [-1, -1, -1],
                     [-1, 1, -1],
                     [1, 1, 1],
                     [1, -1, 1],
                     [-1, -1, 1],
                     [-1, 1, 1]])
    # 꼭지점 계산
    return center + size * 0.5 * dirs

def create_mesh(vertices_list, color):
    faces = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    quad_to_tri = lambda quad: [(quad[0], quad[1], quad[2]), (quad[2], quad[3], quad[0])]
    vertices = []
    i_faces = []
    for i, cube_center in enumerate(vertices_list):
        cube_vertices = create_cube(cube_center)
        vertices.append(cube_vertices)
        for quad in faces:
            i_faces.extend(quad_to_tri(quad + 8 * i))

    vertices = np.concatenate(vertices, axis=0)
    i_faces = np.array(i_faces)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = i_faces[:, 0], i_faces[:, 1], i_faces[:, 2]

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=1)
    return mesh

if __name__ == '__main__':
    file_path = '../../datasets/instance_segmentation_data/preprocessed_training_data_with_terrain.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    house_names = data['house_names']

    data_length = len(house_names)
    for idx in range(data_length):
        house_name = house_names[idx]
        name = house_name.replace(':', '_')
        file_path = f'../../datasets/house_data/houses/{name}/placed.json'

        with open(file_path, 'r') as file:
            placed_list = json.load(file)

        terrain_map = []
        current_map = []

        for jdx in range(len(placed_list)):
            placed_tick = placed_list[jdx][0]
            placed_coord = placed_list[jdx][2]
            placed_block = placed_list[jdx][3]
            placed_action = placed_list[jdx][4]

            if len(current_map) > 0:
                if placed_action == 'P':
                    is_place_able = False

                    for dir in [[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]:
                        try:
                            if [placed_coord[0] + dir[0], placed_coord[1] + dir[1],
                                placed_coord[2] + dir[2]] in current_map:
                                is_place_able = True
                                break
                            if [placed_coord[0] + dir[0], placed_coord[1] + dir[1],
                                placed_coord[2] + dir[2]] in terrain_map:
                                is_place_able = True
                                break
                        except:
                            continue

                    if not is_place_able:
                        terrain_map.append([placed_coord[0], placed_coord[1] - 1, placed_coord[2]])
                    else:
                        current_map.append(placed_coord)

                elif placed_action == 'B':
                    # if not placed_coord in current_map:
                    #     terrain_map.append([placed_coord[0], placed_coord[1], placed_coord[2]])
                    # else:
                    if placed_coord in current_map:
                        current_map.remove(placed_coord)
            else:
                if placed_action == 'P':
                    current_map.append(placed_coord)
                else:
                    terrain_map.append([placed_coord[0], placed_coord[1] - 1, placed_coord[2]])

        terrain_map = np.array(terrain_map, dtype=int)
        current_map = np.array(current_map, dtype=int)

        transform = -1 * np.min(terrain_map, axis=0)
        terrain_map += transform
        current_map += transform

        height_map_shape = np.max(np.array(terrain_map.tolist() + current_map.tolist()), axis=0)
        height_map = np.zeros((height_map_shape[0] + 1, height_map_shape[2] + 1), dtype=int) - 1
        low_height_map = np.zeros((height_map_shape[0] + 1, height_map_shape[2] + 1), dtype=int) + 999
        for map in terrain_map:
            height_map[map[0], map[2]] = max(height_map[map[0], map[2]], map[1])
        for map in current_map:
            low_height_map[map[0], map[2]] = min(low_height_map[map[0], map[2]], map[1])

        queue = deque()
        for x in range(height_map.shape[0]):
            for z in range(height_map.shape[1]):
                if not height_map[x, z] == -1:
                    queue.append((x, z, height_map[x, z]))
        while queue:
            x, z, height = queue.popleft()
            directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for dir in directions:
                try:
                    if height_map[x + dir[0]][z + dir[1]] == -1:
                        mask = height_map != -1
                        height_value = height_map[mask]
                        z_indices, x_indices = np.where(mask)

                        dist_weight = np.sqrt(np.power(z_indices - (z + dir[1]), 2) + np.power(x_indices - (x + dir[0]), 2))
                        dist_weight = dist_weight / sum(dist_weight)
                        height = np.round(np.sum(dist_weight * height_value))

                        if height >= low_height_map[x, z]:
                            height = low_height_map[x, z] - 1

                        height_map[x + dir[0], z + dir[1]] = height

                        queue.append((x + dir[0], z + dir[1], height))
                except:
                    continue
        print(height_map)
        terrain_map = terrain_map.tolist()
        for x in range(height_map.shape[0]):
            for z in range(height_map.shape[1]):
                if [x, height_map[x][z], z] not in terrain_map and height_map[x][z] != -1:
                    terrain_map.append([x, height_map[x][z], z])
        terrain_map = np.array(terrain_map)

        terrain_map = terrain_map[:, [0, 2, 1]]
        current_map = current_map[:, [0, 2, 1]]

        terrain_cubes = create_mesh(terrain_map, color='red')
        current_cubes = create_mesh(current_map, color='blue')

        fig = go.Figure(data=[terrain_cubes, current_cubes])

        fig.update_layout(scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis'),
        ), title='3D Data Visualization')

        fig.show()