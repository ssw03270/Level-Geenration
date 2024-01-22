from collections import deque
import numpy as np
import pickle
from tqdm import tqdm

def bfs_3d(array):
    rows, cols, depths = array.shape
    visited = np.zeros_like(array, dtype=bool)

    def is_valid(x, y, z):
        return 0 <= x < rows and 0 <= y < cols and 0 <= z < depths

    def bfs(x, y, z):
        queue = deque([(x, y, z)])
        visited[x, y, z] = True
        while queue:
            x, y, z = queue.popleft()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        # 자기 자신을 제외한 모든 방향
                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        nx, ny, nz = x + dx, y + dy, z + dz
                        if is_valid(nx, ny, nz) and not visited[nx, ny, nz] and array[nx, ny, nz] >= 1:
                            queue.append((nx, ny, nz))
                            visited[nx, ny, nz] = True

    regions = 0
    for x in range(rows):
        for y in range(cols):
            for z in range(depths):
                if array[x, y, z] >= 1 and not visited[x, y, z]:
                    bfs(x, y, z)
                    regions += 1

    return regions

if __name__ == '__main__':
    schematics = []
    annotated_schematic = []
    annotation_list = []
    house_name = []

    file_path = '../../../datasets/instance_segmentation_data/training_data.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    for output in tqdm(data):
        house = output[1].astype(int)
        count = bfs_3d(house)

        if count == 1:
            schematics.append(output[0])
            annotated_schematic.append(output[1])
            annotation_list.append(output[2])
            house_name.append(output[3])

    file_path = '../../../datasets/instance_segmentation_data/validation_data.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    for idx in tqdm(range(len(data['schematics']))):
        house = data['annotated_schematic'][idx].astype(int)
        count = bfs_3d(house)

        if count == 1:
            schematics.append(data['schematics'][idx])
            annotated_schematic.append(data['annotated_schematic'][idx])
            annotation_list.append(data['annotation_list'][idx])
            house_name.append(data['house_name'][idx])

    print(len(schematics))
    with open('../../../datasets/preprocessed/annotated_datasets.pkl', 'wb') as f:
        pickle.dump({'schematics': schematics,
                     'annotated_schematics': annotated_schematic,
                     'annotation_lists': annotation_list,
                     'house_names': house_name}, f)
