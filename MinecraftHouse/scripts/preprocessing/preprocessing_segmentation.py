from collections import deque
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from plot_house import plot

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

def check_file(data, idx):
    exception_list = [36, 68, 99, 170, 207, 242, 274, 283, 290, 301, 302, 318, 352, 358, 378, 385, 450, 453, 463, 469,
                      508, 512, 526, 579, 581, 658, 907, 913, 956, 971, 987, 996, 1005, 1132, 1150, 1171, 1183, 1208,
                      1234, 1271, 1344, 1363, 1520, 1523, 1526, 1527, 1536, 1630, 1662, 1727, 1805, 1843]

    if idx not in exception_list:
        house = data[1].astype(int)
        count = bfs_3d(house)
        if count == 1:
            # plot(d, folder_name=f'output', file_name=f'{idx}')
            return 1, data

    return 0, None

if __name__ == '__main__':
    file_path = '../../datasets/instance_segmentation_data/training_data.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    schematics = []
    annotated_schematic = []
    annotation_list = []
    house_name = []

    true_files = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(check_file, data[idx], idx) for idx in range(len(data))]
        for future in tqdm(as_completed(futures), total=len(data)):
            output1, output2 = future.result()
            true_files += output1

            if output2 is not None:
                schematics.append(output2[0])
                annotated_schematic.append(output2[1])
                annotation_list.append(output2[2])
                house_name.append(output2[3])

    print(true_files)
    with open('../../datasets/instance_segmentation_data/preprocessed_training_data.pkl', 'wb') as f:
        pickle.dump({'schematics': schematics,
                     'annotated_schematic': annotated_schematic,
                     'annotation_list': annotation_list,
                     'house_name': house_name}, f)