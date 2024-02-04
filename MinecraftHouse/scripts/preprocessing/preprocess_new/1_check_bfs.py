import numpy as np
from collections import deque
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
        region_indices = [[x, y, z]]  # region의 시작점 인덱스 추가
        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in [(0, 0, 1), (0, 0, -1), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if is_valid(nx, ny, nz) and not visited[nx, ny, nz] and array[nx, ny, nz] >= 1:
                    queue.append((nx, ny, nz))
                    visited[nx, ny, nz] = True
                    region_indices.append([nx, ny, nz])

        return region_indices

    largest_region_size = 0
    largest_region_indices = []
    for x in range(rows):
        for y in range(cols):
            for z in range(depths):
                if array[x, y, z] >= 1 and not visited[x, y, z]:
                    current_region_indices = bfs(x, y, z)
                    if len(current_region_indices) > largest_region_size:
                        largest_region_size = len(current_region_indices)
                        largest_region_indices = current_region_indices

    return largest_region_indices

if __name__ == '__main__':
    root_dir = '../../../datasets/preprocessed/'
    data_types = ['train', 'val', 'test']

    for data_type in data_types:
        file_path = f'{root_dir}preprocessed_{data_type}.pkl'
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        block_pos_list = data['block_pos_list']
        block_id_list = data['block_id_list']

        for idx in range(len(block_pos_list)):
            pos_list = np.array(block_pos_list[idx])
            id_list = np.array(block_id_list[idx])

            max_x = max(pos_list[:, 0]) + 1
            max_y = max(pos_list[:, 1]) + 1
            max_z = max(pos_list[:, 2]) + 1
            voxel_grid = np.zeros((max_x, max_y, max_z))

            for pos in pos_list:
                voxel_grid[pos[0], pos[1], pos[2]] = 1

            voxel_indices = bfs_3d(voxel_grid)
            for jdx in range(len(pos_list)):
                if