import pickle
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor, as_completed

def bfs(grid):
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), bool)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    min_val = grid.min()
    min_pos = np.argwhere(grid == min_val)
    queue = deque([tuple(pos) for pos in min_pos])

    changed = False
    while queue:
        x, y = queue.popleft()
        visited[x, y] = True
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                if grid[nx, ny] - grid[x, y] >= 2:
                    grid[nx, ny] = grid[x, y]
                    changed = True
                queue.append((nx, ny))
    return changed

def solve(grid):
    while True:
        if not bfs(grid):
            break
    return grid

if __name__ == '__main__':
    file_path = '../../../datasets/instance_segmentation_data/preprocessed_training_data.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    schematics = data['schematics']
    annotated_schematics = data['annotated_schematic']
    annotation_list = data['annotation_list']
    house_names = data['house_name']

    height_lists = []

    for idx in tqdm(range(len(schematics))):
        max_range = max(schematics[idx].shape[0], schematics[idx].shape[2])
        height_map = np.zeros((max_range, max_range))

        for x in range(schematics[idx].shape[0]):
            for z in range(schematics[idx].shape[2]):
                for y in range(schematics[idx].shape[1]):
                    if schematics[idx][x, y, z] > 0:
                        height_map[x, z] = y
                        break

        height_map = solve(height_map).astype(int)
        height_list = []

        for x in range(schematics[idx].shape[0]):
            for z in range(schematics[idx].shape[2]):
                height_list.append([x, height_map[x, z] - 1, z])
        height_list = np.array(height_list)
        height_lists.append(height_list)

    with open('../../../datasets/instance_segmentation_data/preprocessed_training_data_with_terrain.pkl', 'wb') as f:
        pickle.dump({'schematics': schematics,
                     'annotated_schematics': annotated_schematics,
                     'annotation_list': annotation_list,
                     'house_names': house_names,
                     'height_list': height_lists}, f)

