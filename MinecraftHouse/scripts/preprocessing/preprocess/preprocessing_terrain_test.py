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

def plot(schematic, idx):
    max_range = max(schematic.shape[0], schematic.shape[2])
    height_map = np.zeros((max_range, max_range))

    for x in range(schematic.shape[0]):
        for z in range(schematic.shape[2]):
            for y in range(schematic.shape[1]):
                if schematic[x, y, z] > 0:
                    height_map[x, z] = y
                    break

    height_map = solve(height_map).astype(int)
    height_list = []

    for x in range(schematic.shape[0]):
        for z in range(schematic.shape[2]):
            height_list.append([x, height_map[x, z], z])
    height_list = np.array(height_list)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(height_list[:, 0], height_list[:, 2], height_list[:, 1], c='b')

    for x in range(schematic.shape[0]):
        for z in range(schematic.shape[2]):
            for y in range(schematic.shape[1]):
                if schematic[x, y, z] > 0:
                    ax.scatter(x, z, y, c='r')

    max_range = max(schematic.shape)
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

    # 레이블 및 제목 설정
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Data Visualization')

    # 그래프를 이미지 파일로 저장
    plt.savefig(f'output/{idx}')

if __name__ == '__main__':
    file_path = '../../../datasets/instance_segmentation_data/preprocessed_training_data.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    schematics = data['schematics']
    annotated_schematics = data['annotated_schematic']
    annotation_list = data['annotation_list']
    house_names = data['house_name']

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(plot, schematic, idx) for idx, schematic in enumerate(schematics)]
        for future in tqdm(as_completed(futures), total=len(schematics)):
            future.result()


