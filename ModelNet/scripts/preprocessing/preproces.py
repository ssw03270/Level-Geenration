import pickle
import os
import random
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def select_uniform_points(voxels, num_points=10):
    """Select num_points uniformly distributed points from the voxel grid."""
    random.seed(42)  # 고정된 시드 값
    nz = np.transpose(np.nonzero(voxels))
    selected_points = random.sample(list(nz), k=min(num_points, len(nz)))
    return [tuple(point) for point in selected_points]


def off_to_voxel(off_path, voxel_size=0.05):
    """Convert an OFF file to a voxel grid, normalizing the mesh to fit within a -1 to 1 cube."""
    mesh = trimesh.load(off_path, file_type='off')
    mesh.apply_translation(-mesh.centroid)
    max_extent = np.max(mesh.extents)
    mesh.apply_scale(2 / max_extent)
    voxel_grid = mesh.voxelized(pitch=voxel_size)
    return voxel_grid.matrix

def find_off_files(root_path):
    """Find all .off files within a directory and its subdirectories, separating them into train and test sets."""
    off_files = {'train': [], 'test': []}
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.off'):
                key = 'train' if 'train' in dirpath else 'test'
                off_files[key].append(os.path.join(dirpath, filename))
    return off_files


def bfs_voxel_edges(voxels, start_point, max_edges=256):
    """Perform BFS from a given start point and return only the edges."""
    visited = set([start_point])
    edges = []
    sequence = []
    queue = Queue()
    queue.put(start_point)

    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    while not queue.empty() and len(edges) < max_edges:
        current = queue.get()
        seq = []
        for idx, d in enumerate(directions):
            neighbor = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
            if (0 <= neighbor[0] < voxels.shape[0] and
                    0 <= neighbor[1] < voxels.shape[1] and
                    0 <= neighbor[2] < voxels.shape[2] and
                    voxels[neighbor] and
                    neighbor not in visited):
                visited.add(neighbor)
                edges.append([(current[0] - start_point[0], current[1] - start_point[1], current[2] - start_point[2]),
                              (neighbor[0] - start_point[0], neighbor[1] - start_point[1], neighbor[2] - start_point[2])])  # Store the edge information
                seq.append(idx)
                queue.put(neighbor)
        sequence.append([(current[0] - start_point[0], current[1] - start_point[1], current[2] - start_point[2]), seq])
    return sequence, edges

def plot_voxels(voxels, multiple_voxel_coords=None, title='Voxel Visualization'):
    """Plot voxel data or multiple sets of voxel coordinates."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    max_len = max(voxels.shape)
    ax.set_xlim([0, max_len])
    ax.set_ylim([0, max_len])
    ax.set_zlim([0, max_len])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)

    # 색상을 위한 컬러맵
    colors = plt.cm.jet(np.linspace(0, 1, len(multiple_voxel_coords)))

    if multiple_voxel_coords is not None:
        for voxel_coords, color in zip(multiple_voxel_coords, colors):
            x, y, z = zip(*voxel_coords)
            ax.scatter(x, y, z, c=[color], marker='o')
    else:
        x, y, z = np.nonzero(voxels)
        ax.scatter(x, y, z, zdir='z', c='red')

    plt.show()

def process_single_voxel(file_path, dataset_type, save_dir, visualize=True):
    """Process a single OFF file to voxels, optionally save them, and optionally plot the result."""
    voxels = off_to_voxel(file_path)
    start_points = select_uniform_points(voxels, num_points=20)
    all_edges = []
    all_sequences = []

    multiple_voxel_coords = []

    for start in start_points:
        sequence, edges = bfs_voxel_edges(voxels, start_point=start)
        all_edges.extend(edges)
        all_sequences.extend(sequence)

        coords = [edge[0] for edge in edges]
        multiple_voxel_coords.append(coords)

    if visualize:
        plot_voxels(voxels, multiple_voxel_coords, title='Multiple BFS Voxels')
    else:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        file_name = f'{save_dir}/{dataset_type}/{base_name}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump({'edges': all_edges, 'sequences': all_sequences}, f)

def process_and_save_voxels_parallel(file_paths, dataset_type, save_dir, visualize=True):
    """Process OFF files to voxels in parallel."""
    if not visualize:
        os.makedirs(f'{save_dir}/{dataset_type}', exist_ok=True)

    # 병렬 처리를 위한 ProcessPoolExecutor 사용
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_voxel, file_path, dataset_type, save_dir, visualize) for file_path in file_paths]
        for future in tqdm(as_completed(futures), total=len(file_paths)):
            future.result()  # 결과 처리 혹은 예외 처리

if __name__ == '__main__':
    # Example usage
    root_path = '../../datasets/ModelNet40'
    off_files = find_off_files(root_path)

    print(f"Processing {len(off_files['train'])} train files and {len(off_files['test'])} test files")

    # 저장할 디렉토리 (존재해야 함)
    save_dir = './processed_voxels'

    # Train 데이터 처리 및 저장
    process_and_save_voxels_parallel(off_files['train'], 'train', save_dir, visualize=False)

    # Test 데이터 처리 및 저장
    process_and_save_voxels_parallel(off_files['test'], 'test', save_dir, visualize=False)
