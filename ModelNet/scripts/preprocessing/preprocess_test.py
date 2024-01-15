import time
import os
import random
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

def select_uniform_points(voxels, num_points=10):
    """Select num_points uniformly distributed points from the voxel grid."""
    random.seed(327)
    nz = np.transpose(np.nonzero(voxels))
    selected_points = random.sample(list(nz), k=min(num_points, len(nz)))
    return [tuple(point) for point in selected_points]


def off_to_voxel(off_path, voxel_resolution=32):
    """Convert an OFF file to a voxel grid with a fixed number of voxels."""
    # 메시 로드
    mesh = trimesh.load(off_path, file_type='off')
    # 메시를 원점으로 이동
    mesh.apply_translation(-mesh.centroid)
    # Normalize the mesh to fit it inside a unit cube
    bounding_box = mesh.bounding_box.bounds
    scale = 1.0 / max(bounding_box[1] - bounding_box[0])
    translation = -bounding_box[0]
    mesh.apply_translation(translation)
    mesh.apply_scale(scale)
    # 고정된 복셀 크기를 위한 pitch 계산
    pitch = 2 / voxel_resolution
    # 복셀화
    voxel_grid = mesh.voxelized(pitch=pitch)
    return voxel_grid.matrix

def find_off_files(root_path):
    """Find all .off files within a directory and its subdirectories, separating them into train and test sets."""
    off_files = {'train': [], 'test': []}
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.off'):
                key = 'train' if 'train' in dirpath else 'test'
                off_files[key].append(os.path.join(dirpath, filename))

    random.shuffle(off_files['train'])
    random.shuffle(off_files['test'])

    return off_files

def bfs_voxel_edges(voxels, start_point, max_voxels=256):
    """Perform BFS from a given start point and return only the edges with parent indices."""
    visited = set([start_point])
    index_mapping = {start_point: 0}  # Maps nodes to their index in the parent_sequence
    parent_sequence = [-1]  # Start point has no parent, hence -1
    child_sequence = []
    dir_sequence = []
    queue = Queue()
    queue.put((start_point, 0))  # Store node with its index

    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    while not queue.empty() and len(parent_sequence) < max_voxels:
        current, parent_index = queue.get()

        for idx, d in enumerate(directions):
            neighbor = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
            if (0 <= neighbor[0] < voxels.shape[0] and
                    0 <= neighbor[1] < voxels.shape[1] and
                    0 <= neighbor[2] < voxels.shape[2] and
                    voxels[neighbor] and
                    neighbor not in visited) and len(parent_sequence) < max_voxels:
                visited.add(neighbor)
                index_mapping[neighbor] = len(parent_sequence)  # Assign the next index to this neighbor

                parent_sequence.append(parent_index)  # Store parent index
                child_sequence.append([neighbor[0] - start_point[0], neighbor[1] - start_point[1], neighbor[2] - start_point[2]])

                dir_seq = [0, 0, 0, 0, 0, 0]
                dir_seq[idx] = 1
                dir_sequence.append(dir_seq)

                queue.put((neighbor, index_mapping[neighbor]))  # Put the neighbor and its index

    parent_sequence = parent_sequence[1:]
    return parent_sequence, child_sequence, dir_sequence

def plot_voxels(voxels, title='Voxel Visualization'):
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

    x, y, z = np.nonzero(voxels)
    print(len(x))
    ax.scatter(x, y, z, zdir='z', c='red')

    plt.show()

def process_single_voxel(file_path, dataset_type, save_dir, visualize=True):
    """Process a single OFF file to voxels, optionally save them, and optionally plot the result."""
    voxels = off_to_voxel(file_path)
    plot_voxels(voxels)
    print(file_path, len(voxels))

if __name__ == '__main__':
    # Example usage
    root_path = '../../datasets/ModelNet40'
    off_files = find_off_files(root_path)

    print(f"Processing {len(off_files['train'])} train files and {len(off_files['test'])} test files")

    for file_path in off_files['train']:
        process_single_voxel(file_path, '', '', visualize=True)
        time.sleep(0.5)

