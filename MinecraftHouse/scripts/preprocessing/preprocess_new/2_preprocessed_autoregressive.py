import pickle
import numpy as np
from tqdm import tqdm
import time

if __name__ == '__main__':
    root_dir = '../../../datasets/preprocessed/'
    data_types = ['test']

    for data_type in data_types:
        file_path = f'{root_dir}preprocessed_valid_{data_type}.pkl'
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        valid_block_pos_list = data['valid_block_pos_list']
        valid_block_id_list = data['valid_block_id_list']

        local_grids = []
        node_lists = []
        edge_lists = []
        gt_grids = []
        gt_ids = []
        train_mask = []

        file_num = 0
        for pos_list, id_list in zip(tqdm(valid_block_pos_list), valid_block_id_list):
            voxel_grid = np.zeros((max(np.array(pos_list)[:, 0]) + 1,
                                   max(np.array(pos_list)[:, 1]) + 1,
                                   max(np.array(pos_list)[:, 2]) + 1), dtype=np.int32)
            order_grid = np.zeros_like(voxel_grid)

            for idx in range(len(pos_list) - 1):
                x = pos_list[idx][0]
                y = pos_list[idx][1]
                z = pos_list[idx][2]

                voxel_grid[x, y, z] = id_list[idx]
                order_grid[x, y, z] = idx + 1

                # for gt
                size = 3
                grid_size = size * 2 + 1
                gt_grid = np.zeros((grid_size, grid_size, grid_size), dtype=voxel_grid.dtype)
                gt_id = 0

                gt_pos = np.array(pos_list[idx + 1]) - np.array(pos_list[idx]) + np.array([size, size, size])
                if 0 <= gt_pos[0] < grid_size and 0 <= gt_pos[1] < grid_size and 0 <= gt_pos[2] < grid_size:
                    gt_grid[gt_pos[0], gt_pos[1], gt_pos[2]] = 1
                    gt_id = id_list[idx + 1]
                    train_mask.append(True)
                else:
                    train_mask.append(False)

                gt_grids.append(gt_grid)
                gt_ids.append(gt_id)

                # for local grid
                local_grid = np.zeros((grid_size, grid_size, grid_size), dtype=voxel_grid.dtype)

                # Calculate the start and end indices for each dimension, ensuring they are within the array bounds
                x_start, x_end = x - size, x + size + 1
                y_start, y_end = y - size, y + size + 1
                z_start, z_end = z - size, z + size + 1

                # Calculate the offsets for the local_grid
                offset_x_start = max(0, -x_start)
                x_start = max(0, x_start)
                x_end = min(voxel_grid.shape[0], x_end)

                offset_y_start = max(0, -y_start)
                y_start = max(0, y_start)
                y_end = min(voxel_grid.shape[1], y_end)

                offset_z_start = max(0, -z_start)
                z_start = max(0, z_start)
                z_end = min(voxel_grid.shape[2], z_end)

                # Copy the values from the array to the local_grid
                local_grid[offset_x_start:offset_x_start + (x_end - x_start), offset_y_start:offset_y_start + (y_end - y_start), offset_z_start:offset_z_start + (z_end - z_start)] = voxel_grid[x_start:x_end, y_start:y_end, z_start:z_end]
                local_grids.append(local_grid)

                # for graph
                node_list = np.concatenate((np.array(pos_list[:idx + 1]) - np.array(pos_list[idx]),
                                            np.array(id_list[:idx + 1])[:, np.newaxis]), axis=-1)
                node_lists.append(node_list)

                if idx == 0:
                    edge_list = [[0, 0]]
                    edge_lists.append(edge_list)
                else:
                    edge_list = edge_lists[-1].copy()
                    edge_list.append([idx, idx])

                    dirs = [[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]
                    for dir in dirs:
                        if 0 <= x + dir[0] < voxel_grid.shape[0] and 0 <= y + dir[1] < voxel_grid.shape[1] and 0 <= z + dir[2] < voxel_grid.shape[2]:
                            if voxel_grid[x + dir[0], y + dir[1], z + dir[2]] > 0:
                                edge_list.append([idx, order_grid[x + dir[0], y + dir[1], z + dir[2]] - 1])
                                edge_list.append([order_grid[x + dir[0], y + dir[1], z + dir[2]] - 1, idx])

                    edge_lists.append(edge_list)

                if train_mask[-1]:
                    file_path = f'{root_dir}{data_type}/{file_num}.pkl'
                    file_num += 1
                    with open(file_path, 'wb') as f:
                        pickle.dump({'local_grid': local_grid,
                                     'node_list': node_list,
                                     'edge_list': edge_list,
                                     'gt_grid': gt_grid,
                                     'gt_id': gt_id}, f)

        # local_grids = [item for item, mask in zip(local_grids, train_mask) if mask]
        # node_lists = [item for item, mask in zip(node_lists, train_mask) if mask]
        # edge_lists = [item for item, mask in zip(edge_lists, train_mask) if mask]
        # gt_grids = [item for item, mask in zip(gt_grids, train_mask) if mask]
        # gt_ids = [item for item, mask in zip(gt_ids, train_mask) if mask]
        #
        # file_path = f'{root_dir}{data_type}_dataset.pkl'
        # with open(file_path, 'wb') as f:
        #     pickle.dump({'local_grids': local_grids,
        #                  'node_lists': node_lists,
        #                  'edge_lists': edge_lists,
        #                  'gt_grids': gt_grids,
        #                  'gt_ids': gt_ids}, f)