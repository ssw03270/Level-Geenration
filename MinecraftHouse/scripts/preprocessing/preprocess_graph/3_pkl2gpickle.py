import os
import pickle
import networkx as nx
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

category_dict = {
    "sos": 0,
    "eos": 1,
    "arch": 2,
    "awning": 3,
    "balcony": 4,
    "banister": 5,
    "beam": 6,
    "ceiling": 7,
    "chimney": 8,
    "column": 9,
    "corridor": 10,
    "dome": 11,
    "door": 12,
    "fence": 13,
    "floor": 14,
    "foundation": 15,
    "furnace": 16,
    "furniture": 17,
    "garage": 18,
    "gate": 19,
    "ground": 20,
    "lighting": 21,
    "parapet": 22,
    "plant": 23,
    "pool": 24,
    "road": 25,
    "roof": 26,
    "shutters": 27,
    "stairs": 28,
    "tower": 29,
    "undetermined": 30,
    "vehicle": 31,
    "wall": 32,
    "window": 33
}

def list_files_in_directory(directory):
    """Return a list of file paths in the given directory."""
    files = []
    for filename in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            files.append(file_path)

    return files

if __name__ == '__main__':
    dir_path = '../../../datasets/preprocessed_graph/graph_pkl'
    file_paths = list_files_in_directory(dir_path)

    for file_path in tqdm(file_paths):
        output_file_path = file_path.replace('graph_pkl', 'graph_gpickle')
        output_file_path = output_file_path.replace('.pkl', '')
        # if os.path.exists(output_file_path + '.gpickle'):
        #     continue

        with open(file_path, 'rb') as file:
            data = pickle.load(file)

            temporal_edge_list = data['temporal_edge_list']
            spatial_edge_list = data['spatial_edge_list']
            node_list = data['node_list']
            gt_list = data['gt_list']

        G = nx.Graph()
        # temporal 엣지 추가
        G.add_edges_from(temporal_edge_list, edge_type='temporal')
        # spatial 엣지 추가
        G.add_edges_from(spatial_edge_list, edge_type='spatial')

        n = len(node_list[0])
        # 노드 속성 추가
        for idx in range(n):
            position = node_list[0][idx]
            id = node_list[1][idx]
            category = node_list[2][idx]

            G.nodes[idx]['position'] = [position[0] / 32, position[1] / 32, position[2] / 32]
            G.nodes[idx]['id'] = id
            G.nodes[idx]['category'] = category_dict[category]
            G.nodes[idx]['idx'] = idx

        G.graph['gt'] = {}
        G.graph['gt']['next_position'] = [gt_list[0][0] / 32, gt_list[0][1] / 32, gt_list[0][2] / 32]
        G.graph['gt']['next_id'] = gt_list[1]
        G.graph['gt']['next_category'] = category_dict[gt_list[2]]

        with open(f'{output_file_path}.gpickle', 'wb') as f:
            nx.write_gpickle(G, f)