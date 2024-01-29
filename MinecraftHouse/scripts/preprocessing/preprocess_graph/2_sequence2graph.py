import math
import pickle
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two 3D points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

def sub_process(position_sequence, id_sequence, category_sequence, idx):
    n = len(position_sequence)
    distance_cache = defaultdict(dict)

    # Precompute and store distances
    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(position_sequence[i], position_sequence[j])
            distance_cache[i][j] = distance
            distance_cache[j][i] = distance

    for jdx in tqdm(range(1, n)):
        position = position_sequence[:jdx]
        id = id_sequence[:jdx]
        category = category_sequence[:jdx]

        next_position = position_sequence[jdx]
        next_id = id_sequence[jdx]
        next_category = category_sequence[jdx]

        len_position = len(position)

        spatial_edge_set = set()
        for i in range(len_position):
            spatial_edge_set.add((i, i))  # Self-loop
            for j in range(i + 1, len_position):
                if distance_cache[i][j] == 1:
                    spatial_edge_set.add((i, j))
                    spatial_edge_set.add((j, i))

        spatial_edge_list = list(spatial_edge_set)

        temporal_edge_list = [(i, i) for i in range(len_position)]  # Self-loops
        temporal_edge_list += [(i, i + 1) for i in range(len_position - 1)]  # Forward links
        temporal_edge_list += [(i + 1, i) for i in range(len_position - 1)]  # Backward links

        # 결과를 딕셔너리에 저장
        processed_data = {
            'temporal_edge_list': temporal_edge_list,
            'spatial_edge_list': spatial_edge_list,
            'node_list': [position, id, category],
            'gt_list': [next_position, next_id, next_category]
        }

        # 결과를 파일로 저장
        output_file_path = f'../../../datasets/preprocessed_graph/graph_pkl/{idx}_{jdx}.pkl'
        with open(output_file_path, 'wb') as file:
            pickle.dump(processed_data, file)

def data_preprocessing(data):
    position_sequences = data['position_sequences']
    id_sequences = data['id_sequences']
    category_sequences = data['category_sequences']

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(sub_process, position_sequences[idx], id_sequences[idx], category_sequences[idx], idx) for idx in range(len(position_sequences))]

if __name__ == '__main__':
    file_path = '../../../datasets/preprocessed_graph/sequence_datasets.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    data_preprocessing(data)

